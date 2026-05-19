using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Models.Unigram;

/// <summary>
/// Unigram 分词的格数据结构。
/// 将句子表示为可能分词的有向无环图。
/// 实现 Viterbi、nbest、sample 和 populate_marginal 算法，
/// matching the Rust SentencePiece Lattice implementation.
///
/// 性能优化：Node 使用 struct + 索引引用，减少堆分配和 GC 压力。
/// </summary>
public sealed class Lattice
{
    private string _sentence;
    private int _len;
    private readonly List<Node> _nodes;
    private List<int>[] _beginNodes;  // 存储 node 索引
    private List<int>[] _endNodes;    // 存储 node 索引
    private int _bosId;
    private int _eosId;

    // 线程级 Lattice 池，避免每次 BuildLattice 分配新实例
    private static readonly ThreadLocal<Stack<Lattice>> s_pool = new(() => new());
    // 线程级 List<int>[] 池，避免每次 BuildLattice 分配 2N 个 List
    private static readonly ThreadLocal<Stack<List<int>[]>> s_beginNodesPool = new(() => new());
    private static readonly ThreadLocal<Stack<List<int>[]>> s_endNodesPool = new(() => new());

    /// <summary>
    /// Lattice 中的节点，表示一个可能的 token。
    /// 使用 struct 减少堆分配，使用索引替代引用。
    /// </summary>
    public struct Node
    {
        /// <summary>词汇表 ID。</summary>
        public int Id { get; set; }

        /// <summary>节点在 _nodes 列表中的索引。</summary>
        public int NodeId { get; set; }

        /// <summary>句子中的起始位置（char 索引）。</summary>
        public int Pos { get; set; }

        /// <summary>token 长度（char 数）。</summary>
        public int Length { get; set; }

        /// <summary>Viterbi 回溯中最佳路径的前一个节点索引（-1 表示无）。</summary>
        public int PrevIndex { get; set; }

        /// <summary>累积到此节点的最佳分数。</summary>
        public double BacktraceScore { get; set; }

        /// <summary>此 token 的分数（log 概率）。</summary>
        public double Score { get; set; }
    }

    /// <summary>
    /// nbest 束搜索的假设。
    /// </summary>
    private sealed class Hypothesis : IComparable<Hypothesis>
    {
        public int NodeIndex;  // 改为索引
        public Hypothesis? Next;
        public double Fx;
        public double Gx;

        public Hypothesis(int nodeIndex, Hypothesis? next, double fx, double gx)
        {
            NodeIndex = nodeIndex;
            Next = next;
            Fx = fx;
            Gx = gx;
        }

        public int CompareTo(Hypothesis? other)
        {
            if (other is null) return 1;
            return Fx.CompareTo(other.Fx); // min-heap
        }
    }

    public Lattice(string sentence, int bosId, int eosId)
    {
        _sentence = sentence;
        _len = sentence.Length;
        _bosId = bosId;
        _eosId = eosId;

        const int kReservedNodeSize = 16;
        _nodes = new List<Node>(kReservedNodeSize + 2);

        // 池化 List 数组：复用 List 实例，仅 Clear 清空内容
        var beginPool = s_beginNodesPool.Value!;
        var endPool = s_endNodesPool.Value!;

        _beginNodes = beginPool.Count > 0 ? beginPool.Pop() : new List<int>[_len + 1];
        _endNodes = endPool.Count > 0 ? endPool.Pop() : new List<int>[_len + 1];

        // 确保数组足够大
        if (_beginNodes.Length < _len + 1)
            _beginNodes = new List<int>[_len + 1];
        if (_endNodes.Length < _len + 1)
            _endNodes = new List<int>[_len + 1];

        for (int i = 0; i <= _len; i++)
        {
            if (_beginNodes[i] is null)
                _beginNodes[i] = new List<int>(kReservedNodeSize);
            else
                _beginNodes[i].Clear();

            if (_endNodes[i] is null)
                _endNodes[i] = new List<int>(kReservedNodeSize);
            else
                _endNodes[i].Clear();
        }

        // 创建 BOS 和 EOS 哨兵节点
        int bosIdx = 0;
        int eosIdx = 1;
        _nodes.Add(new Node { Id = bosId, NodeId = 0, Pos = 0, Length = 0, Score = 0.0, PrevIndex = -1, BacktraceScore = 0.0 });
        _nodes.Add(new Node { Id = eosId, NodeId = 1, Pos = _len, Length = 0, Score = 0.0, PrevIndex = -1, BacktraceScore = 0.0 });

        _endNodes[0].Add(bosIdx);
        _beginNodes[_len].Add(eosIdx);
    }

    public string Sentence => _sentence;
    public int Len => _len;

    /// <summary>
    /// 从线程池获取一个 Lattice 实例（复用内部 buffer）。
    /// 与 SentencePiece C++ Lattice::SetSentence 对齐。
    /// </summary>
    public static Lattice Rent(string sentence, int bosId, int eosId)
    {
        var pool = s_pool.Value!;
        if (pool.Count > 0)
        {
            var lattice = pool.Pop();
            lattice.Reset(sentence, bosId, eosId);
            return lattice;
        }
        return new Lattice(sentence, bosId, eosId);
    }

    /// <summary>
    /// 将 Lattice 实例归还到线程池。
    /// </summary>
    public static void Return(Lattice lattice)
    {
        var pool = s_pool.Value!;
        if (pool.Count < 64) // 限制池大小，防止内存膨胀
            pool.Push(lattice);
    }

    /// <summary>
    /// 复用 Lattice 内部 buffer，设置新句子。
    /// 与 SentencePiece C++ Lattice::SetSentence 对齐：
    /// - _nodes.Clear() 复用 List 容量（Clear 不释放内存）
    /// - _beginNodes/_endNodes 复用已有 List 实例
    /// - 只在数组不够大时才重新分配
    /// </summary>
    private void Reset(string sentence, int bosId, int eosId)
    {
        _sentence = sentence;
        _len = sentence.Length;
        _bosId = bosId;
        _eosId = eosId;

        // 复用 _nodes List（Clear 不释放内存，只重置 Count）
        _nodes.Clear();

        // 复用 _beginNodes/_endNodes 数组
        if (_beginNodes.Length < _len + 1)
            _beginNodes = new List<int>[_len + 1];
        if (_endNodes.Length < _len + 1)
            _endNodes = new List<int>[_len + 1];

        for (int i = 0; i <= _len; i++)
        {
            if (_beginNodes[i] is null)
                _beginNodes[i] = new List<int>(16);
            else
                _beginNodes[i].Clear();

            if (_endNodes[i] is null)
                _endNodes[i] = new List<int>(16);
            else
                _endNodes[i].Clear();
        }

        // 创建 BOS 和 EOS 哨兵节点
        _nodes.Add(new Node { Id = bosId, NodeId = 0, Pos = 0, Length = 0, Score = 0.0, PrevIndex = -1, BacktraceScore = 0.0 });
        _nodes.Add(new Node { Id = eosId, NodeId = 1, Pos = _len, Length = 0, Score = 0.0, PrevIndex = -1, BacktraceScore = 0.0 });

        _endNodes[0].Add(0);
        _beginNodes[_len].Add(1);
    }

    /// <summary>获取节点列表的只读视图。</summary>
    public IReadOnlyList<Node> Nodes => _nodes;

    /// <summary>获取指定索引的节点引用。</summary>
    public ref Node GetNode(int index) => ref CollectionsMarshal.AsSpan(_nodes)[index];

    /// <summary>获取 BOS 节点。</summary>
    public Node BosNode => _nodes[0];

    /// <summary>获取 EOS 节点。</summary>
    public Node EosNode => _nodes[1];

    /// <summary>
    /// 在给定位置插入 token。
    /// </summary>
    public void Insert(int pos, int length, double score, int id)
    {
        int nodeId = _nodes.Count;
        var node = new Node
        {
            Id = id,
            NodeId = nodeId,
            Pos = pos,
            Length = length,
            Score = score,
            PrevIndex = -1,
            BacktraceScore = 0.0
        };

        _nodes.Add(node);
        _beginNodes[pos].Add(nodeId);
        _endNodes[pos + length].Add(nodeId);
    }

    /// <summary>
    /// 获取给定节点的片段（子字符串）。
    /// </summary>
    public string Piece(Node node)
    {
        return _sentence.Substring(node.Pos, node.Length);
    }

    /// <summary>
    /// 获取给定节点的片段作为 ReadOnlySpan，避免分配。
    /// </summary>
    public ReadOnlySpan<char> PieceAsSpan(Node node)
    {
        return _sentence.AsSpan(node.Pos, node.Length);
    }

    /// <summary>
    /// 获取从位置 n 开始的表面字符串。
    /// </summary>
    public string Surface(int n)
    {
        if (n >= _len) return "";
        return _sentence.Substring(n);
    }

    /// <summary>
    /// 应用 Viterbi 算法找到最佳分词路径。
    /// 无候选节点时跳过该位置继续处理（按 Rune 长度推进）。
    /// </summary>
    public IReadOnlyList<Node> Viterbi()
    {
        var nodesSpan = CollectionsMarshal.AsSpan(_nodes);

        int pos = 0;
        while (pos <= _len)
        {
            if (_beginNodes[pos].Count == 0)
            {
                // 无候选节点，跳过该位置（按 Rune 长度推进）
                if (pos < _len)
                {
                    if (Rune.TryGetRuneAt(_sentence, pos, out var skipRune))
                        pos += skipRune.Utf16SequenceLength;
                    else
                        pos++;
                    continue;
                }
                else
                {
                    break;
                }
            }

            for (int ri = 0; ri < _beginNodes[pos].Count; ri++)
            {
                int rnodeIdx = _beginNodes[pos][ri];
                ref var rnode = ref nodesSpan[rnodeIdx];
                rnode.PrevIndex = -1;
                double bestScore = 0.0;
                int bestNodeIdx = -1;

                for (int li = 0; li < _endNodes[pos].Count; li++)
                {
                    int lnodeIdx = _endNodes[pos][li];
                    double score = nodesSpan[lnodeIdx].BacktraceScore + rnode.Score;
                    if (bestNodeIdx == -1 || score > bestScore)
                    {
                        bestNodeIdx = lnodeIdx;
                        bestScore = score;
                    }
                }

                if (bestNodeIdx >= 0)
                {
                    rnode.PrevIndex = bestNodeIdx;
                    rnode.BacktraceScore = bestScore;
                }
                else
                {
                    return [];
                }
            }

            // Advance to next position with content
            if (pos < _len)
            {
                if (Rune.TryGetRuneAt(_sentence, pos, out var r))
                    pos += r.Utf16SequenceLength;
                else
                    pos++;
            }
            else
            {
                break;
            }
        }

        // Reconstruct path
        var rootIdx = _beginNodes[_len][0];
        if (_nodes[rootIdx].PrevIndex < 0) return [];

        var results = new List<Node>();
        int nodeIdx = _nodes[rootIdx].PrevIndex;
        while (nodeIdx >= 0 && _nodes[nodeIdx].PrevIndex >= 0)
        {
            results.Add(_nodes[nodeIdx]);
            nodeIdx = _nodes[nodeIdx].PrevIndex;
        }
        results.Reverse();
        return results;
    }

    /// <summary>
    /// 从 Viterbi 路径获取 token。
    /// </summary>
    public IReadOnlyList<string> Tokens()
    {
        return Viterbi().Select(Piece).ToList();
    }

    /// <summary>
    /// 使用束搜索找到前 n 个分词路径。
    /// </summary>
    public IReadOnlyList<IReadOnlyList<Node>> Nbest(int n)
    {
        if (n == 0) return [];
        if (n == 1) return [Viterbi()];

        // 先通过 Viterbi 填充回溯分数
        Viterbi();

        var nodesSpan = CollectionsMarshal.AsSpan(_nodes);
        var hypotheses = new List<IReadOnlyList<Node>>();
        var agenda = new PriorityQueue<Hypothesis, double>(Comparer<double>.Create((a, b) => b.CompareTo(a)));
        int eosIdx = 1;

        agenda.Enqueue(
            new Hypothesis(eosIdx, null, nodesSpan[eosIdx].BacktraceScore, nodesSpan[eosIdx].BacktraceScore),
            nodesSpan[eosIdx].BacktraceScore);

        const int kMaxAgendaSize = 100_000;
        const int kMinAgendaSize = 512;

        while (agenda.Count > 0)
        {
            var top = agenda.Dequeue();
            int nodeIdx = top.NodeIndex;

            if (nodeIdx == 0) // BosNode index
            {
                var hypothesis = new List<Node>();
                var next = top.Next;
                while (next?.Next is not null)
                {
                    hypothesis.Add(_nodes[next.NodeIndex]);
                    next = next.Next;
                }
                hypotheses.Add(hypothesis);
                if (hypotheses.Count == n)
                    return hypotheses;
            }
            else
            {
                for (int li = 0; li < _endNodes[_nodes[nodeIdx].Pos].Count; li++)
                {
                    int lnodeIdx = _endNodes[_nodes[nodeIdx].Pos][li];
                    double fx = nodesSpan[lnodeIdx].BacktraceScore + top.Gx;
                    double gx = nodesSpan[lnodeIdx].Score + top.Gx;
                    var hyp = new Hypothesis(lnodeIdx, top, fx, gx);
                    agenda.Enqueue(hyp, fx);
                }

                if (agenda.Count > kMaxAgendaSize)
                {
                    var newAgenda = new PriorityQueue<Hypothesis, double>(
                        Comparer<double>.Create((a, b) => b.CompareTo(a)));
                    int keep = Math.Min(kMinAgendaSize, n * 10);
                    for (int i = 0; i < keep && agenda.Count > 0; i++)
                    {
                        var hypothesis = agenda.Dequeue();
                        newAgenda.Enqueue(hypothesis, hypothesis.Fx);
                    }
                    agenda = newAgenda;
                }
            }
        }

        return hypotheses;
    }

    /// <summary>
    /// 获取 nbest token 字符串。
    /// </summary>
    public IReadOnlyList<IReadOnlyList<string>> NbestTokens(int n)
    {
        return Nbest(n).Select(path => (IReadOnlyList<string>)path.Select(Piece).ToList()).ToList();
    }

    /// <summary>
    /// 从所有可能的分词中按分数比例采样路径。
    /// 使用温度 theta 的 softmax。
    /// </summary>
    public IReadOnlyList<Node> Sample(double theta)
    {
        if (_len == 0) return [];

        var nodesSpan = CollectionsMarshal.AsSpan(_nodes);
        int nodeCount = _nodes.Count;
        var alpha = ArrayPool<double>.Shared.Rent(nodeCount);
        Array.Clear(alpha, 0, nodeCount);

        try
        {
            for (int pos = 0; pos <= _len; pos++)
            {
                for (int ri = 0; ri < _beginNodes[pos].Count; ri++)
                {
                    int rnodeIdx = _beginNodes[pos][ri];
                    for (int li = 0; li < _endNodes[pos].Count; li++)
                    {
                        int lnodeIdx = _endNodes[pos][li];
                        alpha[rnodeIdx] = RuneHelpers.LogSumExp(
                            alpha[rnodeIdx],
                            theta * (nodesSpan[lnodeIdx].Score + alpha[lnodeIdx]),
                            lnodeIdx == _endNodes[pos][0]);
                    }
                }
            }

            var rng = Random.Shared;
            var results = new List<Node>();
            var probs = new List<double>();
            double z = alpha[1]; // EosNode index
            int currentNodeIdx = 1; // EosNode
            const int kMaxIterations = 10_000; // 安全阀，防止无限循环
            int iterations = 0;

            while (true)
            {
                if (++iterations > kMaxIterations)
                    break; // 安全阀：超过最大迭代次数，返回当前已收集的结果

                probs.Clear();
                int pos = nodesSpan[currentNodeIdx].Pos;
                for (int li = 0; li < _endNodes[pos].Count; li++)
                {
                    int lnodeIdx = _endNodes[pos][li];
                    probs.Add(Math.Exp(alpha[lnodeIdx] + theta * nodesSpan[lnodeIdx].Score - z));
                }

                if (probs.Count == 0)
                    break; // 无候选节点，终止采样

                double total = 0;
                foreach (var p in probs) total += p;
                double r = rng.NextDouble() * total;
                double cumulative = 0;
                int selectedIndex = 0;
                for (int i = 0; i < probs.Count; i++)
                {
                    cumulative += probs[i];
                    if (r <= cumulative)
                    {
                        selectedIndex = i;
                        break;
                    }
                }

                currentNodeIdx = _endNodes[pos][selectedIndex];
                if (currentNodeIdx == 0) break; // BosNode
                z = alpha[currentNodeIdx];
                results.Add(_nodes[currentNodeIdx]);
            }

            results.Reverse();
            return results;
        }
        finally
        {
            ArrayPool<double>.Shared.Return(alpha);
        }
    }

    /// <summary>
    /// 采样并返回 token 字符串。
    /// </summary>
    public IReadOnlyList<string> SampleToken(double theta)
    {
        return Sample(theta).Select(Piece).ToList();
    }

    /// <summary>
    /// 计算边际概率的前向-后向算法。
    /// 用于训练中计算期望 token 计数。
    /// 返回 log(Z)，其中 Z 是配分函数。
    /// </summary>
    public double PopulateMarginal(double freq, double[] expected)
    {
        var nodesSpan = CollectionsMarshal.AsSpan(_nodes);
        int nodeCount = _nodes.Count;
        var alpha = ArrayPool<double>.Shared.Rent(nodeCount);
        var beta = ArrayPool<double>.Shared.Rent(nodeCount);
        Array.Clear(alpha, 0, nodeCount);
        Array.Clear(beta, 0, nodeCount);

        try
        {
            // 前向传播
            for (int pos = 0; pos <= _len; pos++)
            {
                for (int ri = 0; ri < _beginNodes[pos].Count; ri++)
                {
                    int rnodeIdx = _beginNodes[pos][ri];
                    for (int li = 0; li < _endNodes[pos].Count; li++)
                    {
                        int lnodeIdx = _endNodes[pos][li];
                        alpha[rnodeIdx] = RuneHelpers.LogSumExp(
                            alpha[rnodeIdx],
                            nodesSpan[lnodeIdx].Score + alpha[lnodeIdx],
                            lnodeIdx == _endNodes[pos][0]);
                    }
                }
            }

            // Backward pass
            for (int pos = _len; pos >= 0; pos--)
            {
                for (int li = 0; li < _endNodes[pos].Count; li++)
                {
                    int lnodeIdx = _endNodes[pos][li];
                    for (int ri = 0; ri < _beginNodes[pos].Count; ri++)
                    {
                        int rnodeIdx = _beginNodes[pos][ri];
                        beta[lnodeIdx] = RuneHelpers.LogSumExp(
                            beta[lnodeIdx],
                            nodesSpan[rnodeIdx].Score + beta[rnodeIdx],
                            rnodeIdx == _beginNodes[pos][0]);
                    }
                }
            }

            double z = alpha[1]; // EosNode

            for (int pos = 0; pos < _len; pos++)
            {
                for (int ni = 0; ni < _beginNodes[pos].Count; ni++)
                {
                    int nodeIdx = _beginNodes[pos][ni];
                    int id = nodesSpan[nodeIdx].Id;
                    double a = alpha[nodeIdx];
                    double b = beta[nodeIdx];
                    double total = a + nodesSpan[nodeIdx].Score + b - z;
                    double update = freq * Math.Exp(total);
                    expected[id] += update;
                }
            }

            return freq * z;
        }
        finally
        {
            ArrayPool<double>.Shared.Return(alpha);
            ArrayPool<double>.Shared.Return(beta);
        }
    }

    /// <summary>
    /// 重置 Lattice 状态（供外部需要手动重置的场景）。
    /// 训练时应优先使用 Rent/Return 池化 API。
    /// </summary>
    public void Clear()
    {
        _nodes.Clear();
        for (int i = 0; i <= _len; i++)
        {
            _beginNodes[i].Clear();
            _endNodes[i].Clear();
        }
        // Re-add BOS/EOS
        _nodes.Add(new Node { Id = _bosId, NodeId = 0, Pos = 0, Length = 0, Score = 0.0, PrevIndex = -1, BacktraceScore = 0.0 });
        _nodes.Add(new Node { Id = _eosId, NodeId = 1, Pos = _len, Length = 0, Score = 0.0, PrevIndex = -1, BacktraceScore = 0.0 });
        _endNodes[0].Add(0);
        _beginNodes[_len].Add(1);
    }
}
