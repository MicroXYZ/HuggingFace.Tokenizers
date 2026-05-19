using System.Runtime.CompilerServices;

namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// Aho-Corasick 多模式字符串匹配器。
/// 使用紧凑的邻接表（int[]）替代 Dictionary&lt;int,int&gt;，减少内存碎片化。
/// 时间复杂度 O(n + m + z)，其中 n=文本长度，m=模式总长度，z=匹配数。
///
/// 与 Rust daachorse::MatchKind::LeftmostLongest 语义一致。
/// </summary>
internal sealed class AhoCorasickMatcher
{
    // ThreadLocal 池化中间集合，避免每次 FindLeftmostLongest 分配
    private static readonly ThreadLocal<Dictionary<int, (int patternIndex, int length)>> t_bestAtStart = new(() => new());
    private static readonly ThreadLocal<List<KeyValuePair<int, (int patternIndex, int length)>>> t_entries = new(() => new());
    // E8: 池化最终结果列表，避免每次 FindAll/FindLeftmostLongest 分配新 List
    private static readonly ThreadLocal<List<(int, int, int)>> t_results = new(() => new());

    // 紧凑的邻接表：每个状态的转移存储为 (codepoint, nextState) 平行数组
    private readonly List<int[]> _goCp = new();     // 转移的 codepoint
    private readonly List<int[]> _goNext = new();    // 转移的目标状态
    // 失败链接
    private readonly List<int> _fail = new();
    // 输出集：每个状态对应的模式索引列表
    private readonly List<int[]> _output = new();
    // 每个模式的字符长度
    private readonly int[] _patternCharLengths;

    /// <summary>模式数量。</summary>
    public int PatternCount => _patternCharLengths.Length;

    /// <summary>
    /// 从给定模式构建 Aho-Corasick 自动机。
    /// </summary>
    /// <param name="patterns">精确字符串模式列表。</param>
    public AhoCorasickMatcher(IReadOnlyList<string> patterns)
    {
        _patternCharLengths = new int[patterns.Count];
        for (int i = 0; i < patterns.Count; i++)
            _patternCharLengths[i] = patterns[i].Length;

        // 初始化根状态（state 0）
        _goCp.Add([]);
        _goNext.Add([]);
        _fail.Add(0);
        _output.Add([]);

        // 阶段 1：构建 Trie
        for (int pid = 0; pid < patterns.Count; pid++)
        {
            int state = 0;
            foreach (var rune in patterns[pid].EnumerateRunes())
            {
                int cp = rune.Value;
                int next = FindTransition(state, cp);
                if (next == -1)
                {
                    next = _goCp.Count;
                    AddTransition(state, cp, next);
                    _goCp.Add([]);
                    _goNext.Add([]);
                    _fail.Add(0);
                    _output.Add([]);
                }
                state = next;
            }
            // 追加模式索引到输出集
            var output = _output[state];
            Array.Resize(ref output, output.Length + 1);
            output[^1] = pid;
            _output[state] = output;
        }

        // 阶段 2：BFS 构建失败链接
        var queue = new Queue<int>();
        // 根的直接子节点失败链接指向根
        for (int i = 0; i < _goCp[0].Length; i++)
        {
            int child = _goNext[0][i];
            _fail[child] = 0;
            queue.Enqueue(child);
        }

        while (queue.Count > 0)
        {
            int state = queue.Dequeue();
            for (int i = 0; i < _goCp[state].Length; i++)
            {
                int cp = _goCp[state][i];
                int childState = _goNext[state][i];
                queue.Enqueue(childState);

                int failState = _fail[state];
                while (failState != 0 && FindTransition(failState, cp) == -1)
                    failState = _fail[failState];

                int failTarget = FindTransition(failState, cp);
                _fail[childState] = failTarget != -1 ? failTarget : 0;

                // 合并失败状态的输出（后缀匹配）
                if (_output[_fail[childState]].Length > 0)
                {
                    var existing = _output[childState];
                    var suffix = _output[_fail[childState]];
                    int originalLength = existing.Length;
                    Array.Resize(ref existing, originalLength + suffix.Length);
                    Array.Copy(suffix, 0, existing, originalLength, suffix.Length);
                    _output[childState] = existing;
                }
            }
        }

        // BFS 完成后，对每个状态的转移按 codepoint 排序，支持二分查找
        for (int stateIdx = 0; stateIdx < _goCp.Count; stateIdx++)
        {
            var cps = _goCp[stateIdx];
            var nexts = _goNext[stateIdx];
            if (cps.Length <= 1) continue;

            // 冒泡排序（小数组，避免 Array.Sort 的 overhead）
            for (int i = 0; i < cps.Length - 1; i++)
            {
                for (int j = 0; j < cps.Length - 1 - i; j++)
                {
                    if (cps[j] > cps[j + 1])
                    {
                        (cps[j], cps[j + 1]) = (cps[j + 1], cps[j]);
                        (nexts[j], nexts[j + 1]) = (nexts[j + 1], nexts[j]);
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int FindTransition(int state, int cp)
    {
        var cps = _goCp[state];
        // 二分查找：转移已在构造完成后按 codepoint 排序
        int lo = 0, hi = cps.Length - 1;
        while (lo <= hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            if (cps[mid] == cp)
                return _goNext[state][mid];
            if (cps[mid] < cp)
                lo = mid + 1;
            else
                hi = mid - 1;
        }
        return -1;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void AddTransition(int state, int cp, int next)
    {
        var cps = _goCp[state];
        var nexts = _goNext[state];
        Array.Resize(ref cps, cps.Length + 1);
        Array.Resize(ref nexts, nexts.Length + 1);
        cps[^1] = cp;
        nexts[^1] = next;
        _goCp[state] = cps;
        _goNext[state] = nexts;
    }

    /// <summary>
    /// 查找文本中所有精确模式匹配。
    /// 返回 (patternIndex, startCharIndex, charLength) 列表。
    /// 匹配可能重叠，调用方负责重叠解析。
    /// E8 优化：使用 ThreadLocal 池化结果列表，避免每次调用分配新 List。
    /// </summary>
    public List<(int patternIndex, int start, int length)> FindAll(string text)
    {
        var results = t_results.Value!;
        results.Clear();
        int state = 0;
        int charIdx = 0;

        foreach (var rune in text.EnumerateRunes())
        {
            int cp = rune.Value;
            int runeLen = rune.Utf16SequenceLength;

            // 沿失败链接回退直到找到转移或到达根
            while (state != 0 && FindTransition(state, cp) == -1)
                state = _fail[state];

            // 取转移（或停留在根）
            int ns = FindTransition(state, cp);
            state = ns != -1 ? ns : 0;

            // 检查当前状态的输出（包括后缀匹配）
            if (_output[state].Length > 0)
            {
                int endChar = charIdx + runeLen;
                foreach (var pid in _output[state])
                {
                    int startChar = endChar - _patternCharLengths[pid];
                    results.Add((pid, startChar, _patternCharLengths[pid]));
                }
            }

            charIdx += runeLen;
        }

        return results;
    }

    /// <summary>
    /// 左最长匹配：从同一位置开始的多个 pattern 中，只保留最长的匹配。
    /// 与 Rust daachorse::MatchKind::LeftmostLongest 语义一致。
    ///
    /// 实现方式：遍历过程中按 start 位置分组，每组取最长。
    /// 时间复杂度 O(n + m + z)，其中 z 为匹配数。
    /// </summary>
    public List<(int patternIndex, int start, int length)> FindLeftmostLongest(ReadOnlySpan<char> text)
    {
        // 从 ThreadLocal 池获取中间集合
        var bestAtStart = t_bestAtStart.Value!;
        bestAtStart.Clear();
        var entries = t_entries.Value!;
        entries.Clear();

        int state = 0;
        int charIdx = 0;

        foreach (var rune in text.EnumerateRunes())
        {
            int cp = rune.Value;
            int runeLen = rune.Utf16SequenceLength;

            while (state != 0 && FindTransition(state, cp) == -1)
                state = _fail[state];

            int ns = FindTransition(state, cp);
            state = ns != -1 ? ns : 0;

            if (_output[state].Length > 0)
            {
                int endChar = charIdx + runeLen;
                foreach (var pid in _output[state])
                {
                    int startChar = endChar - _patternCharLengths[pid];
                    int len = _patternCharLengths[pid];

                    if (!bestAtStart.TryGetValue(startChar, out var existing) || len > existing.length)
                        bestAtStart[startChar] = (pid, len);
                }
            }

            charIdx += runeLen;
        }

        // 按 start 排序，过滤重叠匹配
        foreach (var kv in bestAtStart)
            entries.Add(kv);
        entries.Sort((a, b) => a.Key.CompareTo(b.Key));

        var results = t_results.Value!;
        results.Clear();
        results.EnsureCapacity(entries.Count);
        int lastEnd = -1;
        foreach (var kv in entries)
        {
            int start = kv.Key;
            int length = kv.Value.length;
            if (start < lastEnd)
                continue;
            results.Add((kv.Value.patternIndex, start, length));
            lastEnd = start + length;
        }
        return results;
    }
}
