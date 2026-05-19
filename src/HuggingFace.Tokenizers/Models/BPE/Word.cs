using System.Buffers;
using System.Runtime.CompilerServices;

namespace HuggingFace.Tokenizers.Models.BPE;

/// <summary>
/// 表示 <see cref="Word"/>.
/// 使用前/后索引的链表结构实现 O(1) 合并操作。
/// </summary>
internal struct Symbol
{
    /// <summary>此符号的 token ID。</summary>
    public uint C;

    /// <summary>链表中前一个符号的索引（-1 表示无）。</summary>
    public int Prev;

    /// <summary>链表中后一个符号的索引（-1 表示无）。</summary>
    public int Next;

    /// <summary>此符号的字节长度。</summary>
    public int Len;

    /// <summary>
    /// 将此符号（左）与另一个符号（右）合并。
    /// 更新 token ID、字节长度和下一个指针。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void MergeWith(in Symbol other, uint newC)
    {
        C = newC;
        Len += other.Len;
        Next = other.Next;
    }
}

/// <summary>
/// 表示 BPE 分词过程中作为符号序列的词。
/// 使用链表架构和基于优先队列的合并（O(n log n)）。
/// 内部使用 ArrayPool 管理 Symbol 数组，减少训练时的 GC 压力。
/// </summary>
internal sealed class Word
{
    // ArrayPool 管理的 Symbol 数组
    private Symbol[] _symbols;
    private int _count;

    // 复用缓冲区，避免每次 Merge 分配新 List
    private List<((uint, uint), int)>? _changesBuf;

    /// <summary>
    /// 合并候选，用于优先队列。
    /// 按排名（升序）排序，然后按位置（升序）打破平局。
    /// pos 升序确保先合并左侧 pair，使合并方向与训练时记录的 merge 方向一致。
    /// 例如训练产生 ".." + "." → "..."，合并时必须先处理左侧 pair 才能匹配。
    /// 存储 NewId 以避免符号状态变化后的过时查找。
    /// </summary>
    private readonly struct MergeCandidate : IComparable<MergeCandidate>
    {
        public readonly int Pos;
        public readonly uint Rank;
        public readonly uint NewId;

        public MergeCandidate(int pos, uint rank, uint newId)
        {
            Pos = pos;
            Rank = rank;
            NewId = newId;
        }

        public int CompareTo(MergeCandidate other)
        {
            int cmp = Rank.CompareTo(other.Rank);
            return cmp != 0 ? cmp : Pos.CompareTo(other.Pos);
        }
    }

    public Word()
    {
        _symbols = ArrayPool<Symbol>.Shared.Rent(8);
        _count = 0;
    }

    public Word(int capacity)
    {
        _symbols = ArrayPool<Symbol>.Shared.Rent(Math.Max(capacity, 8));
        _count = 0;
    }

    /// <summary>
    /// 归还内部数组到 ArrayPool。调用后此 Word 实例不可再使用。
    /// </summary>
    public void Return()
    {
        if (_symbols is not null)
        {
            ArrayPool<Symbol>.Shared.Return(_symbols);
            _symbols = null!;
            _count = 0;
        }
    }

    /// <summary>
    /// 获取此词中的符号数量。
    /// </summary>
    public int Length => _count;

    /// <summary>
    /// 确保容量足够，不足时从 ArrayPool 扩容。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EnsureCapacity(int required)
    {
        if (required <= _symbols.Length) return;
        int newSize = Math.Max(_symbols.Length * 2, required);
        var newArray = ArrayPool<Symbol>.Shared.Rent(newSize);
        Array.Copy(_symbols, newArray, _count);
        ArrayPool<Symbol>.Shared.Return(_symbols);
        _symbols = newArray;
    }

    /// <summary>
    /// 添加具有给定 token ID 和字节长度的符号。
    /// 更新前/后链表指针。
    /// 内部使用字节长度（与 Rust 一致）；字符转换
    /// happens at the output stage via OffsetType.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Add(uint tokenId, int byteLen)
    {
        EnsureCapacity(_count + 1);

        int prev;
        int next = -1;

        if (_count > 0)
        {
            int lastIdx = _count - 1;
            _symbols[lastIdx].Next = _count;
            prev = lastIdx;
        }
        else
        {
            prev = -1;
        }

        _symbols[_count] = new Symbol
        {
            C = tokenId,
            Prev = prev,
            Next = next,
            Len = byteLen
        };
        _count++;
    }

    /// <summary>
    /// 在词中执行 BPE 合并操作。
    /// 使用标记+压缩策略一次遍历完成，避免 Insert/RemoveAt 的 O(n²)。
    /// 注意：Prev/Next 字段仅用于记录相邻关系，不维护链表指针。
    /// </summary>
    public List<((uint, uint), int)> Merge(uint c1, uint c2, uint replacement, int maxLength)
    {
        // 复用缓冲区，避免每次 Merge 分配新 List
        var changes = _changesBuf ??= new List<((uint, uint), int)>();
        changes.Clear();
        var symbols = _symbols;
        int count = _count;

        // 标记+压缩策略：一次遍历完成合并，避免 Insert/RemoveAt 的 O(n²)
        int writeIdx = 0;
        for (int i = 0; i < count;)
        {
            // 检查是否找到 (c1, c2) 对
            if (symbols[i].C == c1 && i + 1 < count && symbols[i + 1].C == c2)
            {
                var first = symbols[i];
                var second = symbols[i + 1];

                var newSymbol = new Symbol
                {
                    C = replacement,
                    Prev = first.Prev,
                    Next = second.Next,
                    Len = first.Len + second.Len
                };

                // 记录左侧相邻对的变化
                if (writeIdx > 0)
                {
                    var prev = symbols[writeIdx - 1];
                    changes.Add(((prev.C, first.C), -1));
                    if (prev.Len + newSymbol.Len < maxLength)
                        changes.Add(((prev.C, replacement), 1));
                }

                // 写入合并后的符号
                symbols[writeIdx] = newSymbol;

                // 记录右侧相邻对的变化
                if (i + 2 < count)
                {
                    changes.Add(((second.C, symbols[i + 2].C), -1));
                    if (symbols[i + 2].Len + newSymbol.Len < maxLength)
                        changes.Add(((replacement, symbols[i + 2].C), 1));
                }

                writeIdx++;
                i += 2; // 跳过已合并的两个符号
            }
            else
            {
                // 未匹配，保留原符号
                if (writeIdx != i)
                    symbols[writeIdx] = symbols[i];
                writeIdx++;
                i++;
            }
        }

        // 更新计数（替代 RemoveRange）
        _count = writeIdx;

        return changes;
    }

    /// <summary>
    /// 使用优先队列合并所有对（O(n log n) 算法）。
    /// 与 Rust BPE Word::merge_all 实现一致。
    /// </summary>
    /// <param name="merges">Map from (token_id_left, token_id_right) to (rank, new_token_id).</param>
    /// <param name="dropout">Optional dropout probability. 当 set, randomly skips merges.</param>
    // ThreadStatic 优先队列复用，避免每次 MergeAll 分配（与 Microsoft.ML.Tokenizers 对齐）
    [ThreadStatic]
    private static PriorityQueue<MergeCandidate, MergeCandidate>? t_queue;
    [ThreadStatic]
    private static List<MergeCandidate>? t_skip;

    public void MergeAll(IReadOnlyDictionary<(uint, uint), (uint Rank, uint NewId)> merges, float? dropout)
    {
        // 复用线程级优先队列
        var queue = t_queue ??= new PriorityQueue<MergeCandidate, MergeCandidate>(_count);
        queue.Clear();
        var skip = t_skip ??= new List<MergeCandidate>();
        skip.Clear();

        // 从相邻符号对构建初始合并候选
        for (int index = 0; index < _count - 1; index++)
        {
            var pair = (_symbols[index].C, _symbols[index + 1].C);
            if (merges.TryGetValue(pair, out var m))
            {
                var candidate = new MergeCandidate(index, m.Rank, m.NewId);
                queue.Enqueue(candidate, candidate);
            }
        }

        while (queue.Count > 0)
        {
            var top = queue.Dequeue();
            int topPos = top.Pos;

            // 处理 dropout：随机跳过合并
            if (dropout.HasValue && Random.Shared.NextSingle() < dropout.Value)
            {
                skip.Add(top);
                continue;
            }

            // Re-insert any skipped elements
            foreach (var sk in skip)
            {
                queue.Enqueue(sk, sk);
            }
            skip.Clear();

            // 跳过已合并的符号（len == 0）
            if (_symbols[topPos].Len == 0)
                continue;

            // 跳过最后一个符号（无后继）
            if (_symbols[topPos].Next == -1)
                continue;

            int nextPos = _symbols[topPos].Next;
            var right = _symbols[nextPos];

            // 验证合并仍然有效 — 此位置的 pair
            // must still map to the same new_id we enqueued.
            var targetPair = (_symbols[topPos].C, right.C);
            if (!merges.TryGetValue(targetPair, out var targetMerge) || targetMerge.NewId != top.NewId)
                continue;

            // Perform the merge using the stored NewId
            _symbols[topPos].MergeWith(right, top.NewId);

            // Mark the right symbol as removed
            _symbols[nextPos].Len = 0;

            // 更新新 next 的 prev 指向当前位置
            if (right.Next >= 0 && right.Next < _count)
            {
                _symbols[right.Next].Prev = topPos;
            }

            // Insert new candidate merge with the previous symbol
            ref var updatedCurrent = ref _symbols[topPos];
            if (updatedCurrent.Prev >= 0)
            {
                int prevIdx = updatedCurrent.Prev;
                var prevSymbol = _symbols[prevIdx];
                var newPair = (prevSymbol.C, updatedCurrent.C);
                if (merges.TryGetValue(newPair, out var prevMerge))
                {
                    var candidate = new MergeCandidate(prevIdx, prevMerge.Rank, prevMerge.NewId);
                    queue.Enqueue(candidate, candidate);
                }
            }

            // Insert new candidate merge with the next symbol
            int newNext = updatedCurrent.Next;
            if (newNext >= 0 && newNext < _count)
            {
                var nextSymbol = _symbols[newNext];
                var newPair = (updatedCurrent.C, nextSymbol.C);
                if (merges.TryGetValue(newPair, out var nextMerge))
                {
                    var candidate = new MergeCandidate(topPos, nextMerge.Rank, nextMerge.NewId);
                    queue.Enqueue(candidate, candidate);
                }
            }
        }

        // In-place compact: 移除 len == 0 的符号
        int write = 0;
        for (int read = 0; read < _count; read++)
        {
            if (_symbols[read].Len != 0)
            {
                if (write != read)
                    _symbols[write] = _symbols[read];
                write++;
            }
        }
        _count = write;
    }

    /// <summary>
    /// 使用优先队列合并所有对 — 位图优化版本。
    /// 使用 MergeBitmapLookup 替代 FrozenDictionary，消除哈希计算开销。
    /// </summary>
    public void MergeAll(MergeBitmapLookup merges, float? dropout)
    {
        var queue = t_queue ??= new PriorityQueue<MergeCandidate, MergeCandidate>(_count);
        queue.Clear();
        var skip = t_skip ??= new List<MergeCandidate>();
        skip.Clear();

        // 从相邻符号对构建初始合并候选
        for (int index = 0; index < _count - 1; index++)
        {
            uint leftId = _symbols[index].C;
            uint rightId = _symbols[index + 1].C;
            if (merges.TryGetValue(leftId, rightId, out var rank, out var newId))
            {
                var candidate = new MergeCandidate(index, rank, newId);
                queue.Enqueue(candidate, candidate);
            }
        }

        while (queue.Count > 0)
        {
            var top = queue.Dequeue();
            int topPos = top.Pos;

            if (dropout.HasValue && Random.Shared.NextSingle() < dropout.Value)
            {
                skip.Add(top);
                continue;
            }

            foreach (var sk in skip)
                queue.Enqueue(sk, sk);
            skip.Clear();

            if (_symbols[topPos].Len == 0)
                continue;

            if (_symbols[topPos].Next == -1)
                continue;

            int nextPos = _symbols[topPos].Next;
            var right = _symbols[nextPos];

            // 验证合并仍然有效
            if (!merges.TryGetValue(_symbols[topPos].C, right.C, out _, out var targetNewId) || targetNewId != top.NewId)
                continue;

            _symbols[topPos].MergeWith(right, top.NewId);
            _symbols[nextPos].Len = 0;

            if (right.Next >= 0 && right.Next < _count)
                _symbols[right.Next].Prev = topPos;

            // 左侧新候选
            ref var updatedCurrent = ref _symbols[topPos];
            if (updatedCurrent.Prev >= 0)
            {
                int prevIdx = updatedCurrent.Prev;
                var prevSymbol = _symbols[prevIdx];
                if (merges.TryGetValue(prevSymbol.C, updatedCurrent.C, out var prevRank, out var prevNewId))
                {
                    var candidate = new MergeCandidate(prevIdx, prevRank, prevNewId);
                    queue.Enqueue(candidate, candidate);
                }
            }

            // 右侧新候选
            int newNext = updatedCurrent.Next;
            if (newNext >= 0 && newNext < _count)
            {
                var nextSymbol = _symbols[newNext];
                if (merges.TryGetValue(updatedCurrent.C, nextSymbol.C, out var nextRank, out var nextNewId))
                {
                    var candidate = new MergeCandidate(topPos, nextRank, nextNewId);
                    queue.Enqueue(candidate, candidate);
                }
            }
        }

        // In-place compact
        int write = 0;
        for (int read = 0; read < _count; read++)
        {
            if (_symbols[read].Len != 0)
            {
                if (write != read)
                    _symbols[write] = _symbols[read];
                write++;
            }
        }
        _count = write;
    }

    /// <summary>
    /// 获取指定索引符号的 token ID。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public uint GetSymbolId(int index) => _symbols[index].C;

    /// <summary>
    /// 获取指定索引符号的字节长度。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetSymbolLen(int index) => _symbols[index].Len;

    /// <summary>
    /// 返回所有符号的 token ID 列表。
    /// </summary>
    public List<uint> GetChars()
    {
        var result = new List<uint>(_count);
        for (int i = 0; i < _count; i++)
            result.Add(_symbols[i].C);
        return result;
    }

}
