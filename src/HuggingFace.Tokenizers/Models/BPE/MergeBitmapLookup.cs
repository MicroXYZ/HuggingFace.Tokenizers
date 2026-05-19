using System.Runtime.CompilerServices;

namespace HuggingFace.Tokenizers.Models.BPE;

/// <summary>
/// 基于两级位图 + 排序数组的 BPE 合并表查找。
/// 替代 FrozenDictionary 的哈希查找，消除哈希计算开销。
///
/// 结构：
///   Level 1: ulong[] 位图，标记哪些 left_token 有合并条目
///   Level 2: 每个 left_token 对应一个按 right_token 排序的 MergeEntry 数组
///
/// 查找复杂度：
///   位图未命中 → O(1) 直接返回 false（无哈希计算）
///   位图命中 → O(1) 位操作 + O(log k) 二分查找
/// </summary>
internal sealed class MergeBitmapLookup
{
    /// <summary>
    /// 合并条目：right_token → (rank, new_id)。
    /// 使用 struct 避免堆分配，按 right 排序支持二分查找。
    /// </summary>
    internal struct MergeEntry
    {
        public uint Right;
        public uint Rank;
        public uint NewId;
    }

    // Level 1 位图：标记哪些 left token 有合并条目
    // 大小 = vocab_size / 64，通常 < 4KB，可放入 L1 cache
    private readonly ulong[] _hasMerges;

    // Level 2：每个 left token 的合并条目，按 right 排序
    // _entries[left_id] = sorted MergeEntry[]，null 表示无合并
    private readonly MergeEntry[]?[] _entries;

    // 总条目数（用于序列化/调试）
    public int TotalMerges { get; }

    /// <summary>
    /// 从 FrozenDictionary 构建位图查找表。
    /// </summary>
    public MergeBitmapLookup(IReadOnlyDictionary<(uint, uint), (uint Rank, uint NewId)> mergeRanks, int vocabSize)
    {
        int bitmapSize = (vocabSize + 63) / 64;
        _hasMerges = new ulong[bitmapSize];
        _entries = new MergeEntry[vocabSize][];
        TotalMerges = mergeRanks.Count;

        // 按 left token 分组
        var groups = new Dictionary<uint, List<MergeEntry>>(1024);
        foreach (var ((left, right), (rank, newId)) in mergeRanks)
        {
            if (!groups.TryGetValue(left, out var list))
            {
                list = new List<MergeEntry>();
                groups[left] = list;
            }
            list.Add(new MergeEntry { Right = right, Rank = rank, NewId = newId });

            // 设置位图 bit
            _hasMerges[left >> 6] |= 1UL << ((int)left & 63);
        }

        // 每组按 right 排序并存储为数组
        foreach (var (left, list) in groups)
        {
            list.Sort((a, b) => a.Right.CompareTo(b.Right));
            _entries[left] = list.ToArray();
        }
    }

    /// <summary>
    /// 查找合并条目。位图未命中时 O(1) 返回 false，无哈希计算。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryGetValue(uint left, uint right, out uint rank, out uint newId)
    {
        // Level 1: 位图快速排除 — O(1) 位操作
        if (left >= (uint)_entries.Length || (_hasMerges[left >> 6] & (1UL << ((int)left & 63))) == 0)
        {
            rank = 0;
            newId = 0;
            return false;
        }

        // Level 2: 二分查找 right_token — O(log k)
        var arr = _entries[left];
        if (arr is null)
        {
            rank = 0;
            newId = 0;
            return false;
        }

        int lo = 0, hi = arr.Length - 1;
        while (lo <= hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            uint midRight = arr[mid].Right;
            if (midRight == right)
            {
                rank = arr[mid].Rank;
                newId = arr[mid].NewId;
                return true;
            }
            if (midRight < right)
                lo = mid + 1;
            else
                hi = mid - 1;
        }

        rank = 0;
        newId = 0;
        return false;
    }
}
