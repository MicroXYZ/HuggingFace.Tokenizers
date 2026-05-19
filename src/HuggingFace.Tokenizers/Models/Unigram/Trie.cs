using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace HuggingFace.Tokenizers.Models.Unigram;

/// <summary>
/// 前缀树（Trie），用于高效字符串前缀匹配。
/// 使用紧凑数组结构（sorted children + 二分查找）替代 Dictionary，
/// 减少每个节点的内存开销，提升查找性能。
/// 根节点使用 BMP 位图 + PopCount 实现 O(1) 子节点查找。
/// </summary>
public sealed class Trie
{
    // 所有节点扁平存储：每个节点是 _nodes 列表中的一个条目
    private readonly List<TrieNodeData> _nodes = new();

    // 根节点 BMP 位图：2048 个 ulong，覆盖 0x0000~0xFFFF（65536 个码位）
    // 每个 ulong 64 bit，2048 * 64 = 65536
    // 用于根节点的 O(1) 子节点查找，替代二分查找
    // 延迟构建：首次查询时从已排序的 Children 数组构建，而非增量维护
    private ulong[]? _rootBmpBitmap;
    private bool _bitmapDirty = true; // 标记位图是否需要重建

    /// <summary>
    /// 节点数据：children 存储为排序的 (codepoint, childIndex) 数组。
    /// </summary>
    private struct TrieNodeData
    {
        /// <summary>子节点数组，按 codepoint 排序。null 表示无子节点。</summary>
        public (int Codepoint, int ChildIndex)[]? Children;
        /// <summary>子节点数量（Children 数组可能更大，用 Count 跟踪实际数量）。</summary>
        public int ChildCount;
        /// <summary>是否为词尾。</summary>
        public bool IsEndOfWord;
        /// <summary>词表 ID。</summary>
        public int TokenId;
    }

    public Trie()
    {
        // 根节点 index = 0
        _nodes.Add(new TrieNodeData());
    }

    /// <summary>
    /// 预分配容量的构造函数，减少训练时的 rehash/扩容开销。
    /// </summary>
    /// <param name="expectedTokens">预期插入的 token 数量。</param>
    public Trie(int expectedTokens)
    {
        // 预分配：每个 token 平均约 8 个节点（含根节点）
        int estimatedNodes = Math.Max(16, expectedTokens * 8);
        _nodes = new List<TrieNodeData>(estimatedNodes);
        _nodes.Add(new TrieNodeData()); // 根节点
    }

    /// <summary>
    /// 从另一个 Trie 原子替换内部数据。
    /// 避免 Clear + Insert 的中间状态。
    /// </summary>
    internal void ReplaceFrom(Trie other)
    {
        _nodes.Clear();
        _nodes.AddRange(other._nodes);
        _rootBmpBitmap = other._rootBmpBitmap is not null
            ? (ulong[])other._rootBmpBitmap.Clone()
            : null;
        _bitmapDirty = other._bitmapDirty;
    }

    /// <summary>
    /// 将一个 token 插入前缀树。
    /// 使用 EnumerateRunes 遍历 Unicode 码位，正确处理代理对。
    /// 位图在首次查询时延迟构建，不在插入时增量维护。
    /// </summary>
    public void Insert(string token, int tokenId)
    {
        int nodeIdx = 0;
        foreach (var rune in token.EnumerateRunes())
        {
            int key = rune.Value;
            var node = _nodes[nodeIdx];

            int childIdx = FindChildIndex(node, key);
            if (childIdx >= 0)
            {
                nodeIdx = node.Children![childIdx].ChildIndex;
            }
            else
            {
                int newNodeIdx = _nodes.Count;
                _nodes.Add(new TrieNodeData());
                int insertPos = ~childIdx;
                InsertChild(nodeIdx, insertPos, key, newNodeIdx);
                nodeIdx = newNodeIdx;
            }
        }

        var leaf = _nodes[nodeIdx];
        leaf.IsEndOfWord = true;
        leaf.TokenId = tokenId;
        _nodes[nodeIdx] = leaf;

        // 插入新 token 后标记位图需要重建
        _bitmapDirty = true;
    }

    /// <summary>
    /// 查找所有以给定文本（从 start 位置开始）为前缀的 token。
    /// 结果写入调用方提供的 buffer 以避免分配。
    /// 根节点使用 BMP 位图加速查找。
    /// </summary>
    public void FindPrefixes(string text, int start, List<(int End, int TokenId)> results)
    {
        results.Clear();
        int nodeIdx = 0;

        int charPos = start;
        while (charPos < text.Length)
        {
            if (!Rune.TryGetRuneAt(text, charPos, out var rune))
            {
                charPos++;
                continue;
            }

            int key = rune.Value;
            int runeCharLen = rune.Utf16SequenceLength;

            if (nodeIdx == 0)
            {
                // 根节点：优先使用 BMP 位图
                int childNodeIdx = RootBitmapFindChild(key);
                if (childNodeIdx >= 0)
                {
                    nodeIdx = childNodeIdx;
                }
                else if (key <= 0xFFFF)
                {
                    // BMP 码位不在位图中，直接终止
                    break;
                }
                else
                {
                    // 非 BMP：回退到二分查找
                    var rootNode = _nodes[0];
                    int binIdx = FindChildIndex(rootNode, key);
                    if (binIdx < 0)
                        break;
                    nodeIdx = rootNode.Children![binIdx].ChildIndex;
                }
            }
            else
            {
                // 非根节点：二分查找
                var node = _nodes[nodeIdx];
                int childIdx = FindChildIndex(node, key);
                if (childIdx < 0)
                    break;
                nodeIdx = node.Children![childIdx].ChildIndex;
            }

            var child = _nodes[nodeIdx];
            if (child.IsEndOfWord)
            {
                results.Add((charPos + runeCharLen, child.TokenId));
            }

            charPos += runeCharLen;
        }
    }

    /// <summary>
    /// 查找所有以给定文本（从 start 位置开始）为前缀的 token。
    /// </summary>
    public List<(int End, int TokenId)> FindPrefixes(string text, int start)
    {
        var results = new List<(int End, int TokenId)>(32);
        FindPrefixes(text, start, results);
        return results;
    }

    /// <summary>
    /// 检查前缀树中是否包含指定 token。
    /// 根节点使用 BMP 位图加速查找。
    /// </summary>
    public bool Contains(string token)
    {
        int nodeIdx = 0;
        foreach (var rune in token.EnumerateRunes())
        {
            int key = rune.Value;
            if (nodeIdx == 0)
            {
                int childNodeIdx = RootBitmapFindChild(key);
                if (childNodeIdx >= 0)
                {
                    nodeIdx = childNodeIdx;
                    continue;
                }
                if (key <= 0xFFFF) return false;
                // 非 BMP 回退
                var rootNode = _nodes[0];
                int binIdx = FindChildIndex(rootNode, key);
                if (binIdx < 0) return false;
                nodeIdx = rootNode.Children![binIdx].ChildIndex;
            }
            else
            {
                var node = _nodes[nodeIdx];
                int childIdx = FindChildIndex(node, key);
                if (childIdx < 0) return false;
                nodeIdx = node.Children![childIdx].ChildIndex;
            }
        }
        return _nodes[nodeIdx].IsEndOfWord;
    }

    /// <summary>
    /// 获取指定 token 的 ID，如果不存在则返回 -1。
    /// 根节点使用 BMP 位图加速查找。
    /// </summary>
    public int GetTokenId(string token)
    {
        int nodeIdx = 0;
        foreach (var rune in token.EnumerateRunes())
        {
            int key = rune.Value;
            if (nodeIdx == 0)
            {
                int childNodeIdx = RootBitmapFindChild(key);
                if (childNodeIdx >= 0)
                {
                    nodeIdx = childNodeIdx;
                    continue;
                }
                if (key <= 0xFFFF) return -1;
                var rootNode = _nodes[0];
                int binIdx = FindChildIndex(rootNode, key);
                if (binIdx < 0) return -1;
                nodeIdx = rootNode.Children![binIdx].ChildIndex;
            }
            else
            {
                var node = _nodes[nodeIdx];
                int childIdx = FindChildIndex(node, key);
                if (childIdx < 0) return -1;
                nodeIdx = node.Children![childIdx].ChildIndex;
            }
        }
        var leaf = _nodes[nodeIdx];
        return leaf.IsEndOfWord ? leaf.TokenId : -1;
    }

    /// <summary>
    /// 通过 Unicode 码位直接查找单字符 token ID，避免 string 分配。
    /// 根节点使用 BMP 位图加速查找。
    /// </summary>
    public int GetTokenIdByCodepoint(int codepoint)
    {
        // 优先使用 BMP 位图
        int childNodeIdx = RootBitmapFindChild(codepoint);
        if (childNodeIdx >= 0)
        {
            var child = _nodes[childNodeIdx];
            return child.IsEndOfWord ? child.TokenId : -1;
        }
        if (codepoint <= 0xFFFF) return -1;
        // 非 BMP 回退
        var root = _nodes[0];
        int binIdx = FindChildIndex(root, codepoint);
        if (binIdx < 0) return -1;
        var childNode = _nodes[root.Children![binIdx].ChildIndex];
        return childNode.IsEndOfWord ? childNode.TokenId : -1;
    }

    /// <summary>
    /// 清空前缀树。
    /// </summary>
    public void Clear()
    {
        _nodes.Clear();
        _nodes.Add(new TrieNodeData()); // 重建根节点
        _rootBmpBitmap = null;
        _bitmapDirty = true;
    }

    // ── 辅助方法 ──

    /// <summary>
    /// 在节点的子节点数组中二分查找指定 codepoint。
    /// 返回 >= 0 表示找到，< 0 表示未找到（~result 为插入位置）。
    /// </summary>
    private static int FindChildIndex(in TrieNodeData node, int codepoint)
    {
        if (node.Children is null || node.ChildCount == 0)
            return -1;

        // 二分查找
        int lo = 0, hi = node.ChildCount - 1;
        while (lo <= hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            int cmp = node.Children[mid].Codepoint.CompareTo(codepoint);
            if (cmp == 0) return mid;
            if (cmp < 0) lo = mid + 1;
            else hi = mid - 1;
        }
        return ~lo; // 未找到，~lo 为插入位置
    }

    /// <summary>
    /// 在节点的子节点数组中插入一个新子节点。
    /// </summary>
    private void InsertChild(int nodeIdx, int insertPos, int codepoint, int childIndex)
    {
        var node = _nodes[nodeIdx];
        if (node.Children is null)
        {
            node.Children = new (int, int)[4];
        }
        else if (node.ChildCount >= node.Children.Length)
        {
            var newChildren = new (int, int)[node.Children.Length * 2];
            Array.Copy(node.Children, newChildren, node.ChildCount);
            node.Children = newChildren;
        }

        for (int i = node.ChildCount; i > insertPos; i--)
            node.Children[i] = node.Children[i - 1];

        node.Children[insertPos] = (codepoint, childIndex);
        node.ChildCount++;
        _nodes[nodeIdx] = node;
    }

    // ── 根节点 BMP 位图 ──

    /// <summary>
    /// 通过 BMP 位图在根节点中查找子节点索引。
    /// 仅用于 BMP 码位（0~0xFFFF），非 BMP 码位回退到二分查找。
    /// 返回子节点在 Children 数组中的索引，未找到返回 -1。
    /// 首次调用时自动从已排序的 Children 数组构建位图。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int RootBitmapFindChildIndex(int codepoint)
    {
        if (codepoint > 0xFFFF)
            return -1;

        // 延迟构建位图
        if (_bitmapDirty || _rootBmpBitmap is null)
            EnsureRootBitmap();

        ulong word = _rootBmpBitmap![codepoint >> 6];
        ulong mask = 1UL << (codepoint & 63);
        if ((word & mask) == 0)
            return -1;

        // PopCount：统计 codepoint 之前有多少个已设置的 bit
        // 这等于该 codepoint 在排序 Children 数组中的索引
        int index = BitOperations.PopCount(word & (mask - 1));
        // 累加前面所有 ulong word 的 popcount
        for (int i = 0; i < (codepoint >> 6); i++)
            index += BitOperations.PopCount(_rootBmpBitmap[i]);

        return index;
    }

    /// <summary>
    /// 在根节点 BMP 位图中查找子节点，返回子节点的全局索引（在 _nodes 中的位置）。
    /// 未找到返回 -1。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private int RootBitmapFindChild(int codepoint)
    {
        int childIdx = RootBitmapFindChildIndex(codepoint);
        if (childIdx < 0) return -1;
        return _nodes[0].Children![childIdx].ChildIndex;
    }

    /// <summary>
    /// 确保根节点 BMP 位图已构建。
    /// 延迟构建：首次查询时从已排序的根节点 Children 数组一次性构建。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void EnsureRootBitmap()
    {
        if (!_bitmapDirty && _rootBmpBitmap is not null)
            return;

        _rootBmpBitmap ??= new ulong[1024]; // 1024 * 64 = 65536
        Array.Clear(_rootBmpBitmap);

        // 从已排序的根节点 Children 数组构建位图
        var root = _nodes[0];
        if (root.Children is not null)
        {
            for (int i = 0; i < root.ChildCount; i++)
            {
                int cp = root.Children[i].Codepoint;
                if (cp <= 0xFFFF)
                    _rootBmpBitmap[cp >> 6] |= 1UL << (cp & 63);
            }
        }

        _bitmapDirty = false;
    }
}
