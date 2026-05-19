using System.Runtime.CompilerServices;
using System.Text;

namespace HuggingFace.Tokenizers.Models.Unigram;

/// <summary>
/// Unigram 模型专用双数组字典树（Double-Array Trie）。
/// 在 Unicode 码位上操作，O(k) 前缀查找，替代 sorted children + 二分查找的 O(k·log(d))。
///
/// 使用分离数组编码（base[], check[], label[], value[]）：
/// - base[i]：节点 i 的子节点偏移量，子节点位置 = base[i] XOR childCodepoint
/// - check[i]：节点 i 的父节点位置（路径验证）
/// - label[i]：节点 i 对应的 Unicode 码位（标签匹配）
/// - value[i]：叶节点的 token ID（-1 表示非叶节点）
/// </summary>
internal sealed class UnigramDoubleArrayTrie
{
    private readonly int[] _base;
    private readonly int[] _check;
    private readonly int[] _label;
    private readonly int[] _value;
    private readonly int _size;

    private UnigramDoubleArrayTrie(int[] baseArr, int[] check, int[] label, int[] value, int size)
    {
        _base = baseArr;
        _check = check;
        _label = label;
        _value = value;
        _size = size;
    }

    /// <summary>
    /// 在给定位置查找所有以文本前缀匹配的 token。
    /// </summary>
    public void FindPrefixes(string text, int start, List<(int CharEnd, int TokenId)> results)
    {
        results.Clear();
        int pos = 0;

        int charIdx = start;
        while (charIdx < text.Length)
        {
            if (!Rune.TryGetRuneAt(text, charIdx, out var rune))
            {
                charIdx++;
                continue;
            }

            int cp = rune.Value;
            int charLen = rune.Utf16SequenceLength;
            int childPos = _base[pos] ^ cp;

            if (childPos < 0 || childPos >= _size || _check[childPos] != pos || _label[childPos] != cp)
                break;

            pos = childPos;
            charIdx += charLen;

            if (_value[pos] >= 0)
                results.Add((charIdx, _value[pos]));
        }
    }

    /// <summary>
    /// 通过码位直接查找单字符 token ID。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public int GetTokenIdByCodepoint(int codepoint)
    {
        int childPos = _base[0] ^ codepoint;
        if (childPos < 0 || childPos >= _size || _check[childPos] != 0 || _label[childPos] != codepoint)
            return -1;
        return _value[childPos];
    }

    // ═══════════════════════════════════════════════════════════════
    //  构建
    // ═══════════════════════════════════════════════════════════════

    public static UnigramDoubleArrayTrie Build(IReadOnlyList<(string Token, int TokenId)> sortedTokens)
    {
        if (sortedTokens.Count == 0)
            return new UnigramDoubleArrayTrie([0], [-1], [0], [-1], 1);

        var trie = new TrieNode();
        foreach (var (token, tokenId) in sortedTokens)
            InsertIntoTrie(trie, token, tokenId);

        var b = new DATBuilder(Math.Max(512, sortedTokens.Count * 10));
        b.BuildDAT(trie, 0);
        return new UnigramDoubleArrayTrie(b.Base, b.Check, b.Label, b.Value, b.NextPos);
    }

    // ── Trie 节点 ──

    private sealed class TrieNode
    {
        public List<(int Codepoint, TrieNode Child)> Children = new();
        public int TokenId = -1;
    }

    private static void InsertIntoTrie(TrieNode root, string token, int tokenId)
    {
        var node = root;
        foreach (var rune in token.EnumerateRunes())
        {
            int cp = rune.Value;
            TrieNode? child = null;
            for (int i = 0; i < node.Children.Count; i++)
            {
                if (node.Children[i].Codepoint == cp) { child = node.Children[i].Child; break; }
            }
            if (child is null)
            {
                child = new TrieNode();
                node.Children.Add((cp, child));
                node.Children.Sort((a, b) => a.Codepoint.CompareTo(b.Codepoint));
            }
            node = child;
        }
        node.TokenId = tokenId;
    }

    // ── DAT 构建器（实例字段，避免 ref 数组问题）──

    private sealed class DATBuilder
    {
        public int[] Base;
        public int[] Check;
        public int[] Label;
        public int[] Value;
        public int NextPos = 1;

        public DATBuilder(int cap)
        {
            Base = new int[cap];
            Check = new int[cap];
            Label = new int[cap];
            Value = new int[cap];
            Array.Fill(Check, -1);
            Array.Fill(Value, -1);
        }

        private void EnsureCapacity(int required)
        {
            if (required <= Base.Length) return;
            int newCap = Math.Max(Base.Length * 2, required);
            Array.Resize(ref Base, newCap);
            Array.Resize(ref Check, newCap);
            Array.Resize(ref Label, newCap);
            Array.Resize(ref Value, newCap);
            for (int i = Base.Length; i < newCap; i++) { Check[i] = -1; Value[i] = -1; }
        }

        public void BuildDAT(TrieNode node, int pos)
        {
            if (node.Children.Count == 0) return;

            var cps = new int[node.Children.Count];
            for (int i = 0; i < node.Children.Count; i++)
                cps[i] = node.Children[i].Codepoint;

            int baseValue = FindValidBase(cps, pos);
            Base[pos] = baseValue;

            int maxChildPos = 0;
            for (int i = 0; i < cps.Length; i++)
                maxChildPos = Math.Max(maxChildPos, baseValue ^ cps[i]);
            EnsureCapacity(maxChildPos + 1);

            for (int i = 0; i < node.Children.Count; i++)
            {
                int cp = cps[i];
                var child = node.Children[i].Child;
                int childPos = baseValue ^ cp;
                Check[childPos] = pos;
                Label[childPos] = cp;
                if (child.TokenId >= 0) Value[childPos] = child.TokenId;
                NextPos = Math.Max(NextPos, childPos + 1);
            }

            for (int i = 0; i < node.Children.Count; i++)
            {
                var child = node.Children[i].Child;
                if (child.Children.Count > 0)
                    BuildDAT(child, baseValue ^ cps[i]);
            }
        }

        private int FindValidBase(int[] childCps, int parentPos)
        {
            for (int bv = 1; ; bv++)
            {
                EnsureCapacity(bv + 256); // 确保有足够空间
                bool valid = true;
                for (int i = 0; i < childCps.Length; i++)
                {
                    int childPos = bv ^ childCps[i];
                    // childPos == 0 会覆盖根节点数据（根节点固定在位置 0），
                    // 导致 FindPrefixes 在非根位置误匹配根节点的子节点。
                    if (childPos == 0 || childPos >= Base.Length || (Check[childPos] >= 0 && Check[childPos] != parentPos))
                    { valid = false; break; }
                }
                if (valid) return bv;
            }
        }
    }
}
