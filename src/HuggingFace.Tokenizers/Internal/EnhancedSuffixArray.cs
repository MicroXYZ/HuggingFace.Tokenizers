using System.Buffers;

namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// 增强后缀数组（Enhanced Suffix Array）。
/// 从后缀数组构建 suffix tree 结构，枚举所有子串及其频率。
/// 移植自 esaxx-rs esa.rs（https://github.com/Narsil/esaxx-rs）。
/// </summary>
internal static class EnhancedSuffixArray
{
    /// <summary>
    /// 子串信息：起始偏移 + 长度 + 频率。
    /// </summary>
    internal readonly record struct SubstringEntry(int Offset, int Length, int Frequency);

    /// <summary>
    /// 从输入字符串构建增强后缀数组，枚举所有子串及频率。
    /// 等价于 Rust esaxx_rs::suffix_rs 的 Suffix 迭代器。
    /// </summary>
    /// <param name="text">输入字符数组（UTF-32 码位）</param>
    /// <returns>所有子串（偏移+长度+频率），按字典序排列</returns>
    public static List<SubstringEntry> EnumerateSubstrings(int[] text)
    {
        int n = text.Length;
        if (n == 0) return [];

        var sa = ArrayPool<int>.Shared.Rent(n);
        var left = ArrayPool<int>.Shared.Rent(n);
        var right = ArrayPool<int>.Shared.Rent(n);
        var depth = ArrayPool<int>.Shared.Rent(n);
        try
        {
            // Step 1: 构建后缀数组
            Sais.Build(text, sa, 0x110000);

            // Step 2: 构建 suffix tree（Psi / PLCP / H arrays）
            int nodeNum = BuildSuffixTree(text, sa, left, right, depth, n);

            // Step 3: 枚举所有子串
            var result = new List<SubstringEntry>(nodeNum);
            for (int i = 0; i < nodeNum; i++)
            {
                int offset = sa[left[i]];
                int len = depth[i];
                int freq = right[i] - left[i];
                result.Add(new SubstringEntry(offset, len, freq));
            }

            return result;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(sa);
            ArrayPool<int>.Shared.Return(left);
            ArrayPool<int>.Shared.Return(right);
            ArrayPool<int>.Shared.Return(depth);
        }
    }

    /// <summary>
    /// 从 UTF-8 字节数组构建增强后缀数组，枚举所有子串及频率。
    /// 与 Rust esaxx_rs::suffix_rs(&amp;flat_string) 对齐：在字节级别构建后缀数组。
    /// </summary>
    /// <param name="utf8Bytes">输入 UTF-8 字节数组</param>
    /// <returns>所有子串（字节偏移+字节长度+频率），按字典序排列</returns>
    public static List<SubstringEntry> EnumerateSubstrings(ReadOnlySpan<byte> utf8Bytes)
    {
        // 将字节数组转为 int[]（每个字节作为一个码位，字母表大小 256）
        int n = utf8Bytes.Length;
        if (n == 0) return [];

        var text = new int[n];
        for (int i = 0; i < n; i++)
            text[i] = utf8Bytes[i];

        return EnumerateSubstrings(text, alphabetSize: 256);
    }

    /// <summary>
    /// 从输入数组构建增强后缀数组（可指定字母表大小）。
    /// </summary>
    private static List<SubstringEntry> EnumerateSubstrings(int[] text, int alphabetSize)
    {
        int n = text.Length;
        if (n == 0) return [];

        var sa = ArrayPool<int>.Shared.Rent(n);
        var left = ArrayPool<int>.Shared.Rent(n);
        var right = ArrayPool<int>.Shared.Rent(n);
        var depth = ArrayPool<int>.Shared.Rent(n);
        try
        {
            Sais.Build(text, sa, alphabetSize);
            int nodeNum = BuildSuffixTree(text, sa, left, right, depth, n);

            var result = new List<SubstringEntry>(nodeNum);
            for (int i = 0; i < nodeNum; i++)
            {
                int offset = sa[left[i]];
                int len = depth[i];
                int freq = right[i] - left[i];
                result.Add(new SubstringEntry(offset, len, freq));
            }

            return result;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(sa);
            ArrayPool<int>.Shared.Return(left);
            ArrayPool<int>.Shared.Return(right);
            ArrayPool<int>.Shared.Return(depth);
        }
    }

    /// <summary>
    /// 从后缀数组构建 suffix tree 结构。
    /// 对应 Rust esa.rs 中的 suffixtree 函数。
    /// </summary>
    /// <returns>节点数量</returns>
    private static int BuildSuffixTree(int[] text, int[] sa, int[] left, int[] right, int[] depth, int n)
    {
        if (n == 0) return 0;

        // Psi = left（前驱数组）
        left[sa[0]] = sa[n - 1];
        for (int i = 1; i < n; i++)
            left[sa[i]] = sa[i - 1];

        // PLCP = right（Permuted Longest-Common-Prefix）
        int h = 0;
        for (int i = 0; i < n; i++)
        {
            int j = left[i];
            while (i + h < n && j + h < n && text[i + h] == text[j + h])
                h++;
            right[i] = h;
            h = Math.Max(h - 1, 0);
        }

        // H = left（按后缀数组顺序重排 LCP 值）
        for (int i = 0; i < n; i++)
            left[i] = right[sa[i]];

        // 枚举内部节点（对应 Rust esa.rs 的 stack-based 算法）
        var stack = new List<(int Index, int Depth)> { (-1, -1) };
        int nodeNum = 0;
        int idx = 0;
        int iterGuard = 0;

        while (true)
        {
            var cur = (Index: idx, Depth: (idx == n) ? -1 : left[idx]);
            var cand = stack[^1];

            while (cand.Depth > cur.Depth)
            {
                if (idx - cand.Index > 1)
                {
                    left[nodeNum] = cand.Index;
                    right[nodeNum] = idx;
                    depth[nodeNum] = cand.Depth;
                    nodeNum++;
                    if (nodeNum >= n) break;
                }
                cur = (cand.Index, cur.Depth);
                stack.RemoveAt(stack.Count - 1);
                cand = stack[^1];
            }

            if (cand.Depth < cur.Depth)
                stack.Add(cur);

            if (idx == n) break;

            stack.Add((idx, n - sa[idx] + 1));
            idx++;
            if (++iterGuard > n + 2)
                throw new InvalidOperationException($"BuildSuffixTree infinite loop at idx={idx}, n={n}");
        }

        return nodeNum;
    }
}
