using System.Runtime.CompilerServices;

namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// SA-IS 后缀数组构建算法。
/// 移植自 esaxx-rs（https://github.com/Narsil/esaxx-rs），
/// 原始实现来自 SentencePiece（Yuta Mori 的 SA-IS + Daisuke Okanohara 的 esa）。
/// 时间复杂度 O(n)，空间复杂度 O(n)。
/// </summary>
internal static class Sais
{
    /// <summary>
    /// 构建后缀数组。
    /// </summary>
    /// <param name="text">输入字符串（UTF-32 码位数组）</param>
    /// <param name="suffixArray">输出后缀数组（长度必须等于 text.Length）</param>
    /// <param name="alphabetSize">字母表大小（默认 0x110000 = 全部 UCS4）</param>
    public static void Build(int[] text, int[] suffixArray, int alphabetSize = 0x110000)
    {
        int n = text.Length;
        if (n == 0) return;
        if (n == 1)
        {
            suffixArray[0] = 0;
            return;
        }
        SuffixSort(text, suffixArray, 0, n, alphabetSize);
    }

    /// <summary>
    /// 判断高位是否为 1（用于 L/S type 标记的取反编码）。
    /// SA-IS 用 ~j 标记：对正整数取反后 bit 31 = 1（负数）。
    /// 与 C++ esaxx 中 `j < 0` 检查等价。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool HasHighBit(int j) => j < 0;

    // 注意：Rust 版本用 usize 的最高位作为 L/S type 标记。
    // !j 表示取反（翻转所有位），j > 0 && !has_high_bit(j) 表示"未标记的正索引"。
    // C# 中 int 是有符号 32 位，我们用同样的位操作方式。

    /// <summary>
    /// 统计每个字符的出现次数。
    /// </summary>
    private static void GetCounts(int[] text, int[] counts, int n, int k)
    {
        counts.AsSpan(0, k).Clear();
        for (int i = 0; i < n; i++)
            counts[text[i]]++;
    }

    /// <summary>
    /// 计算桶的起始或结束位置。
    /// </summary>
    private static void GetBuckets(int[] counts, int[] buckets, int k, bool end)
    {
        int sum = 0;
        if (end)
        {
            for (int i = 0; i < k; i++)
            {
                sum += counts[i];
                buckets[i] = sum;
            }
        }
        else
        {
            for (int i = 0; i < k; i++)
            {
                buckets[i] = sum;
                sum += counts[i];
            }
        }
    }

    /// <summary>
    /// 诱导排序：从 L-type 和 S-type 子串诱导完整后缀数组。
    /// </summary>
    private static void InduceSa(int[] text, int[] sa, int[] counts, int[] buckets, int n, int k)
    {
        GetCounts(text, counts, n, k);
        GetBuckets(counts, buckets, k, false);

        int c1;
        int j = n - 1;
        c1 = text[j];
        int index = buckets[c1];
        sa[index] = (j > 0 && text[j - 1] < c1) ? ~j : j;
        index++;

        for (int i = 0; i < n; i++)
        {
            j = sa[i];
            sa[i] = ~j;
            if (!HasHighBit(j) && j > 0)
            {
                j--;
                int c0 = text[j];
                if (c0 != c1)
                {
                    buckets[c1] = index;
                    c1 = c0;
                    index = buckets[c1];
                }
                sa[index] = (j > 0 && !HasHighBit(j) && text[j - 1] < c1) ? ~j : j;
                index++;
            }
        }

        // 构建后缀数组
        GetCounts(text, counts, n, k);
        GetBuckets(counts, buckets, k, true);
        c1 = 0;
        index = buckets[c1];
        for (int i = n - 1; i >= 0; i--)
        {
            j = sa[i];
            if (j > 0 && !HasHighBit(j))
            {
                j--;
                int c0 = text[j];
                if (c0 != c1)
                {
                    buckets[c1] = index;
                    c1 = c0;
                    index = buckets[c1];
                }
                index--;
                sa[index] = (j == 0 || text[j - 1] > c1) ? ~j : j;
            }
            else
            {
                sa[i] = ~j;
            }
        }
    }

    /// <summary>
    /// SA-IS 后缀排序核心算法。
    /// </summary>
    private static void SuffixSort(int[] text, int[] sa, int fs, int n, int k)
    {
        int[] counts = new int[k];
        int[] buckets = new int[k];

        GetCounts(text, counts, n, k);
        GetBuckets(counts, buckets, k, true);

        // Stage 1: 递归减小问题规模
        sa.AsSpan(0, n).Clear();

        int cIndex = 0;
        int c1 = text[n - 1];
        for (int i = n - 2; i >= 0; i--)
        {
            int c0 = text[i];
            if (c0 < c1 + cIndex)
                cIndex = 1;
            else if (cIndex != 0)
            {
                buckets[c1]--;
                sa[buckets[c1]] = i + 1;
                cIndex = 0;
            }
            c1 = c0;
        }

        InduceSa(text, sa, counts, buckets, n, k);

        // 将所有已排序子串压缩到前 m 个元素中
        int m = 0;
        int j;
        for (int i = 0; i < n; i++)
        {
            int p = sa[i];
            int c0 = text[p];
            if (p > 0 && text[p - 1] > c0)
            {
                j = p + 1;
                int c1j = (j < n) ? text[j] : 0;
                while (j < n && c0 == c1j)
                {
                    c1j = text[j];
                    j++;
                }
                if (j < n && c0 < c1j)
                {
                    sa[m] = p;
                    m++;
                }
            }
        }

        j = m + (n >> 1);
        for (int i = m; i < j; i++)
            sa[i] = 0;

        // 存储所有子串的长度
        j = n;
        cIndex = 0;
        c1 = text[n - 1];
        for (int i = n - 2; i >= 0; i--)
        {
            int c0 = text[i];
            if (c0 < c1 + cIndex)
                cIndex = 1;
            else if (cIndex != 0)
            {
                sa[m + ((i + 1) >> 1)] = j - i - 1;
                j = i + 1;
                cIndex = 0;
            }
            c1 = c0;
        }

        // 查找字典序名称
        int name = 0;
        int q = n;
        int qlen = 0;
        for (int i = 0; i < m; i++)
        {
            int p = sa[i];
            int plen = sa[m + (p >> 1)];
            bool diff = true;
            if (plen == qlen)
            {
                int ji = 0;
                while (ji < plen && text[p + ji] == text[q + ji])
                    ji++;
                if (ji == plen)
                    diff = false;
            }
            if (diff)
            {
                name++;
                q = p;
                qlen = plen;
            }
            sa[m + (p >> 1)] = name;
        }

        // Stage 2: 递归求解
        if (name < m)
        {
            int raIndex = n + fs - m;
            j = m - 1;
            int a = m + (n >> 1);
            for (int i = a - 1; i >= m; i--)
            {
                if (sa[i] != 0)
                {
                    sa[raIndex + j] = sa[i] - 1;
                    j = Math.Max(j - 1, 0);
                }
            }

            // 构建递归输入
            int[] ra = new int[m];
            for (int i = 0; i < m; i++)
                ra[i] = sa[raIndex + i];

            SuffixSort(ra, sa, fs + n - m * 2, m, name);

            j = m - 1;
            cIndex = 0;
            c1 = text[n - 1];
            for (int i = n - 2; i >= 0; i--)
            {
                int c0 = text[i];
                if (c0 < c1 + cIndex)
                    cIndex = 1;
                else if (cIndex != 0)
                {
                    sa[raIndex + j] = i + 1;
                    cIndex = 0;
                    j = Math.Max(j - 1, 0);
                }
                c1 = c0;
            }

            for (int i = 0; i < m; i++)
                sa[i] = sa[raIndex + sa[i]];
        }

        // Stage 3: 诱导最终结果
        GetCounts(text, counts, n, k);
        GetBuckets(counts, buckets, k, true);
        for (int i = m; i < n; i++)
            sa[i] = 0;
        for (int i = m - 1; i >= 0; i--)
        {
            j = sa[i];
            sa[i] = 0;
            if (buckets[text[j]] > 0)
            {
                buckets[text[j]]--;
                sa[buckets[text[j]]] = j;
            }
        }

        InduceSa(text, sa, counts, buckets, n, k);
    }

    /// <summary>
    /// int.ReverseBits() 扩展方法（与 Rust reverse_bits 对应）。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ReverseBits(this int value)
    {
        uint v = (uint)value;
        v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
        v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
        v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
        v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
        v = (v >> 16) | (v << 16);
        return (int)v;
    }

    /// <summary>
    /// uint.ReverseBitsUInt() 扩展方法，用于无符号位操作。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static uint ReverseBitsUInt(this uint v)
    {
        v = ((v >> 1) & 0x55555555) | ((v & 0x55555555) << 1);
        v = ((v >> 2) & 0x33333333) | ((v & 0x33333333) << 2);
        v = ((v >> 4) & 0x0F0F0F0F) | ((v & 0x0F0F0F0F) << 4);
        v = ((v >> 8) & 0x00FF00FF) | ((v & 0x00FF00FF) << 8);
        v = (v >> 16) | (v << 16);
        return v;
    }
}
