using System.Text;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// 跨 JIT/AOT 一致的 Unicode 规范化。
///
/// 使用预编译 NormalizationTables 查找表，完全消除 string.Normalize() 调用。
/// 确保 JIT 和 AOT 产生完全一致的结果。
/// </summary>
internal static class UnicodeConsistency
{
    /// <summary>
    /// 一致的 NFC 规范化。使用预编译表：NFD 分解 → CCC 排序 → NFC 重组。
    /// </summary>
    public static string Nfc(string input) => Nfc(input.AsSpan());

    /// <summary>
    /// 一致的 NFC 规范化（Span 重载）。
    /// </summary>
    public static string Nfc(ReadOnlySpan<char> input)
    {
        if (IsAsciiOnly(input)) return input.ToString();

        // 1. NFD 分解（查表）
        Span<int> decomposed = stackalloc int[input.Length * 4]; // 足够大
        int decompLen = DecomposeFully(input, decomposed, isNfkd: false);

        // 2. Canonical Combining Class 排序
        SortByCombiningClass(decomposed.Slice(0, decompLen));

        // 3. NFC 重组（查表 compose）
        Span<int> composed = stackalloc int[decompLen];
        int compLen = NormalizationTables.ComposeNfc(decomposed.Slice(0, decompLen), composed);

        // 4. 转换为 string
        return StringFromCodePoints(composed.Slice(0, compLen));
    }

    /// <summary>
    /// NFC 规范化 + 同步构建变换对。
    /// 消除 BuildNormalizationTransform 的二次 NFD 分解。
    /// </summary>
    public static (string result, List<(char Char, int Change)> transforms) NfcWithTransform(ReadOnlySpan<char> input)
    {
        if (IsAsciiOnly(input))
        {
            // ASCII 无变化：每个字符 change=0
            var transforms = new List<(char, int)>(input.Length);
            foreach (var c in input) transforms.Add((c, 0));
            return (input.ToString(), transforms);
        }

        // 1. NFD 分解
        Span<int> decomposed = stackalloc int[input.Length * 4];
        int decompLen = DecomposeFully(input, decomposed, isNfkd: false);

        // 2. CCC 排序
        SortByCombiningClass(decomposed.Slice(0, decompLen));

        // 3. NFC 重组 + 同步构建变换对
        Span<int> composed = stackalloc int[decompLen];
        Span<int> consumedMap = stackalloc int[decompLen]; // 每个 composed 消耗的 decomposed 码点数
        int compLen = NormalizationTables.ComposeNfcWithConsumed(decomposed.Slice(0, decompLen), composed, consumedMap);

        // 4. 构建 (char, change) 变换对
        var result = new List<(char, int)>(compLen);
        Span<char> charBuf = stackalloc char[2];
        for (int i = 0; i < compLen; i++)
        {
            int cp = composed[i];
            int consumed = consumedMap[i];
            int change = -(consumed - 1); // 消耗 N 个输入码点产出 1 个 → change = -(N-1)

            if (cp <= 0xFFFF)
            {
                result.Add(((char)cp, change));
            }
            else
            {
                charBuf[0] = (char)(((cp - 0x10000) >> 10) + 0xD800);
                charBuf[1] = (char)(((cp - 0x10000) & 0x3FF) + 0xDC00);
                result.Add((charBuf[0], change));
                result.Add((charBuf[1], 1));
            }
        }

        return (StringFromCodePoints(composed.Slice(0, compLen)), result);
    }

    /// <summary>
    /// 一致的 NFD 规范化。
    /// </summary>
    public static string Nfd(string input) => Nfd(input.AsSpan());

    /// <summary>
    /// 一致的 NFD 规范化（Span 重载）。
    /// </summary>
    public static string Nfd(ReadOnlySpan<char> input)
    {
        if (IsAsciiOnly(input)) return input.ToString();

        Span<int> decomposed = stackalloc int[input.Length * 4];
        int decompLen = DecomposeFully(input, decomposed, isNfkd: false);

        // CCC 排序
        SortByCombiningClass(decomposed.Slice(0, decompLen));

        return StringFromCodePoints(decomposed.Slice(0, decompLen));
    }

    /// <summary>
    /// 一致的 NFKC 规范化。
    /// </summary>
    public static string Nfkc(string input) => Nfkc(input.AsSpan());

    /// <summary>
    /// 一致的 NFKC 规范化（Span 重载）。
    /// </summary>
    public static string Nfkc(ReadOnlySpan<char> input)
    {
        if (IsAsciiOnly(input)) return input.ToString();

        // NFKD 分解
        Span<int> decomposed = stackalloc int[input.Length * 6];
        int decompLen = DecomposeFully(input, decomposed, isNfkd: true);

        // CCC 排序
        SortByCombiningClass(decomposed.Slice(0, decompLen));

        // NFC 重组
        Span<int> composed = stackalloc int[decompLen];
        int compLen = NormalizationTables.ComposeNfc(decomposed.Slice(0, decompLen), composed);

        return StringFromCodePoints(composed.Slice(0, compLen));
    }

    /// <summary>
    /// 一致的 NFKD 规范化。
    /// </summary>
    public static string Nfkd(string input) => Nfkd(input.AsSpan());

    /// <summary>
    /// 一致的 NFKD 规范化（Span 重载）。
    /// </summary>
    public static string Nfkd(ReadOnlySpan<char> input)
    {
        if (IsAsciiOnly(input)) return input.ToString();

        Span<int> decomposed = stackalloc int[input.Length * 6];
        int decompLen = DecomposeFully(input, decomposed, isNfkd: true);

        // CCC 排序
        SortByCombiningClass(decomposed.Slice(0, decompLen));

        return StringFromCodePoints(decomposed.Slice(0, decompLen));
    }

    // ── 内部实现 ──

    /// <summary>
    /// 完全分解：将输入字符串分解为码点数组。
    /// isNfkd=true 使用兼容分解，isNfkd=false 使用规范分解。
    /// </summary>
    private static int DecomposeFully(ReadOnlySpan<char> input, Span<int> output, bool isNfkd)
    {
        int pos = 0;
        var enumerator = input.EnumerateRunes();
        while (enumerator.MoveNext())
        {
            int cp = enumerator.Current.Value;
            if (isNfkd)
                pos += DecomposeNfkd(cp, output.Slice(pos));
            else
                pos += DecomposeNfd(cp, output.Slice(pos));
        }
        return pos;
    }

    /// <summary>
    /// NFD 递归分解（处理链式分解）。
    /// </summary>
    private static int DecomposeNfd(int cp, Span<int> output)
    {
        var (d1, d2) = NormalizationTables.GetNfd(cp);
        if (d1 == cp) { output[0] = cp; return 1; }
        int pos = DecomposeNfd(d1, output);
        if (d2 != 0) pos += DecomposeNfd(d2, output.Slice(pos));
        return pos;
    }

    /// <summary>
    /// NFKD 递归分解（处理兼容分解和链式分解）。
    /// </summary>
    private static int DecomposeNfkd(int cp, Span<int> output)
    {
        var nfkd = NormalizationTables.GetNfkd(cp);
        if (nfkd.IsEmpty) { output[0] = cp; return 1; }

        int pos = 0;
        foreach (var c in nfkd)
        {
            int charCp = c;
            // 处理 surrogate pair
            if (char.IsHighSurrogate(c) && pos + 1 < nfkd.Length)
            {
                char next = nfkd[nfkd.IndexOf(c) + 1];
                if (char.IsLowSurrogate(next))
                    charCp = char.ConvertToUtf32(c, next);
            }
            pos += DecomposeNfkd(charCp, output.Slice(pos));
        }
        return pos;
    }

    /// <summary>
    /// 按 Canonical Combining Class 排序（冒泡排序，稳定）。
    /// 仅对 CCC > 0 的连续序列排序。
    /// </summary>
    private static void SortByCombiningClass(Span<int> decomposed)
    {
        // 使用简单的冒泡排序（稳定），对连续的非零 CCC 字符排序
        int len = decomposed.Length;
        for (int i = 0; i < len - 1; i++)
        {
            int cccI = NormalizationTables.GetCcc(decomposed[i]);
            if (cccI == 0) continue;

            for (int j = i + 1; j < len; j++)
            {
                int cccJ = NormalizationTables.GetCcc(decomposed[j]);
                if (cccJ == 0) break; // 遇到 CCC=0 停止

                if (cccI > cccJ)
                {
                    (decomposed[i], decomposed[j]) = (decomposed[j], decomposed[i]);
                    cccI = cccJ;
                }
            }
        }
    }

    /// <summary>
    /// 从码点数组构建 string。
    /// </summary>
    private static string StringFromCodePoints(ReadOnlySpan<int> codePoints)
    {
        int charLen = 0;
        foreach (var cp in codePoints)
            charLen += cp <= 0xFFFF ? 1 : 2;

        return string.Create(charLen, codePoints, (span, cps) =>
        {
            int pos = 0;
            foreach (var cp in cps)
            {
                if (cp <= 0xFFFF)
                {
                    span[pos++] = (char)cp;
                }
                else
                {
                    span[pos++] = (char)(((cp - 0x10000) >> 10) + 0xD800);
                    span[pos++] = (char)(((cp - 0x10000) & 0x3FF) + 0xDC00);
                }
            }
        });
    }

    private static bool IsAsciiOnly(ReadOnlySpan<char> s)
    {
        int i = 0;
        // SIMD 路径：Vector<ushort> 批量比较
        if (System.Numerics.Vector.IsHardwareAccelerated)
        {
            int vecCount = System.Numerics.Vector<ushort>.Count;
            var threshold = new System.Numerics.Vector<ushort>(0x7F);
            for (; i + vecCount <= s.Length; i += vecCount)
            {
                var chunk = new System.Numerics.Vector<ushort>(System.Runtime.InteropServices.MemoryMarshal.Cast<char, ushort>(s.Slice(i, vecCount)));
                if (System.Numerics.Vector.GreaterThanAny(chunk, threshold))
                    return false;
            }
        }
        // 标量处理剩余
        for (; i < s.Length; i++)
            if (s[i] > 0x7F) return false;
        return true;
    }
}
