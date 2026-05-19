using System.Text;

namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// Unicode 规范化变换构建器。
/// 为 NFD/NFC/NFKD/NFKC 构建 (char, change) 变换对，
/// 供 AlignmentTracker.Transform 使用。
///
/// 使用预编译 NormalizationTables，完全消除 string.Normalize() 调用。
/// </summary>
internal static class UnicodeNormalizer
{
    /// <summary>
    /// 构建 Unicode 规范化的变换对。
    /// </summary>
    public static List<(char Char, int Change)> BuildNormalizationTransform(
        string oldStr, string newStr, NormalizationForm form)
        => BuildNormalizationTransform(oldStr.AsSpan(), newStr, form);

    /// <summary>
    /// 构建 Unicode 规范化的变换对（Span 重载）。
    /// </summary>
    public static List<(char Char, int Change)> BuildNormalizationTransform(
        ReadOnlySpan<char> oldStr, string newStr, NormalizationForm form)
    {
        if (form == NormalizationForm.FormD || form == NormalizationForm.FormKD)
            return BuildDecompositionTransform(oldStr, form);
        else
            return BuildCompositionTransform(oldStr, newStr, form);
    }

    /// <summary>
    /// NFD/NFKD 分解变换：逐 Rune 分解，首个 change=0，其余 change=1。
    /// 使用预编译 NormalizationTables 查找，零 string.Normalize()。
    /// </summary>
    private static List<(char Char, int Change)> BuildDecompositionTransform(
        ReadOnlySpan<char> oldStr, NormalizationForm form)
    {
        bool isNfkd = form == NormalizationForm.FormKD;
        var result = new List<(char, int)>(oldStr.Length * 2);
        Span<int> decompBuf = stackalloc int[18]; // 单字符最大分解长度
        Span<char> charBuf = stackalloc char[2];

        var enumerator = oldStr.EnumerateRunes();
        while (enumerator.MoveNext())
        {
            int cp = enumerator.Current.Value;

            // 使用预编译表分解
            int decompLen;
            if (isNfkd)
                decompLen = DecomposeNfkd(cp, decompBuf);
            else
                decompLen = DecomposeNfd(cp, decompBuf);

            // 转换为 (char, change) 对
            for (int i = 0; i < decompLen; i++)
            {
                int dcp = decompBuf[i];
                if (dcp <= 0xFFFF)
                {
                    result.Add(((char)dcp, i == 0 ? 0 : 1));
                }
                else
                {
                    // surrogate pair
                    charBuf[0] = (char)(((dcp - 0x10000) >> 10) + 0xD800);
                    charBuf[1] = (char)(((dcp - 0x10000) & 0x3FF) + 0xDC00);
                    result.Add((charBuf[0], i == 0 ? 0 : 1));
                    result.Add((charBuf[1], 1));
                }
            }
        }
        return result;
    }

    /// <summary>
    /// NFC/NFKC 组合变换：通过 NFD 作为中间表示精确匹配。
    /// 使用预编译 NormalizationTables，零 string.Normalize()。
    /// </summary>
    private static List<(char Char, int Change)> BuildCompositionTransform(
        ReadOnlySpan<char> oldStr, string newStr, NormalizationForm form)
    {
        bool isNfkc = form == NormalizationForm.FormKC;

        // 1. 对 oldStr 做完全 NFD/NFKD 分解
        Span<int> nfdBuf = stackalloc int[oldStr.Length * 6];
        int nfdLen = DecomposeFully(oldStr, nfdBuf, isNfkc);

        var result = new List<(char, int)>(newStr.Length);
        int nfdIdx = 0;
        Span<char> charBuf = stackalloc char[2];
        Span<int> charNfd = stackalloc int[18];

        // 2. 逐 Rune 遍历 newStr（已 NFC/NFKC 组合），匹配 NFD 中间表示
        var enumerator = newStr.EnumerateRunes();
        while (enumerator.MoveNext())
        {
            var rune = enumerator.Current;

            // 对 newStr 中的每个字符做 NFD/NFKD 分解
            int charNfdLen = isNfkc
                ? DecomposeNfkc(rune.Value, charNfd)
                : DecomposeNfd(rune.Value, charNfd);

            // 尝试在 nfdBuf 中匹配
            int consumed = 0;
            bool matched = true;
            for (int i = 0; i < charNfdLen; i++)
            {
                if (nfdIdx < nfdLen && nfdBuf[nfdIdx] == charNfd[i])
                {
                    consumed++;
                    nfdIdx++;
                }
                else { matched = false; break; }
            }

            // 输出 (char, change) 对
            if (matched && consumed > 0)
            {
                // 这个组合字符消耗了 NFD 中的 consumed 个码点
                if (rune.IsBmp)
                {
                    result.Add(((char)rune.Value, -(consumed - 1)));
                }
                else
                {
                    charBuf[0] = (char)(((rune.Value - 0x10000) >> 10) + 0xD800);
                    charBuf[1] = (char)(((rune.Value - 0x10000) & 0x3FF) + 0xDC00);
                    result.Add((charBuf[0], -(consumed - 1)));
                    result.Add((charBuf[1], 1));
                }
            }
            else
            {
                // 无法匹配，跳过 nfdBuf 中的一个码点
                nfdIdx++;
                if (rune.IsBmp)
                {
                    result.Add(((char)rune.Value, 0));
                }
                else
                {
                    charBuf[0] = (char)(((rune.Value - 0x10000) >> 10) + 0xD800);
                    charBuf[1] = (char)(((rune.Value - 0x10000) & 0x3FF) + 0xDC00);
                    result.Add((charBuf[0], 0));
                    result.Add((charBuf[1], 1));
                }
            }
        }

        // 输出 nfdBuf 中剩余的码点
        while (nfdIdx < nfdLen)
        {
            int cp = nfdBuf[nfdIdx++];
            if (cp <= 0xFFFF)
            {
                result.Add(((char)cp, 0));
            }
            else
            {
                charBuf[0] = (char)(((cp - 0x10000) >> 10) + 0xD800);
                charBuf[1] = (char)(((cp - 0x10000) & 0x3FF) + 0xDC00);
                result.Add((charBuf[0], 0));
                result.Add((charBuf[1], 1));
            }
        }

        return result;
    }

    // ── 分解辅助方法 ──

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

    private static int DecomposeNfd(int cp, Span<int> output)
    {
        var (d1, d2) = NormalizationTables.GetNfd(cp);
        if (d1 == cp) { output[0] = cp; return 1; }
        int pos = DecomposeNfd(d1, output);
        if (d2 != 0) pos += DecomposeNfd(d2, output.Slice(pos));
        return pos;
    }

    private static int DecomposeNfkd(int cp, Span<int> output)
    {
        var nfkd = NormalizationTables.GetNfkd(cp);
        if (nfkd.IsEmpty) { output[0] = cp; return 1; }

        int pos = 0;
        for (int i = 0; i < nfkd.Length; i++)
        {
            char c = nfkd[i];
            int charCp;
            if (char.IsHighSurrogate(c) && i + 1 < nfkd.Length && char.IsLowSurrogate(nfkd[i + 1]))
            {
                charCp = char.ConvertToUtf32(c, nfkd[i + 1]);
                i++; // skip low surrogate
            }
            else
            {
                charCp = c;
            }
            pos += DecomposeNfkd(charCp, output.Slice(pos));
        }
        return pos;
    }

    private static int DecomposeNfkc(int cp, Span<int> output)
    {
        // NFKC = NFKD 分解 + NFC 重组
        // 这里只做分解，重组在 ComposeNfc 中完成
        return DecomposeNfkd(cp, output);
    }
}
