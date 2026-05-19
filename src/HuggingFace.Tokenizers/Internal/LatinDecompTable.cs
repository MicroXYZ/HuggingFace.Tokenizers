using System.Text;

namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// 拉丁扩展字符 NFD 分解查表。
/// 覆盖 U+00C0-U+02FF（Latin Extended），这是 BERT strip_accents 的主要目标区间。
/// 查表避免调用 ICU 的完整 NFD 分解，性能提升 3-5x。
///
/// 设计：
/// - 每个条目存储 base char + combining mark 数（0=无分解，1=单 combining，2=双 combining）
/// - combining marks 本身（U+0300-U+036F）直接标记为需要移除
/// - 对于需要完整 NFD 输出的场景，仍回退到 ICU
/// </summary>
internal static class LatinDecompTable
{
    // 查表条目：(baseChar, combiningCount)
    // combiningCount=0 表示该码点无需分解或不在表中
    private static readonly (char BaseChar, byte CombiningCount)[] s_table = BuildTable();

    // combining mark 范围快速检测表：U+0300-U+036F
    // 用 bitfield 压缩，128 个码点只需 16 bytes
    private static readonly ulong[] s_combiningBits = BuildCombiningBits();

    private static (char BaseChar, byte CombiningCount)[] BuildTable()
    {
        var table = new (char BaseChar, byte CombiningCount)[0x0300];

        // 用预编译 NormalizationTables 做 NFD 分解，替代运行时 string.Normalize()
        for (int cp = 0x00C0; cp < 0x0300; cp++)
        {
            if (cp >= 0x0300) continue;

            var (d1, d2) = NormalizationTables.GetNfd(cp);
            if (d1 != cp && d1 < 0x0300)
            {
                if (d2 != 0)
                    table[cp] = ((char)d1, 2); // base + 2 combining
                else
                    table[cp] = ((char)d1, 1); // base + 1 combining
            }
        }

        return table;
    }

    private static ulong[] BuildCombiningBits()
    {
        // U+0300-U+036F (112 个码点) 用 112 bits = 14 ulongs 表示
        // 但用 16 ulongs (128 bits) 覆盖到 U+037F，对齐到 ulong 边界
        var bits = new ulong[12]; // 0x0300-0x036F = 112 bits, 12 ulongs = 768 bits 足够
        for (int cp = 0x0300; cp <= 0x036F; cp++)
        {
            int idx = (cp - 0x0300) >> 6;   // / 64
            int bit = (cp - 0x0300) & 63;    // % 64
            bits[idx] |= 1UL << bit;
        }
        return bits;
    }

    /// <summary>
    /// 判断字符是否为 combining mark（U+0300-U+036F 范围的 NonSpacingMark）。
    /// 比 CharUnicodeInfo.GetUnicodeCategory 快得多（纯位运算）。
    /// </summary>
    public static bool IsCombiningMark(char c)
    {
        int idx = c - 0x0300;
        if ((uint)idx > 0x6F) return false;
        return (s_combiningBits[idx >> 6] & (1UL << (idx & 63))) != 0;
    }

    /// <summary>
    /// 判断字符是否为 combining mark（Rune 版本，处理补充平面）。
    /// 补充平面的 combining marks 极少，直接返回 false。
    /// </summary>
    public static bool IsCombiningMark(Rune rune)
    {
        if (!rune.IsBmp) return false;
        return IsCombiningMark((char)rune.Value);
    }

    /// <summary>
    /// 获取拉丁扩展字符的 base char。
    /// 返回 true 表示查表成功（有分解），false 表示不在表中或无需分解。
    /// </summary>
    public static bool TryGetBaseChar(char c, out char baseChar)
    {
        if (c >= 0x00C0 && c < 0x0300)
        {
            var (bc, cnt) = s_table[c];
            if (cnt > 0) { baseChar = bc; return true; }
        }
        baseChar = default;
        return false;
    }

    /// <summary>
    /// 判断文本中是否含有需要 NFD 分解的拉丁扩展字符。
    /// 快速预检，避免不必要的 normalize 调用。
    /// </summary>
    public static bool NeedsLatinDecomp(ReadOnlySpan<char> text)
    {
        foreach (var c in text)
        {
            if (c >= 0x00C0 && c < 0x0300 && s_table[c].CombiningCount > 0)
                return true;
        }
        return false;
    }

    /// <summary>
    /// 判断文本中是否含有需要 NFD 分解的拉丁扩展字符（string 版本）。
    /// </summary>
    public static bool NeedsLatinDecomp(string text)
    {
        for (int i = 0; i < text.Length; i++)
        {
            var ch = text[i];
            if (ch >= 0x00C0 && ch < 0x0300 && s_table[ch].CombiningCount > 0)
                return true;
        }
        return false;
    }

    /// <summary>
    /// 判断文本中是否含有 ZWJ（U+200D）。
    /// </summary>
    public static bool ContainsZwj(string text) => text.Contains('\u200D');
    public static bool ContainsZwj(ReadOnlySpan<char> text) => text.Contains('\u200D');

    /// <summary>
    /// 判断文本中是否含有 combining mark。
    /// 用于已 NFD 分解后去除 combining marks。
    /// </summary>
    public static bool ContainsCombiningMark(string text)
    {
        for (int i = 0; i < text.Length; i++)
        {
            if (IsCombiningMark(text[i])) return true;
        }
        return false;
    }
}
