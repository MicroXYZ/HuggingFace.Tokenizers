using System.Runtime.CompilerServices;

namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// ASCII 字符分类位图，用于热路径的快速字符分类。
/// 使用 256-bit 位图（4 个 ulong），单次位操作判断字符类别。
/// 比逐字符范围比较或 char.IsXxx() 调用更快。
/// </summary>
internal static class AsciiBitmap
{
    // 每个 ulong 64 bit，4 个 = 256 bit，覆盖全部 ASCII 字符 (0x00~0x7F)
    // 索引: char >> 6 得到 ulong 索引，char & 63 得到 bit 位
    private static readonly ulong[] Whitespace = InitWhitespace();
    private static readonly ulong[] Punctuation = InitPunctuation();
    private static readonly ulong[] Digit = InitDigit();
    private static readonly ulong[] Letter = InitLetter();
    private static readonly ulong[] WordChar = InitWordChar();

    /// <summary>
    /// 判断 ASCII 字符是否为空白字符（space/tab/CR/LF/VT/FF）。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsWhitespace(char c) =>
        c < 128 && (Whitespace[c >> 6] & (1UL << (c & 63))) != 0;

    /// <summary>
    /// 判断 ASCII 字符是否为标点符号。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsPunctuation(char c) =>
        c < 128 && (Punctuation[c >> 6] & (1UL << (c & 63))) != 0;

    /// <summary>
    /// 判断 ASCII 字符是否为数字 (0-9)。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsDigit(char c) =>
        c < 128 && (Digit[c >> 6] & (1UL << (c & 63))) != 0;

    /// <summary>
    /// 判断 ASCII 字符是否为字母 (a-z, A-Z)。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsLetter(char c) =>
        c < 128 && (Letter[c >> 6] & (1UL << (c & 63))) != 0;

    /// <summary>
    /// 判断 ASCII 字符是否为 word 字符（字母 + 数字 + 下划线）。
    /// 与 Rust regex \w 的 ASCII 子集一致。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsWordChar(char c) =>
        c < 128 && (WordChar[c >> 6] & (1UL << (c & 63))) != 0;

    /// <summary>
    /// 判断字符是否为纯 ASCII（&lt; 0x80）。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsAscii(char c) => c < 128;

    /// <summary>
    /// 批量检查 span 中是否所有字符都是 ASCII。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsAllAscii(ReadOnlySpan<char> span)
    {
        for (int i = 0; i < span.Length; i++)
            if (span[i] >= 128) return false;
        return true;
    }

    // ── 初始化方法 ──

    private static ulong[] InitWhitespace()
    {
        // 空白字符: 0x09(\t), 0x0A(\n), 0x0B(\v), 0x0C(\f), 0x0D(\r), 0x20(space)
        var bits = new ulong[4];
        SetBit(bits, '\t');
        SetBit(bits, '\n');
        SetBit(bits, '\v');
        SetBit(bits, '\f');
        SetBit(bits, '\r');
        SetBit(bits, ' ');
        return bits;
    }

    private static ulong[] InitPunctuation()
    {
        // ASCII 标点: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
        var bits = new ulong[4];
        for (char c = (char)0x21; c <= 0x2F; c++) SetBit(bits, c);  // !"#$%&'()*+,-./
        for (char c = (char)0x3A; c <= 0x40; c++) SetBit(bits, c);  // :;<=>?@
        for (char c = (char)0x5B; c <= 0x60; c++) SetBit(bits, c);  // [\]^_`
        for (char c = (char)0x7B; c <= 0x7E; c++) SetBit(bits, c);  // {|}~
        return bits;
    }

    private static ulong[] InitDigit()
    {
        // 数字: 0x30-0x39
        var bits = new ulong[4];
        for (char c = '0'; c <= '9'; c++) SetBit(bits, c);
        return bits;
    }

    private static ulong[] InitLetter()
    {
        // 字母: A-Z, a-z
        var bits = new ulong[4];
        for (char c = 'A'; c <= 'Z'; c++) SetBit(bits, c);
        for (char c = 'a'; c <= 'z'; c++) SetBit(bits, c);
        return bits;
    }

    private static ulong[] InitWordChar()
    {
        // word 字符: 字母 + 数字 + 下划线
        var bits = new ulong[4];
        for (char c = 'A'; c <= 'Z'; c++) SetBit(bits, c);
        for (char c = 'a'; c <= 'z'; c++) SetBit(bits, c);
        for (char c = '0'; c <= '9'; c++) SetBit(bits, c);
        SetBit(bits, '_');
        return bits;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SetBit(ulong[] bits, char c)
    {
        bits[c >> 6] |= 1UL << (c & 63);
    }
}
