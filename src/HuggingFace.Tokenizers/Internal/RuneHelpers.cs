using System.Runtime.CompilerServices;
using System.Text;

namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// Unicode Rune 通用工具方法。
/// 消除项目中 Rune 遍历、byte fallback token、LogSumExp 等重复代码。
/// </summary>
internal static class RuneHelpers
{
    // ────────────────────────────────────────────────────────────────────────
    //  Byte Fallback Token 格式
    // ────────────────────────────────────────────────────────────────────────

    // 预计算的 byte fallback token 查找表，避免每次 string interpolation 分配
    private static readonly string[] ByteFallbackTokens = InitByteFallbackTokens();

    // 预计算的 byte fallback token UTF-8 字节表，避免 UTF-16→UTF-8 转换开销
    private static readonly byte[][] ByteFallbackUtf8 = InitByteFallbackUtf8();

    private static string[] InitByteFallbackTokens()
    {
        var tokens = new string[256];
        for (int i = 0; i < 256; i++)
            tokens[i] = $"<0x{i:X2}>";
        return tokens;
    }

    private static byte[][] InitByteFallbackUtf8()
    {
        var result = new byte[256][];
        for (int i = 0; i < 256; i++)
            result[i] = System.Text.Encoding.UTF8.GetBytes(ByteFallbackTokens[i]);
        return result;
    }

    /// <summary>
    /// 格式化 byte fallback token：&lt;0x{byte:X2}&gt;。
    /// 使用预计算查找表，零分配。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static string FormatByteFallbackToken(byte b) => ByteFallbackTokens[b];

    /// <summary>
    /// 获取 byte fallback token 的 UTF-8 字节表示。
    /// 使用预计算查找表，零分配。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ReadOnlySpan<byte> FormatByteFallbackTokenUtf8(byte b) => ByteFallbackUtf8[b];

    /// <summary>
    /// 尝试解析 byte fallback token。
    /// </summary>
    /// <param name="token">形如 &lt;0x41&gt; 的 token 字符串。</param>
    /// <param name="b">解析出的字节值。</param>
    /// <returns>是否成功解析。</returns>
    public static bool TryParseByteFallbackToken(string token, out byte b)
    {
        b = 0;
        if (token.Length != 6 || token[0] != '<' || token[1] != '0' || token[2] != 'x'
            || token[5] != '>')
            return false;

        int hi = HexValue(token[3]);
        int lo = HexValue(token[4]);
        if (hi < 0 || lo < 0) return false;

        b = (byte)((hi << 4) | lo);
        return true;
    }

    private static int HexValue(char c)
    {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        return -1;
    }

    // ────────────────────────────────────────────────────────────────────────
    //  LogSumExp — 数值稳定的 log(exp(x) + exp(y))
    // ────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// 数值稳定的 log(exp(x) + exp(y))。
    /// 用于 Unigram 模型的 Viterbi、采样和前向-后向算法。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double LogSumExp(double x, double y)
    {
        if (double.IsNegativeInfinity(x)) return y;
        if (double.IsNegativeInfinity(y)) return x;
        double vmin = Math.Min(x, y);
        double vmax = Math.Max(x, y);
        const double kMinusLogEpsilon = 50.0;
        if (vmax > vmin + kMinusLogEpsilon)
            return vmax;
        return vmax + Math.Log(Math.Exp(vmin - vmax) + 1.0);
    }

    /// <summary>
    /// 带初始化模式的 LogSumExp。
    /// initMode 为 true 时返回 y（用于第一个元素）。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double LogSumExp(double x, double y, bool initMode)
    {
        if (initMode) return y;
        return LogSumExp(x, y);
    }
}
