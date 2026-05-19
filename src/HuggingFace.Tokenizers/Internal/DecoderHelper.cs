namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// 解码器共用的字符串拼接辅助方法。
/// 使用 string.Create 替代 string.Join，避免中间分配。
/// </summary>
internal static class DecoderHelper
{
    /// <summary>
    /// 将 token 列表无分隔符拼接为单个字符串。
    /// 等价于 string.Join("", tokens)，但使用 string.Create 零中间分配。
    /// </summary>
    public static string JoinTokens(IReadOnlyList<string> tokens)
    {
        if (tokens.Count == 0) return string.Empty;
        if (tokens.Count == 1) return tokens[0];

        int totalLen = 0;
        for (int i = 0; i < tokens.Count; i++)
            totalLen += tokens[i].Length;

        if (totalLen == 0) return string.Empty;

        return string.Create(totalLen, tokens, static (dest, t) =>
        {
            int pos = 0;
            for (int i = 0; i < t.Count; i++)
            {
                t[i].AsSpan().CopyTo(dest.Slice(pos));
                pos += t[i].Length;
            }
        });
    }
}
