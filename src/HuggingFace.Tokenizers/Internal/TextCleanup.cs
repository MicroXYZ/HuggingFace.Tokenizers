namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// 文本清理工具方法。
/// 提供 WordPiece/CTC 解码器共用的英语缩写清理逻辑。
/// 与 Rust wordpiece::cleanup 一致。
/// </summary>
internal static class TextCleanup
{
    // 缩写模式：(前缀空格, 缩写文本, 替换结果)
    // 按长度降序排列，确保最长匹配优先
    private static readonly (string Pattern, string Replacement)[] s_contractionRules =
    [
        (" do not", " don't"),
        (" n't", "n't"),
        (" 'm", "'m"),
        (" 's", "'s"),
        (" 've", "'ve"),
        (" 're", "'re"),
    ];

    /// <summary>
    /// 清理英语缩写间距：移除缩写词前的多余空格。
    /// 例如：" n't" → "n't"，" 'm" → "'m"。
    /// 单次遍历 + string.Create，避免 6 次 Replace 的中间分配。
    /// </summary>
    public static string CleanContractions(string text)
    {
        if (text.Length == 0) return text;

        // 第一遍：计算匹配数和输出长度
        int matchCount = 0;
        int outputLen = text.Length;
        int pos = 0;
        while (pos < text.Length)
        {
            bool matched = false;
            foreach (var (pattern, replacement) in s_contractionRules)
            {
                if (pos + pattern.Length <= text.Length &&
                    text.AsSpan(pos, pattern.Length).SequenceEqual(pattern))
                {
                    matchCount++;
                    outputLen += replacement.Length - pattern.Length;
                    pos += pattern.Length;
                    matched = true;
                    break;
                }
            }
            if (!matched) pos++;
        }

        if (matchCount == 0) return text;

        // 第二遍：string.Create 一次性构建结果
        return string.Create(outputLen, text, static (dest, src) =>
        {
            int read = 0;
            int write = 0;
            while (read < src.Length)
            {
                bool matched = false;
                foreach (var (pattern, replacement) in s_contractionRules)
                {
                    if (read + pattern.Length <= src.Length &&
                        src.AsSpan(read, pattern.Length).SequenceEqual(pattern))
                    {
                        replacement.AsSpan().CopyTo(dest.Slice(write));
                        write += replacement.Length;
                        read += pattern.Length;
                        matched = true;
                        break;
                    }
                }
                if (!matched)
                    dest[write++] = src[read++];
            }
        });
    }
}
