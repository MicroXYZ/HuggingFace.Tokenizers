using System.Text.RegularExpressions;

namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// 正则表达式工具方法，供 Resolver 和 Decoder 共享。
/// </summary>
internal static class RegexHelper
{
    /// <summary>
    /// 检查正则模式是否兼容 NonBacktracking 引擎。
    /// 不支持：前瞻 (?!)、(?=)、后顾 (?&lt;!)、(?&lt;=)、反向引用 \1。
    /// </summary>
    public static bool SupportsNonBacktracking(string pattern)
    {
        for (int i = 0; i < pattern.Length - 1; i++)
        {
            if (pattern[i] == '(' && pattern[i + 1] == '?')
            {
                if (i + 2 < pattern.Length && (pattern[i + 2] == '!' || pattern[i + 2] == '='
                    || (i + 3 < pattern.Length && pattern[i + 2] == '<' && (pattern[i + 3] == '!' || pattern[i + 3] == '='))))
                    return false;
            }
            if (pattern[i] == '\\' && i + 1 < pattern.Length && char.IsDigit(pattern[i + 1]))
                return false;
        }
        return true;
    }

    /// <summary>
    /// 创建 Regex 实例，优先使用 NonBacktracking 引擎。
    /// </summary>
    public static Regex CreateRegex(string pattern)
    {
        if (SupportsNonBacktracking(pattern))
            return new Regex(pattern, RegexOptions.NonBacktracking);
        return new Regex(pattern);
    }
}
