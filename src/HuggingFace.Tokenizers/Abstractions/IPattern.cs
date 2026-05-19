namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 用于拆分字符串的模式。实现包括正则、字符、字符串等。
/// </summary>
public interface IPattern
{
    /// <summary>
    /// 在给定 Span 中查找匹配。
    /// 返回覆盖整个字符串的 (start, end, isMatch) 元组列表。
    /// </summary>
    /// <param name="inside">要搜索的字符 Span。</param>
    /// <returns>匹配结果列表。</returns>
    IReadOnlyList<(int Start, int End, bool IsMatch)> FindMatches(ReadOnlySpan<char> inside);
}
