using System.Buffers;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// 匹配单个字符的模式。
/// </summary>
internal sealed class CharPattern(char c) : IPattern
{
    /// <summary>要匹配的字符。</summary>
    public char Char => c;

    /// <inheritdoc/>
    public IReadOnlyList<(int Start, int End, bool IsMatch)> FindMatches(ReadOnlySpan<char> inside)
    {
        // 使用 ReadOnlySpan.IndexOf 进行 SIMD 加速查找
        var result = new List<(int, int, bool)>();
        int last = 0;
        int searchFrom = 0;
        while (searchFrom < inside.Length)
        {
            int idx = inside.Slice(searchFrom).IndexOf(c);
            if (idx < 0) break;
            idx += searchFrom;
            if (last < idx) result.Add((last, idx, false));
            result.Add((idx, idx + 1, true));
            last = idx + 1;
            searchFrom = idx + 1;
        }
        if (last < inside.Length) result.Add((last, inside.Length, false));
        return result;
    }
}

/// <summary>
/// 匹配字符串字面量的模式。
/// </summary>
internal sealed class StringPattern(string pattern) : IPattern
{
    /// <summary>要匹配的字符串字面量。</summary>
    public string Pattern => pattern;

    /// <inheritdoc/>
    public IReadOnlyList<(int Start, int End, bool IsMatch)> FindMatches(ReadOnlySpan<char> inside)
    {
        if (pattern.Length == 0)
            return [(0, inside.Length, false)];

        var result = new List<(int, int, bool)>();
        int last = 0;
        int idx = 0;
        var patternSpan = pattern.AsSpan();
        while (idx <= inside.Length - patternSpan.Length)
        {
            if (inside.Slice(idx).StartsWith(patternSpan))
            {
                if (last < idx) result.Add((last, idx, false));
                result.Add((idx, idx + patternSpan.Length, true));
                last = idx + patternSpan.Length;
                idx += patternSpan.Length;
            }
            else
            {
                idx++;
            }
        }
        if (last < inside.Length) result.Add((last, inside.Length, false));
        return result;
    }
}

/// <summary>
/// 使用正则表达式匹配的模式。
/// 参考 Microsoft.ML.Tokenizers 的 EnumerateMatches(ReadOnlySpan) 模式。
/// </summary>
internal sealed class RegexPattern(System.Text.RegularExpressions.Regex regex) : IPattern
{
    /// <summary>编译后的正则表达式。</summary>
    public System.Text.RegularExpressions.Regex Regex => regex;

    /// <inheritdoc/>
    public IReadOnlyList<(int Start, int End, bool IsMatch)> FindMatches(ReadOnlySpan<char> inside)
    {
        if (inside.IsEmpty)
            return [(0, 0, false)];

        var result = new List<(int, int, bool)>();
        int prev = 0;
        foreach (var m in regex.EnumerateMatches(inside))
        {
            if (prev < m.Index)
                result.Add((prev, m.Index, false));
            result.Add((m.Index, m.Index + m.Length, true));
            prev = m.Index + m.Length;
        }
        if (prev < inside.Length)
            result.Add((prev, inside.Length, false));
        return result;
    }
}
