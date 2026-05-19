using System.Text.RegularExpressions;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Normalizers;

/// <summary>
/// 将模式的所有出现替换为替换字符串。
/// 支持字符串和正则模式。
/// 字符串模式走 span 级 IndexOf 路径（无正则开销、无中间 string 分配），
/// 正则模式走 <see cref="NormalizedString.Replace(Regex, string)"/> 路径。
/// 输入输出与 Rust 一致，内部根据模式类型选择最优路径。
/// </summary>
[TokenizerComponent("Replace")]
public sealed class ReplaceNormalizer : INormalizer
{
    private readonly Regex? _regex;
    private readonly ReadOnlyMemory<char> _pattern;
    private readonly ReadOnlyMemory<char> _replacement;
    private readonly string _patternString;
    private readonly bool _isRegexPattern;

    /// <summary>
    /// 原始模式字符串（正则转义前）。
    /// </summary>
    [SkipSerialization]
    public string PatternString => _patternString;

    /// <summary>
    /// 模式是否为正则模式（相对于字面字符串）。
    /// </summary>
    [SkipSerialization]
    public bool IsRegexPattern => _isRegexPattern;

    /// <summary>
    /// 替换字符串。
    /// JSON 字段名为 "content"，与 Rust 的 Replace normalizer 对齐。
    /// </summary>
    [JsonKey("content")]
    public string Replacement => _replacement.ToString();

    /// <summary>
    /// 创建新的 <see cref="ReplaceNormalizer"/> with a string pattern.
    /// 字符串模式直接使用 span 级替换，无正则编译开销、无中间 string 分配。
    /// </summary>
    public ReplaceNormalizer(string pattern, string replacement)
    {
        ArgumentNullException.ThrowIfNull(pattern);
        ArgumentNullException.ThrowIfNull(replacement);
        _pattern = pattern.AsMemory();
        _replacement = replacement.AsMemory();
        _patternString = pattern;
        _isRegexPattern = false;
    }

    /// <summary>
    /// 创建新的 <see cref="ReplaceNormalizer"/> with a regex pattern.
    /// </summary>
    public ReplaceNormalizer(Regex regex, string replacement)
    {
        ArgumentNullException.ThrowIfNull(regex);
        ArgumentNullException.ThrowIfNull(replacement);
        _regex = regex;
        _pattern = regex.ToString().AsMemory();
        _replacement = replacement.AsMemory();
        _patternString = regex.ToString();
        _isRegexPattern = true;
    }

    /// <summary>
    /// 替换标准化字符串中模式的所有出现。
    /// 字符串模式走 span 级替换路径（无正则开销），
    /// 正则模式走正则替换路径。
    /// 输入输出与 Rust 一致。
    /// </summary>
    public void Normalize(NormalizedString normalized)
    {
        if (_isRegexPattern)
            normalized.Replace(_regex!, _replacement.ToString());
        else
            normalized.Replace(_pattern.Span, _replacement.Span);
    }
}
