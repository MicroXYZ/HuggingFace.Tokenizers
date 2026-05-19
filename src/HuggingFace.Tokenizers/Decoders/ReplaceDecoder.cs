using System.Diagnostics.CodeAnalysis;
using System.Text.RegularExpressions;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Decoders;

/// <summary>
/// 将解码后 token 中模式的出现替换为替换字符串。
/// 支持字符串和正则模式（与 Rust Replace 标准化器/解码器一致）。
/// </summary>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
[TokenizerComponent("Replace")]
public sealed class ReplaceDecoder : IDecoder
{
    private readonly ReadOnlyMemory<char> _pattern;
    private readonly ReadOnlyMemory<char> _replacement;
    private readonly ReplacePatternType _patternType;
    private readonly Regex? _regex;

    /// <summary>
    /// 模式字符串。
    /// </summary>
    [SkipSerialization]
    public string Pattern => _pattern.ToString();

    /// <summary>
    /// 替换字符串。
    /// </summary>
    [SkipSerialization]
    public string ReplacementValue => _replacement.ToString();

    /// <summary>
    /// 模式是否为正则表达式。
    /// </summary>
    [SkipSerialization]
    public ReplacePatternType PatternType => _patternType;

    /// <summary>
    /// 初始化 <see cref="ReplaceDecoder"/> 的新实例（字符串模式）。
    /// </summary>
    /// <param name="pattern">搜索的字符串模式。</param>
    /// <param name="replacement">替换字符串。</param>
    public ReplaceDecoder(string pattern, string replacement)
        : this(pattern, replacement, ReplacePatternType.String)
    {
    }

    /// <summary>
    /// 初始化 <see cref="ReplaceDecoder"/> 的新实例。
    /// </summary>
    /// <param name="pattern">搜索的模式（字符串或正则）。</param>
    /// <param name="replacement">替换字符串。</param>
    /// <param name="patternType">模式类型（字面字符串或正则）。</param>
    public ReplaceDecoder(string pattern, string replacement, ReplacePatternType patternType)
    {
        ArgumentNullException.ThrowIfNull(pattern);
        ArgumentNullException.ThrowIfNull(replacement);
        _pattern = pattern.AsMemory();
        _replacement = replacement.AsMemory();
        _patternType = patternType;

        if (patternType == ReplacePatternType.Regex)
            _regex = CreateRegex(pattern);
    }

    private static Regex CreateRegex(string pattern)
        => HuggingFace.Tokenizers.Internal.RegexHelper.CreateRegex(pattern);

    /// <inheritdoc/>
    public IReadOnlyList<string> DecodeChain(IReadOnlyList<string> tokens)
    {
        var result = new string[tokens.Count];
        for (int i = 0; i < tokens.Count; i++)
        {
            result[i] = _patternType switch
            {
                ReplacePatternType.Regex => _regex!.Replace(tokens[i], _replacement.ToString()),
                _ => StringTransforms.SpanReplace(tokens[i].AsSpan(), _pattern.Span, _replacement.Span),
            };
        }
        return result;
    }

    /// <inheritdoc/>
    public string Decode(IReadOnlyList<string> tokens)
    {
        return DecoderHelper.JoinTokens(DecodeChain(tokens));
    }
}
