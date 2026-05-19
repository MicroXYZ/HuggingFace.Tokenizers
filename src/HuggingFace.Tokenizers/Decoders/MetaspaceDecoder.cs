using System.Diagnostics.CodeAnalysis;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Decoders;

/// <summary>
/// SentencePiece metaspace 解码器。
/// 将替换字符（默认 ▁, U+2581）替换回空格。
/// </summary>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
[TokenizerComponent("Metaspace")]
public sealed class MetaspaceDecoder : IDecoder
{
    private readonly char _replacement;
    private readonly PrependScheme _prependScheme;
    private readonly bool _split;

    /// <summary>替换字符（默认 ▁, U+2581）。</summary>
    public char Replacement => _replacement;

    /// <summary>前导空格方案。</summary>
    [JsonKey("prepend_scheme")]
    public PrependScheme PrependSchemeValue => _prependScheme;

    /// <summary>是否在替换字符处拆分。</summary>
    public bool Split => _split;

    /// <summary>
    /// 初始化 <see cref="MetaspaceDecoder"/> 的新实例。
    /// </summary>
    /// <param name="replacement">替换字符。</param>
    /// <param name="prependScheme">前导空格方案。</param>
    /// <param name="split">是否在替换字符处拆分。</param>
    public MetaspaceDecoder(
        char replacement = '\u2581',
        PrependScheme prependScheme = PrependScheme.Always,
        bool split = true)
    {
        _replacement = replacement;
        _prependScheme = prependScheme;
        _split = split;
    }

    /// <inheritdoc/>
    public IReadOnlyList<string> DecodeChain(IReadOnlyList<string> tokens)
    {
        var result = new string[tokens.Count];
        for (int i = 0; i < tokens.Count; i++)
        {
            var token = tokens[i];
            ReadOnlySpan<char> span = token.AsSpan();

            // 替换字符 → 空格（string.Create 避免 stackalloc 循环内分配）
            string tokenCopy = token;
            char replacement = _replacement;
            result[i] = string.Create(span.Length, (tokenCopy, replacement), static (dest, s) =>
            {
                int written = 0;
                foreach (char c in s.tokenCopy)
                    dest[written++] = c == s.replacement ? ' ' : c;
            });

            if (i == 0)
            {
                var decoded = result[i];
                switch (_prependScheme)
                {
                    case PrependScheme.Always:
                        if (!decoded.StartsWith(' '))
                            result[i] = PrependSpace(decoded);
                        break;
                    case PrependScheme.First:
                        if (!token.StartsWith(_replacement) && !decoded.StartsWith(' '))
                            result[i] = PrependSpace(decoded);
                        break;
                    case PrependScheme.Never:
                        result[i] = decoded.TrimStart(' ');
                        break;
                }
            }
        }
        return result;
    }

    private static string PrependSpace(string decoded)
    {
        return string.Create(1 + decoded.Length, decoded, (span, d) =>
        {
            span[0] = ' ';
            d.AsSpan().CopyTo(span.Slice(1));
        });
    }

    /// <inheritdoc/>
    public string Decode(IReadOnlyList<string> tokens)
    {
        return DecoderHelper.JoinTokens(DecodeChain(tokens));
    }
}
