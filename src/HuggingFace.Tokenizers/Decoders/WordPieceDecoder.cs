using System.Diagnostics.CodeAnalysis;
using System.Text;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Decoders;

/// <summary>
/// WordPiece 解码器（BERT 风格）。
/// 从子词 token 中移除连续前缀（如 "##"）并清理标点间距和缩写。
/// </summary>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
[TokenizerComponent("WordPiece")]
public sealed class WordPieceDecoder : IDecoder
{
    private readonly ReadOnlyMemory<char> _prefix;
    private readonly bool _cleanup;

    /// <summary>子词前缀（如 "##"）。</summary>
    public string Prefix => _prefix.ToString();
    /// <summary>是否清理标点间距和缩写。</summary>
    public bool Cleanup => _cleanup;

    public WordPieceDecoder(string prefix = "##", bool cleanup = true)
    {
        _prefix = prefix.AsMemory();
        _cleanup = cleanup;
    }

    /// <inheritdoc/>
    public IReadOnlyList<string> DecodeChain(IReadOnlyList<string> tokens)
    {
        var result = new string[tokens.Count];
        ReadOnlySpan<char> prefixSpan = _prefix.Span;
        for (int i = 0; i < tokens.Count; i++)
        {
            var token = tokens[i];
            // Span 切片去除 prefix（零分配）
            if (token.AsSpan().StartsWith(prefixSpan))
                token = token[_prefix.Length..];
            else if (i > 0)
                token = " " + token;

            if (_cleanup)
                token = CleanupToken(token);

            result[i] = token;
        }
        return result;
    }

    /// <inheritdoc/>
    public string Decode(IReadOnlyList<string> tokens)
    {
        return DecoderHelper.JoinTokens(DecodeChain(tokens));
    }

    /// <summary>
    /// 清理标点间距和英语缩写（与 Rust wordpiece::cleanup 一致）。
    /// 使用 ValueStringBuilder 替代 StringBuilder，减少堆分配。
    /// </summary>
    private static string CleanupToken(string text)
    {
        text = TextCleanup.CleanContractions(text);

        var sb = new ValueStringBuilder(stackalloc char[text.Length]);
        bool lastWasSpace = false;

        for (int i = 0; i < text.Length; i++)
        {
            char ch = text[i];

            if (ch == ' ')
            {
                if (i + 1 < text.Length && IsPunctuationNoSpaceBefore(text[i + 1]))
                    continue;

                if (!lastWasSpace)
                {
                    sb.Append(ch);
                    lastWasSpace = true;
                }
            }
            else
            {
                sb.Append(ch);
                lastWasSpace = false;
            }
        }

        return sb.ToString();
    }

    private static bool IsPunctuationNoSpaceBefore(char c) =>
        c is '.' or ',' or '!' or '?' or ';' or ':' or ')' or ']' or '}' or
            '\'' or '"' or '/' or '\\' or '-' or '%';
}
