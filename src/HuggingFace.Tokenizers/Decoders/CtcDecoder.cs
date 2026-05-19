using System.Diagnostics.CodeAnalysis;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Decoders;

/// <summary>
/// CTC（连接主义时序分类）解码器。
/// 处理空白/填充 token 并折叠连续重复字符。
/// </summary>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
[TokenizerComponent("CTC")]
public sealed class CtcDecoder : IDecoder
{
    private readonly ReadOnlyMemory<char> _padToken;
    private readonly ReadOnlyMemory<char> _wordDelimiter;
    private readonly bool _cleanup;

    [SkipSerialization]
    /// <summary>填充 token 字符串。</summary>
    public string PadToken => _padToken.ToString();
    /// <summary>单词分隔符字符串。</summary>
    public string WordDelimiter => _wordDelimiter.ToString();
    /// <summary>是否清理标点间距和缩写。</summary>
    public bool Cleanup => _cleanup;

    /// <summary>
    /// 初始化 <see cref="CtcDecoder"/> 的新实例。
    /// </summary>
    /// <param name="padToken">填充 token 字符串。</param>
    /// <param name="wordDelimiter">单词分隔符字符串。</param>
    /// <param name="cleanup">是否清理标点间距和缩写。</param>
    public CtcDecoder(string padToken = "<pad>", string wordDelimiter = "|", bool cleanup = true)
    {
        _padToken = padToken.AsMemory();
        _wordDelimiter = wordDelimiter.AsMemory();
        _cleanup = cleanup;
    }

    /// <inheritdoc/>
    public IReadOnlyList<string> DecodeChain(IReadOnlyList<string> tokens)
    {
        var result = new List<string>(tokens.Count);
        string? previous = null;
        ReadOnlySpan<char> padSpan = _padToken.Span;
        ReadOnlySpan<char> delimSpan = _wordDelimiter.Span;

        foreach (var token in tokens)
        {
            if (token == previous)
                continue;
            previous = token;

            // span 级 Replace：移除 padToken
            var replaced = StringTransforms.SpanReplace(token.AsSpan(), padSpan, ReadOnlySpan<char>.Empty);

            if (_cleanup)
            {
                replaced = WordPieceCleanup(replaced);
                // span 级 Replace：wordDelimiter → " "
                replaced = StringTransforms.SpanReplace(replaced.AsSpan(), delimSpan, " ".AsSpan());
            }

            if (replaced.Length > 0)
                result.Add(replaced);
        }

        return result;
    }

    /// <summary>
    /// 与 Rust wordpiece::cleanup 一致：处理标点间距和英语缩写。
    /// </summary>
    private static string WordPieceCleanup(string input)
    {
        var text = TextCleanup.CleanContractions(input);
        return text
            .Replace(" .", ".")
            .Replace(" ?", "?")
            .Replace(" !", "!")
            .Replace(" ,", ",")
            .Replace(" ' ", "'");
    }

    /// <inheritdoc/>
    public string Decode(IReadOnlyList<string> tokens)
    {
        return DecoderHelper.JoinTokens(DecodeChain(tokens));
    }
}
