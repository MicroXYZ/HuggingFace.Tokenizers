using System.Diagnostics.CodeAnalysis;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Decoders;

/// <summary>
/// 去除解码器，从 token 的首尾移除特定字符。
/// 内容类型为 char（与 Rust Strip 实现一致）。
/// </summary>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
[TokenizerComponent("Strip")]
public sealed class StripDecoder : IDecoder
{
    private readonly char _content;
    private readonly int _start;
    private readonly int _stop;

    /// <summary>要去除的字符。</summary>
    public char Content => _content;

    /// <summary>从开头去除的字符数。</summary>
    public int Start => _start;

    /// <summary>从末尾去除的字符数。</summary>
    public int Stop => _stop;

    /// <summary>
    /// 初始化 <see cref="StripDecoder"/> 类的新实例。
    /// </summary>
    /// <param name="content">要去除的字符。默认为空格 (' ')。</param>
    /// <param name="start">从开头去除的字符数。默认为 1。</param>
    /// <param name="stop">从末尾去除的字符数。默认为 1。</param>
    public StripDecoder(char content = '\0', int start = 0, int stop = 0)
    {
        _content = content;
        _start = start;
        _stop = stop;
    }

    /// <inheritdoc/>
    public IReadOnlyList<string> DecodeChain(IReadOnlyList<string> tokens)
    {
        var result = new string[tokens.Count];
        for (int i = 0; i < tokens.Count; i++)
        {
            result[i] = StripToken(tokens[i]);
        }
        return result;
    }

    /// <inheritdoc/>
    public string Decode(IReadOnlyList<string> tokens)
    {
        return DecoderHelper.JoinTokens(DecodeChain(tokens));
    }

    /// <summary>
    /// 从 token 的首尾去除内容字符。
    /// 与 Rust Strip::decode_chain 完全一致：
    /// - 从开头最多去除 _start 个匹配字符（遇到不匹配时停止）
    /// - 从末尾最多去除 _stop 个匹配字符（遇到不匹配时停止）
    /// </summary>
    private string StripToken(string token)
    {
        if (string.IsNullOrEmpty(token))
            return token;

        ReadOnlySpan<char> span = token.AsSpan();

        // 从开头去除
        int startCut = 0;
        for (int i = 0; i < _start && i < span.Length; i++)
        {
            if (span[i] == _content) startCut = i + 1;
            else break;
        }

        // 从末尾去除
        int stopCut = span.Length;
        for (int i = 0; i < _stop; i++)
        {
            int index = span.Length - i - 1;
            if (index < startCut) break;
            if (span[index] == _content) stopCut = index;
            else break;
        }

        if (startCut >= stopCut)
            return string.Empty;

        return span.Slice(startCut, stopCut - startCut).ToString();
    }
}
