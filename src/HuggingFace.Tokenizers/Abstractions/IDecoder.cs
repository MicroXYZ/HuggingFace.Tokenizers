namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 将 token ID/字符串解码回可读文本。
/// 分词管道的组成部分。
/// </summary>
public interface IDecoder
{
    /// <summary>
    /// 将 token 字符串列表解码为单个字符串。
    /// </summary>
    /// <param name="tokens">token 字符串列表。</param>
    /// <returns>解码后的文本。</returns>
    string Decode(IReadOnlyList<string> tokens);

    /// <summary>
    /// 将 token 解码为子部分链（合并前）。
    /// </summary>
    /// <param name="tokens">token 字符串列表。</param>
    /// <returns>解码后的子部分列表。</returns>
    IReadOnlyList<string> DecodeChain(IReadOnlyList<string> tokens);

    /// <summary>
    /// 将解码结果写入 Span 缓冲区，返回实际写入的字符数。
    /// 默认实现调用 <see cref="Decode"/> 后复制到目标 Span。
    /// 性能关键的 Decoder 可覆写此方法直接写入 Span，避免中间 string 分配。
    /// </summary>
    /// <param name="destination">目标缓冲区。</param>
    /// <param name="tokens">token 字符串列表。</param>
    /// <returns>实际写入的字符数。</returns>
    int DecodeTo(Span<char> destination, IReadOnlyList<string> tokens)
    {
        var result = Decode(tokens);
        result.AsSpan().CopyTo(destination);
        return result.Length;
    }

    /// <summary>
    /// 预估解码输出长度（字符数），供调用方预分配缓冲区。
    /// 默认实现调用 <see cref="Decode"/> 获取实际长度。
    /// 性能关键的 Decoder 可覆写此方法提供更高效的估算。
    /// </summary>
    /// <param name="tokens">token 字符串列表。</param>
    /// <returns>预估输出字符数。</returns>
    int GetDecodeLength(IReadOnlyList<string> tokens) => Decode(tokens).Length;
}
