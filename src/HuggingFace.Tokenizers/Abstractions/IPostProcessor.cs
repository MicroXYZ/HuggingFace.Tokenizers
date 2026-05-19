namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 后处理编码结果，添加特殊 token（CLS、SEP 等）。
/// 分词管道的组成部分。
/// </summary>
public interface IPostProcessor
{
    /// <summary>
    /// 返回处理过程中添加的 token 数量。
    /// </summary>
    /// <param name="isPair">是否为双序列输入。</param>
    /// <returns>添加的 token 数量。</returns>
    int AddedTokens(bool isPair);

    /// <summary>
    /// 处理单个编码，可选配对编码。
    /// </summary>
    /// <param name="encoding">主序列编码。</param>
    /// <param name="pairEncoding">配对序列编码，可为 null。</param>
    /// <param name="addSpecialTokens">是否添加特殊 token。</param>
    /// <returns>处理后的编码。</returns>
    Encoding Process(Encoding encoding, Encoding? pairEncoding, bool addSpecialTokens);

    /// <summary>
    /// 批量处理多个编码。
    /// </summary>
    /// <param name="encodings">编码列表。</param>
    /// <param name="addSpecialTokens">是否添加特殊 token。</param>
    /// <returns>处理后的编码列表。</returns>
    IReadOnlyList<Encoding> ProcessEncodings(IReadOnlyList<Encoding> encodings, bool addSpecialTokens);
}
