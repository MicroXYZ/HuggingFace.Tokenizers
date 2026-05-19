namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 通过将文本拆分为初始词/片段进行预分词，在模型分词之前执行。
/// 分词管道的组成部分。
/// </summary>
public interface IPreTokenizer
{
    /// <summary>
    /// 就地预分词给定的 <see cref="PreTokenizedString"/>。
    /// </summary>
    /// <param name="pretokenized">要预分词的字符串。</param>
    void PreTokenize(PreTokenizedString pretokenized);
}
