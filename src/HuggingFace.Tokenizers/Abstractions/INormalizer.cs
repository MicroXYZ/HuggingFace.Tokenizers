namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 标准化输入文本（如 Unicode 规范化、小写转换、去除音标）。
/// 分词管道的组成部分。
/// </summary>
public interface INormalizer
{
    /// <summary>
    /// 就地标准化给定的 <see cref="NormalizedString"/>。
    /// </summary>
    /// <param name="normalized">要标准化的字符串。</param>
    void Normalize(NormalizedString normalized);
}
