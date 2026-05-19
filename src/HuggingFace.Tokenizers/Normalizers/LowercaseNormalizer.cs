using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Normalizers;

/// <summary>
/// 使用固定区域性规则将文本转换为小写来标准化。
/// </summary>
[TokenizerComponent("Lowercase")]
public sealed class LowercaseNormalizer : INormalizer
{
    /// <summary>
    /// 标准化给定的 <see cref="NormalizedString"/> 通过小写转换。
    /// </summary>
    /// <param name="normalized">The string to normalize in-place.</param>
    public void Normalize(NormalizedString normalized) => normalized.Lowercase();
}
