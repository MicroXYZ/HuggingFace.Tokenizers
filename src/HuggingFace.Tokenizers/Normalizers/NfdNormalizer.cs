using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Normalizers;

/// <summary>
/// 应用 Unicode NFD（规范分解） 标准化文本。
/// </summary>
[TokenizerComponent("NFD")]
public sealed class NfdNormalizer : INormalizer
{
    /// <summary>
    /// 标准化给定的 <see cref="NormalizedString"/> 使用 Unicode NFD 形式。
    /// </summary>
    /// <param name="normalized">The string to normalize in-place.</param>
    public void Normalize(NormalizedString normalized) => normalized.Nfd();
}
