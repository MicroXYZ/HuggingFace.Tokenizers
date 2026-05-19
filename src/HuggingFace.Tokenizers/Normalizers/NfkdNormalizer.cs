using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Normalizers;

/// <summary>
/// 应用 Unicode NFKD（兼容分解） 标准化文本。
/// </summary>
[TokenizerComponent("NFKD")]
public sealed class NfkdNormalizer : INormalizer
{
    /// <summary>
    /// 标准化给定的 <see cref="NormalizedString"/> 使用 Unicode NFKD 形式。
    /// </summary>
    /// <param name="normalized">The string to normalize in-place.</param>
    public void Normalize(NormalizedString normalized) => normalized.Nfkd();
}
