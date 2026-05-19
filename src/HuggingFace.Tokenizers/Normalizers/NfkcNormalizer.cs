using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Normalizers;

/// <summary>
/// 应用 Unicode NFKC（兼容分解，然后规范组合） 标准化文本。
/// </summary>
[TokenizerComponent("NFKC")]
public sealed class NfkcNormalizer : INormalizer
{
    /// <summary>
    /// 标准化给定的 <see cref="NormalizedString"/> 使用 Unicode NFKC 形式。
    /// </summary>
    /// <param name="normalized">The string to normalize in-place.</param>
    public void Normalize(NormalizedString normalized) => normalized.Nfkc();
}
