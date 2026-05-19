using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Normalizers;

/// <summary>
/// 通过分解为 NFD 从文本中移除变音符号（音标），
/// 并过滤掉 NonSpacingMark 字符。
/// </summary>
[TokenizerComponent("StripAccents")]
public sealed class StripAccentsNormalizer : INormalizer
{
    /// <summary>
    /// 对给定的 <see cref="NormalizedString"/> 移除变音符号。
    /// </summary>
    /// <param name="normalized">要原位标准化的字符串。</param>
    public void Normalize(NormalizedString normalized) => normalized.StripAccents();
}
