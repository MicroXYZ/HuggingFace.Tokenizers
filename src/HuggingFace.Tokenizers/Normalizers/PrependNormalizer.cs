using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Normalizers;

/// <summary>
/// 在文本开头插入固定字符串。
/// 常用于 SentencePiece 模型以插入 "▁" (U+2581) 字符。
/// </summary>
[TokenizerComponent("Prepend")]
public sealed class PrependNormalizer : INormalizer
{
    private readonly ReadOnlyMemory<char> _prepend;

    /// <summary>要前插的字符串。</summary>
    public string Prepend => _prepend.ToString();

    /// <summary>
    /// 创建新的 <see cref="PrependNormalizer"/>.
    /// </summary>
    /// <param name="prepend">The string to prepend. Must not be null.</param>
    public PrependNormalizer(string prepend)
    {
        ArgumentNullException.ThrowIfNull(prepend);
        _prepend = prepend.AsMemory();
    }

    /// <summary>
    /// 将配置的字符串插入到标准化文本前。
    /// </summary>
    /// <param name="normalized">The string to normalize in-place.</param>
    public void Normalize(NormalizedString normalized) => normalized.Prepend(_prepend.Span);
}
