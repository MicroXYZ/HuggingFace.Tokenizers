using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Normalizers;

/// <summary>
/// 链式组合多个 <see cref="INormalizer"/> 实例，按顺序应用。
/// 每个标准化器处理前一个的输出。
/// </summary>
[TokenizerComponent("Sequence")]
public sealed class SequenceNormalizer : INormalizer
{
    private readonly IReadOnlyList<INormalizer> _normalizers;

    /// <summary>此序列中的标准化器列表。</summary>
    public IReadOnlyList<INormalizer> Normalizers => _normalizers;

    /// <summary>
    /// 创建新的 <see cref="SequenceNormalizer"/>.
    /// </summary>
    /// <param name="normalizers">The normalizers to apply in order. Must not be null 或 empty.</param>
    /// <exception cref="ArgumentNullException"><paramref name="normalizers"/> is null.</exception>
    public SequenceNormalizer(IReadOnlyList<INormalizer> normalizers)
    {
        ArgumentNullException.ThrowIfNull(normalizers);
        _normalizers = normalizers;
    }

    /// <summary>
    /// 按顺序将每个标准化器应用于给定的 <see cref="NormalizedString"/>.
    /// </summary>
    /// <param name="normalized">The string to normalize in-place.</param>
    public void Normalize(NormalizedString normalized)
    {
        foreach (var normalizer in _normalizers)
        {
            normalizer.Normalize(normalized);
        }
    }
}
