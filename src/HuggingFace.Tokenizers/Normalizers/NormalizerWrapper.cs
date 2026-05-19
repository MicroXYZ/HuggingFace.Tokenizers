using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Normalizers;

/// <summary>
/// 包装器，将标准化委托给内部的 <see cref="INormalizer"/>.
/// 适用于在标准化器周围添加额外行为（日志、验证等），同时保持 <see cref="INormalizer"/> 契约。
/// </summary>
[TokenizerComponent("NormalizerWrapper")]
public sealed class NormalizerWrapper : INormalizer
{
    private readonly INormalizer _inner;

    /// <summary>
    /// 创建新的 <see cref="NormalizerWrapper"/>.
    /// </summary>
    /// <param name="inner">被包装的内部标准化器，不能为 null。</param>
    public NormalizerWrapper(INormalizer inner)
    {
        ArgumentNullException.ThrowIfNull(inner);
        _inner = inner;
    }

    /// <summary>
    /// 获取被包装的内部标准化器。
    /// </summary>
    public INormalizer Inner => _inner;

    /// <summary>
    /// 将标准化委托给内部标准化器。
    /// </summary>
    /// <param name="normalized">要就地标准化的字符串。</param>
    public void Normalize(NormalizedString normalized)
    {
        _inner.Normalize(normalized);
    }
}
