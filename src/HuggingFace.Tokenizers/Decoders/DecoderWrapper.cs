using System.Diagnostics.CodeAnalysis;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Decoders;

/// <summary>
/// A wrapper that delegates all decoding operations to an inner <see cref="IDecoder"/>.
/// 适用于在任何解码器周围添加横切关注点（日志、指标等）。
/// </summary>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
[TokenizerComponent("DecoderWrapper")]
public sealed class DecoderWrapper : IDecoder
{
    private readonly IDecoder _inner;

    /// <summary>
    /// 初始化新的 instance of the <see cref="DecoderWrapper"/> 类的新实例。
    /// </summary>
    /// <param name="inner">The decoder to delegate to.</param>
    /// <exception cref="ArgumentNullException"><paramref name="inner"/> is null.</exception>
    public DecoderWrapper(IDecoder inner)
    {
        ArgumentNullException.ThrowIfNull(inner);
        _inner = inner;
    }

    /// <summary>
    /// 获取被包装的内部解码器。
    /// </summary>
    public IDecoder Inner => _inner;

    /// <inheritdoc/>
    public IReadOnlyList<string> DecodeChain(IReadOnlyList<string> tokens)
    {
        return _inner.DecodeChain(tokens);
    }

    /// <inheritdoc/>
    public string Decode(IReadOnlyList<string> tokens)
    {
        return _inner.Decode(tokens);
    }
}
