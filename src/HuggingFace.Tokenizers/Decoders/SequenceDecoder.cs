using System.Diagnostics.CodeAnalysis;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Decoders;

/// <summary>
/// 链式组合多个 <see cref="IDecoder"/> 实例。
/// 每个解码器的输出成为下一个的输入。
/// </summary>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
[TokenizerComponent("Sequence")]
public sealed class SequenceDecoder : IDecoder
{
    private readonly IReadOnlyList<IDecoder> _decoders;

    /// <summary>此序列中的解码器列表。</summary>
    public IReadOnlyList<IDecoder> Decoders => _decoders;

    /// <summary>
    /// 初始化 <see cref="SequenceDecoder"/> 类的新实例。
    /// </summary>
    /// <param name="decoders">要链式组合的解码器序列。</param>
    /// <exception cref="ArgumentNullException"><paramref name="decoders"/> 为 null。</exception>
    public SequenceDecoder(IReadOnlyList<IDecoder> decoders)
    {
        ArgumentNullException.ThrowIfNull(decoders);
        _decoders = decoders;
    }

    /// <inheritdoc/>
    public IReadOnlyList<string> DecodeChain(IReadOnlyList<string> tokens)
    {
        var current = tokens;
        foreach (var decoder in _decoders)
        {
            current = decoder.DecodeChain(current);
        }
        return current;
    }

    /// <inheritdoc/>
    public string Decode(IReadOnlyList<string> tokens)
    {
        var current = tokens;
        for (int i = 0; i < _decoders.Count; i++)
        {
            if (i == _decoders.Count - 1)
            {
                // 最后一个解码器：使用 Decode 返回最终字符串输出
                return _decoders[i].Decode(current);
            }
            current = _decoders[i].DecodeChain(current);
        }

        // 正常情况下不会到达此处，除非解码器列表为空
        return DecoderHelper.JoinTokens(tokens);
    }
}
