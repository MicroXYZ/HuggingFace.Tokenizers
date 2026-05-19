using System.Diagnostics.CodeAnalysis;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Decoders;

/// <summary>
/// 合并解码器，简单地将所有 token 连接在一起，不使用任何分隔符。
/// </summary>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
[TokenizerComponent("Fuse")]
public sealed class FuseDecoder : IDecoder
{
    /// <inheritdoc/>
    public IReadOnlyList<string> DecodeChain(IReadOnlyList<string> tokens)
    {
        // 返回单个融合 token
        return new[] { DecoderHelper.JoinTokens(tokens) };
    }

    /// <inheritdoc/>
    public string Decode(IReadOnlyList<string> tokens)
    {
        return DecoderHelper.JoinTokens(tokens);
    }
}
