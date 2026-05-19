using System.Diagnostics.CodeAnalysis;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Decoders;

/// <summary>
/// BPE（字节对编码）解码器。
/// 将每个 token 的可选后缀替换为空格（最后一个 token 为空）。
/// </summary>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
[TokenizerComponent("BPEDecoder")]
public sealed class BpeDecoder : IDecoder
{
    private readonly ReadOnlyMemory<char> _suffix;

    public string Suffix => _suffix.ToString();

    public BpeDecoder(string suffix = "</w>")
    {
        _suffix = suffix.AsMemory();
    }

    /// <inheritdoc/>
    public IReadOnlyList<string> DecodeChain(IReadOnlyList<string> tokens)
    {
        if (_suffix.Length == 0 || tokens.Count == 0)
            return tokens;

        var result = new string[tokens.Count];
        ReadOnlySpan<char> suffixSpan = _suffix.Span;
        int last = tokens.Count - 1;

        for (int i = 0; i <= last; i++)
        {
            var token = tokens[i];
            ReadOnlySpan<char> span = token.AsSpan();
            ReadOnlySpan<char> replacement = i == last ? ReadOnlySpan<char>.Empty : " ".AsSpan();

            // Span 切片替换：查找 suffix 位置，拼接前缀 + 替换 + 后缀
            int idx = span.IndexOf(suffixSpan);
            if (idx < 0)
            {
                result[i] = token;
                continue;
            }

            // 预计算长度后 string.Create 一次性构建
            int newLen = idx + replacement.Length + (span.Length - idx - suffixSpan.Length);
            string tokenCopy = token;
            result[i] = string.Create(newLen, (tokenCopy, idx, suffixLen: suffixSpan.Length, replacementStr: replacement.ToString()), static (dest, s) =>
            {
                s.tokenCopy.AsSpan().Slice(0, s.idx).CopyTo(dest);
                int pos = s.idx;
                s.replacementStr.AsSpan().CopyTo(dest.Slice(pos));
                pos += s.replacementStr.Length;
                s.tokenCopy.AsSpan().Slice(s.idx + s.suffixLen).CopyTo(dest.Slice(pos));
            });
        }
        return result;
    }

    /// <inheritdoc/>
    public string Decode(IReadOnlyList<string> tokens)
    {
        return DecoderHelper.JoinTokens(DecodeChain(tokens));
    }
}
