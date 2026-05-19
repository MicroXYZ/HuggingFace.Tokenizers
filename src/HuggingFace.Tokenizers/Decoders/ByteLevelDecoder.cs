using System.Buffers;
using System.Diagnostics.CodeAnalysis;
using System.Text;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Decoders;

/// <summary>
/// GPT-2 字节级解码器。
/// 将字节级编码的 token 转换回可读文本。
/// </summary>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
[TokenizerComponent("ByteLevel")]
public sealed class ByteLevelDecoder : IDecoder
{

    /// <inheritdoc/>
    public IReadOnlyList<string> DecodeChain(IReadOnlyList<string> tokens)
    {
        var result = new string[tokens.Count];
        // 计算最大 token 长度以复用缓冲区
        int maxLen = 0;
        foreach (var t in tokens) maxLen = Math.Max(maxLen, t.Length);

        char[]? pooled = null;
        Span<char> buf = maxLen <= 256
            ? stackalloc char[maxLen]
            : (pooled = ArrayPool<char>.Shared.Rent(maxLen));
        try
        {
            for (int t = 0; t < tokens.Count; t++)
            {
                var token = tokens[t];
                int written = 0;
                foreach (var ch in token)
                    buf[written++] = ByteLevelMapping.CharToByte.TryGetValue(ch, out byte b) ? (char)b : ch;
                result[t] = new string(buf.Slice(0, written));
            }
        }
        finally
        {
            if (pooled is not null) ArrayPool<char>.Shared.Return(pooled);
        }
        return result;
    }

    /// <inheritdoc/>
    public string Decode(IReadOnlyList<string> tokens)
    {
        int totalLen = 0;
        foreach (var t in tokens) totalLen += t.Length;

        byte[]? pooled = null;
        Span<byte> buf = totalLen <= 1024
            ? stackalloc byte[totalLen]
            : (pooled = ArrayPool<byte>.Shared.Rent(totalLen));
        try
        {
            int written = 0;
            foreach (var token in tokens)
            {
                foreach (var ch in token)
                {
                    if (ByteLevelMapping.CharToByte.TryGetValue(ch, out byte b))
                        buf[written++] = b;
                    else
                    {
                        // 未识别字符：UTF-8 编码作为字节追加
                        written += global::System.Text.Encoding.UTF8.GetBytes(new ReadOnlySpan<char>(in ch), buf.Slice(written));
                    }
                }
            }
            return global::System.Text.Encoding.UTF8.GetString(buf.Slice(0, written));
        }
        finally
        {
            if (pooled is not null)
                ArrayPool<byte>.Shared.Return(pooled);
        }
    }
}
