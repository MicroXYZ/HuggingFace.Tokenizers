using System.Buffers;
using System.Diagnostics.CodeAnalysis;
using System.Text;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Decoders;

/// <summary>
/// 字节回退解码器（LLaMA 风格）。
/// 将类似 "&lt;0xXX&gt;"（十六进制字节表示）的 token 转换回字节，
/// 然后使用 UTF-8 将所得字节序列解码为字符串。
/// </summary>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
[TokenizerComponent("ByteFallback")]
public sealed class ByteFallbackDecoder : IDecoder
{
    /// <inheritdoc/>
    public IReadOnlyList<string> DecodeChain(IReadOnlyList<string> tokens)
    {
        var result = new List<string>();
        byte[]? pooledBytes = null;
        Span<byte> byteBuf = stackalloc byte[1024];
        int byteCount = 0;

        foreach (var token in tokens)
        {
            // S3: 使用 RuneHelpers 零分配解析替代正则表达式
            if (RuneHelpers.TryParseByteFallbackToken(token, out byte b))
            {
                if (byteCount < byteBuf.Length)
                    byteBuf[byteCount++] = b;
                else
                {
                    if (pooledBytes is null)
                    {
                        pooledBytes = ArrayPool<byte>.Shared.Rent(byteCount + 1);
                        byteBuf.Slice(0, byteCount).CopyTo(pooledBytes);
                    }
                    else if (pooledBytes.Length <= byteCount)
                    {
                        var larger = ArrayPool<byte>.Shared.Rent(byteCount + 1);
                        pooledBytes.AsSpan(0, byteCount).CopyTo(larger);
                        ArrayPool<byte>.Shared.Return(pooledBytes);
                        pooledBytes = larger;
                    }
                    pooledBytes[byteCount++] = b;
                }
            }
            else
            {
                if (byteCount > 0)
                {
                    var bytes = pooledBytes is not null ? pooledBytes.AsSpan(0, byteCount) : byteBuf.Slice(0, byteCount);
                    result.Add(DecodeBytesWithFallback(bytes));
                    byteCount = 0;
                }
                result.Add(token);
            }
        }

        if (byteCount > 0)
        {
            var bytes = pooledBytes is not null ? pooledBytes.AsSpan(0, byteCount) : byteBuf.Slice(0, byteCount);
            result.Add(DecodeBytesWithFallback(bytes));
        }

        if (pooledBytes is not null)
            ArrayPool<byte>.Shared.Return(pooledBytes);

        return result;
    }

    /// <summary>
    /// 手动解码字节序列，对每个无效字节输出一个 U+FFFD。
    /// 匹配 Rust 的行为：每个无效字节单独生成一个 U+FFFD。
    /// </summary>
    private static string DecodeBytesWithFallback(ReadOnlySpan<byte> bytes)
    {
        char[]? pooledChars = null;
        Span<char> buf = bytes.Length <= 512
            ? stackalloc char[bytes.Length]
            : (pooledChars = ArrayPool<char>.Shared.Rent(bytes.Length));
        try
        {
            int written = 0;
            int i = 0;
            while (i < bytes.Length)
            {
                byte currentByte = bytes[i];
                int charLen;

                if (currentByte <= 0x7F)
                {
                    buf[written++] = (char)currentByte;
                    i++;
                    continue;
                }
                else if ((currentByte & 0xE0) == 0xC0) charLen = 2;
                else if ((currentByte & 0xF0) == 0xE0) charLen = 3;
                else if ((currentByte & 0xF8) == 0xF0) charLen = 4;
                else
                {
                    buf[written++] = '\uFFFD';
                    i++;
                    continue;
                }

                if (i + charLen > bytes.Length)
                {
                    for (int j = i; j < bytes.Length; j++)
                        buf[written++] = '\uFFFD';
                    break;
                }

                bool valid = true;
                for (int j = 1; j < charLen; j++)
                {
                    if ((bytes[i + j] & 0xC0) != 0x80) { valid = false; break; }
                }

                if (valid)
                {
                    // 使用 Span 避免堆分配
                    ReadOnlySpan<byte> charBytes = bytes.Slice(i, charLen);
                    var str = global::System.Text.Encoding.UTF8.GetString(charBytes);
                    if (str.Length == 1 && str[0] == '\uFFFD')
                    {
                        for (int j = 0; j < charLen; j++)
                            buf[written++] = '\uFFFD';
                    }
                    else
                    {
                        foreach (var c in str)
                            buf[written++] = c;
                    }
                    i += charLen;
                }
                else
                {
                    buf[written++] = '\uFFFD';
                    i++;
                }
            }
            return new string(buf.Slice(0, written));
        }
        finally
        {
            if (pooledChars is not null)
                ArrayPool<char>.Shared.Return(pooledChars);
        }
    }

    /// <inheritdoc/>
    public string Decode(IReadOnlyList<string> tokens)
    {
        return DecoderHelper.JoinTokens(DecodeChain(tokens));
    }
}
