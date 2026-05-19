using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.PreTokenizers;

/// <summary>
/// 仅按空白边界拆分文本的预分词器，
/// without further splitting on punctuation.
/// 使用 cross-platform SIMD-accelerated whitespace detection for large inputs.
/// </summary>
[TokenizerComponent("WhitespaceSplit")]
public sealed class WhitespaceSplitPreTokenizer : IPreTokenizer
{
    /// <summary>
    /// 单例实例，方便使用。
    /// </summary>
    public static readonly WhitespaceSplitPreTokenizer Instance = new();

    /// <inheritdoc />
    public void PreTokenize(PreTokenizedString pretokenized)
    {
        pretokenized.Split((_, normalized) =>
        {
            var text = normalized.GetSpan();
            if (text.IsEmpty)
                return Enumerable.Empty<NormalizedString>();

            var parts = new List<NormalizedString>();
            int start = FindNextNonWhitespace(text, 0);

            while (start < text.Length)
            {
                int end = FindNextWhitespace(text, start + 1);
                parts.Add(normalized.Slice(start, end - start));
                start = FindNextNonWhitespace(text, end);
            }

            return parts;
        });
    }

    /// <summary>
    /// 使用跨平台 SIMD 查找下一个非空白字符。
    /// 使用 Vector&lt;ushort&gt; （char 为 2 字节）自动利用 AVX-512/AVX2/SSE/NEON。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static int FindNextNonWhitespace(ReadOnlySpan<char> text, int offset)
    {
        int i = offset;
        int vecCount = Vector<ushort>.Count;

        if (Vector.IsHardwareAccelerated && text.Length - i >= vecCount)
        {
            ReadOnlySpan<ushort> data = MemoryMarshal.Cast<char, ushort>(text);
            var spaceVec = new Vector<ushort>(' ');
            var tabVec = new Vector<ushort>('\t');
            var nlVec = new Vector<ushort>('\n');
            var crVec = new Vector<ushort>('\r');
            int simdEnd = data.Length - vecCount + 1;

            for (; i < simdEnd; i += vecCount)
            {
                var chunk = new Vector<ushort>(data.Slice(i, vecCount));
                var isWs = Vector.Equals(chunk, spaceVec)
                         | Vector.Equals(chunk, tabVec)
                         | Vector.Equals(chunk, nlVec)
                         | Vector.Equals(chunk, crVec);

                // Fast skip if all are non-whitespace (no whitespace matches)
                if (isWs == Vector<ushort>.Zero) return i;

                // Some whitespace found — locate first non-whitespace lane
                for (int j = 0; j < vecCount; j++)
                {
                    if (!char.IsWhiteSpace((char)data[i + j]))
                        return i + j;
                }
            }
        }

        // Scalar tail
        for (; i < text.Length; i++)
        {
            if (!char.IsWhiteSpace(text[i]))
                return i;
        }

        return text.Length;
    }

    /// <summary>
    /// 使用跨平台 SIMD 查找下一个空白字符。
    /// 使用 Vector&lt;ushort&gt; （char 为 2 字节）自动利用 AVX-512/AVX2/SSE/NEON。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static int FindNextWhitespace(ReadOnlySpan<char> text, int offset)
    {
        int i = offset;
        int vecCount = Vector<ushort>.Count;

        if (Vector.IsHardwareAccelerated && text.Length - i >= vecCount)
        {
            ReadOnlySpan<ushort> data = MemoryMarshal.Cast<char, ushort>(text);
            var spaceVec = new Vector<ushort>(' ');
            var tabVec = new Vector<ushort>('\t');
            var nlVec = new Vector<ushort>('\n');
            var crVec = new Vector<ushort>('\r');
            int simdEnd = data.Length - vecCount + 1;

            for (; i < simdEnd; i += vecCount)
            {
                var chunk = new Vector<ushort>(data.Slice(i, vecCount));
                var isWs = Vector.Equals(chunk, spaceVec)
                         | Vector.Equals(chunk, tabVec)
                         | Vector.Equals(chunk, nlVec)
                         | Vector.Equals(chunk, crVec);

                // Fast skip if no whitespace in this chunk
                if (isWs == Vector<ushort>.Zero) continue;

                // Whitespace found — locate first whitespace lane
                for (int j = 0; j < vecCount; j++)
                {
                    if (char.IsWhiteSpace((char)data[i + j]))
                        return i + j;
                }
            }
        }

        // Scalar tail
        for (; i < text.Length; i++)
        {
            if (char.IsWhiteSpace(text[i]))
                return i;
        }

        return text.Length;
    }
}
