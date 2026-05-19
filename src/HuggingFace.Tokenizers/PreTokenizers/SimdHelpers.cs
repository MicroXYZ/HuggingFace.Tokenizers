using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace HuggingFace.Tokenizers.PreTokenizers;

/// <summary>
/// 跨平台 SIMD 加速辅助方法，用于预分词器热路径。
/// 使用 Vector&lt;T&gt; 自动利用最宽可用的 SIMD：
///   - AVX-512 (512-bit) → AVX2 (256-bit) → SSE (128-bit) → NEON (128-bit) → scalar fallback
///
/// 注意： Vector&lt;T&gt; 不直接支持 char。使用 Vector&lt;ushort&gt; 并重新解释。
/// </summary>
internal static class SimdHelpers
{
    /// <summary>
    /// 查找 span 中目标字节所有出现的索引。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int FindByteIndices(ReadOnlySpan<byte> data, byte target, Span<int> results)
    {
        int count = 0;
        int i = 0;
        int vecCount = Vector<byte>.Count;

        if (Vector.IsHardwareAccelerated && data.Length >= vecCount)
        {
            var targetVec = new Vector<byte>(target);
            int simdEnd = data.Length - vecCount + 1;

            for (; i < simdEnd; i += vecCount)
            {
                var chunk = new Vector<byte>(data.Slice(i, vecCount));
                var eq = Vector.Equals(chunk, targetVec);

                // Fast skip if no matches in this chunk
                if (eq == Vector<byte>.Zero) continue;

                // Scalar check for exact positions
                for (int j = 0; j < vecCount; j++)
                {
                    if (data[i + j] == target && count < results.Length)
                        results[count++] = i + j;
                }
            }
        }

        // Scalar tail
        for (; i < data.Length; i++)
        {
            if (data[i] == target && count < results.Length)
                results[count++] = i;
        }

        return count;
    }

    /// <summary>
    /// SIMD-accelerated count of a specific byte value in a span.
    /// 使用 Vector&lt;byte&gt; + Vector.Dot 技巧进行高效计数。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int CountByte(ReadOnlySpan<byte> data, byte target)
    {
        int count = 0;
        int i = 0;
        int vecCount = Vector<byte>.Count;

        if (Vector.IsHardwareAccelerated && data.Length >= vecCount)
        {
            var targetVec = new Vector<byte>(target);
            var ones = Vector<byte>.One;
            int simdEnd = data.Length - vecCount + 1;

            for (; i < simdEnd; i += vecCount)
            {
                var chunk = new Vector<byte>(data.Slice(i, vecCount));
                var eq = Vector.Equals(chunk, targetVec);

                if (eq == Vector<byte>.Zero) continue;

                // Count matches: eq has 0xFF (255) for matches, Dot with ones sums all bytes.
                // Each match contributes 255, so divide by 255 to get actual count.
                count += (int)Vector.Dot(eq, ones) / 255;
            }
        }

        // Scalar tail
        for (; i < data.Length; i++)
        {
            if (data[i] == target) count++;
        }

        return count;
    }

    /// <summary>
    /// 检查当前硬件是否支持 SIMD。
    /// </summary>
    public static bool IsSimdAvailable => Vector.IsHardwareAccelerated;
}
