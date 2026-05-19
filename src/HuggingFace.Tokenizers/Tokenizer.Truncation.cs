using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// Tokenizer 的截断相关方法（partial class 拆分）。
/// </summary>
public sealed partial class Tokenizer
{
    private (Encoding First, Encoding? Second) Truncate(
        Encoding encoding,
        Encoding? pairEncoding,
        TruncationParams truncation)
    {
        if (truncation is null) return (encoding, pairEncoding);

        int maxLen = truncation.MaxLength;
        int stride = truncation.Stride;

        int totalLen = encoding.Length + (pairEncoding?.Length ?? 0);
        if (totalLen <= maxLen) return (encoding, pairEncoding);

        int toRemove = totalLen - maxLen;

        switch (truncation.Strategy)
        {
            case TruncationStrategy.OnlyFirst:
                encoding = TruncateEncoding(encoding, encoding.Length - toRemove, stride, truncation.Direction);
                break;
            case TruncationStrategy.OnlySecond:
                if (pairEncoding is null)
                    break; // No pair to truncate — nothing to do (matches Rust: SecondSequenceNotProvided)
                pairEncoding = TruncateEncoding(pairEncoding, pairEncoding.Length - toRemove, stride, truncation.Direction);
                break;
            case TruncationStrategy.LongestFirst:
                if (pairEncoding is null)
                {
                    // Single sequence: truncate to fit
                    encoding = TruncateEncoding(encoding, maxLen, stride, truncation.Direction);
                }
                else
                {
                    // Rust n1/n2 algorithm: ensure n1 is the shorter sequence
                    int n1 = encoding.Length;
                    int n2 = pairEncoding.Length;
                    bool swap = false;

                    if (n1 > n2)
                    {
                        swap = true;
                        (n1, n2) = (n2, n1);
                    }

                    if (n1 > maxLen)
                    {
                        // Both are longer than maxLen — special case
                        n2 = n1;
                    }
                    else
                    {
                        n2 = Math.Max(n1, maxLen - n1);
                    }

                    if (n1 + n2 > maxLen)
                    {
                        n1 = maxLen / 2;
                        n2 = n1 + maxLen % 2;
                    }

                    if (swap)
                        (n1, n2) = (n2, n1);

                    encoding = TruncateEncoding(encoding, n1, stride, truncation.Direction);
                    pairEncoding = TruncateEncoding(pairEncoding, n2, stride, truncation.Direction);
                }
                break;
        }

        return (encoding, pairEncoding);
    }

    /// <summary>
    /// 将编码截断为 <paramref name="maxLen"/> 个 token，当 <paramref name="stride"/> &gt; 0 时创建重叠的溢出编码。
    /// </summary>
    /// <param name="encoding">要截断的编码。</param>
    /// <param name="maxLen">目标最大长度（保留的 token 数量）。</param>
    /// <param name="stride">滑动窗口步长。必须 &lt; maxLen。</param>
    /// <param name="direction">从右侧（头部）还是左侧（尾部）保留 token。</param>
    private static Encoding TruncateEncoding(
        Encoding encoding,
        int maxLen,
        int stride,
        TruncationDirection direction)
    {
        int encodingLen = encoding.Length;

        // No truncation needed
        if (maxLen >= encodingLen)
            return encoding;

        // Truncate to empty: entire encoding becomes overflowing
        if (maxLen <= 0)
        {
            var empty = new Encoding([], [], [], [], [], [], [], [encoding]);
            return empty;
        }

        int offset = maxLen - stride; // step size between windows

        // 计算窗口数上限，选择 stackalloc 或 ArrayPool
        int maxWindows = (encodingLen + offset - 1) / offset; // 向上取整
        (int Start, int End)[]? rentedRanges = null;
        Span<(int Start, int End)> partsRanges = maxWindows <= 16
            ? stackalloc (int, int)[16]
            : (rentedRanges = System.Buffers.ArrayPool<(int, int)>.Shared.Rent(maxWindows));
        int rangeCount = 0;

        try
        {

        if (direction == TruncationDirection.Right)
        {
            bool end = false;
            for (int start = 0; start < encodingLen && !end; start += offset)
            {
                int stop = Math.Min(start + maxLen, encodingLen);
                if (stop == encodingLen) end = true;
                partsRanges[rangeCount++] = (start, stop);
            }
        }
        else // Left
        {
            bool end = false;
            for (int stop = encodingLen; stop > 0 && !end; stop -= offset)
            {
                int start = Math.Max(stop - maxLen, 0);
                if (start == 0) end = true;
                partsRanges[rangeCount++] = (start, stop);
            }
        }

        // First range becomes the main encoding, rest become overflowing
        var (mainStart, mainEnd) = partsRanges[0];

        if (rangeCount == 1)
        {
            // 常见路径：只有 1 个窗口，无溢出
            return SliceEncoding(encoding, mainStart, mainEnd);
        }

        var overflowing = new List<Encoding>(rangeCount - 1);
        for (int i = 1; i < rangeCount; i++)
        {
            var (s, e) = partsRanges[i];
            overflowing.Add(SliceEncoding(encoding, s, e));
        }

        var result = SliceEncoding(encoding, mainStart, mainEnd);
        result.GetOverflowing().AddRange(overflowing);
        return result;

        }
        finally
        {
            if (rentedRanges is not null)
                System.Buffers.ArrayPool<(int, int)>.Shared.Return(rentedRanges);
        }
    }

    /// <summary>
    /// 切片编码以提取 [start, end) 范围内的 token。
    /// 使用零拷贝 Slice，共享底层数组。
    /// </summary>
    private static Encoding SliceEncoding(Encoding encoding, int start, int end)
    {
        // 调整 SequenceRanges 以适应切片范围
        Dictionary<int, Range>? adjustedRanges = null;
        if (encoding.SequenceRanges.Count > 0)
        {
            adjustedRanges = new Dictionary<int, Range>(encoding.SequenceRanges.Count);
            int sliceLen = end - start;
            foreach (var (seqId, range) in encoding.SequenceRanges)
            {
                int rStart = Math.Max(range.Start.Value - start, 0);
                int rEnd = Math.Min(range.End.Value - start, sliceLen);
                if (rStart < rEnd)
                    adjustedRanges[seqId] = rStart..rEnd;
            }
        }

        // 零拷贝 Slice：共享底层数组，只记录范围
        var slice = encoding.Slice(start, end - start);
        if (adjustedRanges is not null)
        {
            slice.SequenceRanges.Clear();
            foreach (var kvp in adjustedRanges)
                slice.SequenceRanges[kvp.Key] = kvp.Value;
        }
        return slice;
    }
}
