using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Padding;

/// <summary>
/// 为 <see cref="Encoding"/> 集合提供填充工具方法。
/// </summary>
public static class PaddingHelper
{
    /// <summary>
    /// 按照指定的填充参数对编码列表进行填充。
    /// </summary>
    public static void PadEncodings(IList<Encoding> encodings, PaddingParams paddingParams)
    {
        if (encodings == null || encodings.Count == 0) return;

        int targetLength = ComputeTargetLength(encodings, paddingParams);

        foreach (var encoding in encodings)
        {
            encoding.Pad(
                targetLength,
                paddingParams.PadId,
                paddingParams.PadTypeId,
                paddingParams.PadToken,
                paddingParams.Direction);
        }
    }

    private static int ComputeTargetLength(IList<Encoding> encodings, PaddingParams paddingParams)
    {
        int maxLength = 0;
        foreach (var encoding in encodings)
        {
            if (encoding.Length > maxLength)
                maxLength = encoding.Length;
        }

        int targetLength = paddingParams.Strategy switch
        {
            PaddingStrategy.Fixed => paddingParams.MaxLength,
            _ => maxLength // BatchLongest
        };

        // Ensure target is at least as long as the longest encoding
        if (targetLength < maxLength)
            targetLength = maxLength;

        // Apply pad_to_multiple_of
        if (paddingParams.PadToMultipleOf is int multiple && multiple > 0)
        {
            int remainder = targetLength % multiple;
            if (remainder != 0)
                targetLength += multiple - remainder;
        }

        return targetLength;
    }
}
