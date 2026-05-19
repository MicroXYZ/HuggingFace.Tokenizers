using HuggingFace.Tokenizers.Internal;
using System.Text.RegularExpressions;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.PreTokenizers;

/// <summary>
/// 将文本拆分以分离数字与非数字的预分词器。
/// 当 <see cref="IndividualDigits"/> 为 true 时，每个数字成为独立的拆分。
/// </summary>
[TokenizerComponent("Digits")]
public sealed partial class DigitsPreTokenizer : IPreTokenizer
{
    /// <summary>
    /// 是否拆分单个数字。
    /// </summary>
    public bool IndividualDigits { get; }

    /// <summary>
    /// 初始化新的 <see cref="DigitsPreTokenizer"/>.
    /// </summary>
    /// <param name="individualDigits">
    /// 如果 <c>true</c>, 每个数字被隔离为独立的拆分。
    /// 如果 <c>false</c>, 连续数字保持在一起。
    /// </param>
    public DigitsPreTokenizer(bool individualDigits = false)
    {
        IndividualDigits = individualDigits;
    }

#if NET7_0_OR_GREATER
    [GeneratedRegex(@"\d|[^\d]+")]
    private static partial Regex IndividualDigitRegex();

    [GeneratedRegex(@"\d+|[^\d]+")]
    private static partial Regex GroupedDigitRegex();
#else
    private static Regex IndividualDigitRegex() => s_individualRegex;
    private static readonly Regex s_individualRegex = new(@"\d|[^\d]+", RegexOptions.Compiled);

    private static Regex GroupedDigitRegex() => s_groupedRegex;
    private static readonly Regex s_groupedRegex = new(@"\d+|[^\d]+", RegexOptions.Compiled);
#endif

    /// <inheritdoc />
    public void PreTokenize(PreTokenizedString pretokenized)
    {
        var regex = IndividualDigits ? IndividualDigitRegex() : GroupedDigitRegex();
        var pattern = new RegexPattern(regex);

        pretokenized.Split((_, normalized) =>
        {
            var matches = pattern.FindMatches(normalized.GetSpan());
            return matches
                .Where(m => m.IsMatch)
                .Select(m => normalized.Slice(m.Start, m.End - m.Start));
        });
    }
}
