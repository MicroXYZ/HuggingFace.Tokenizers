using HuggingFace.Tokenizers.Internal;
using System.Text.RegularExpressions;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.PreTokenizers;

/// <summary>
/// BERT 风格预分词器，按空白和标点边界拆分文本。
/// 使用正则模式 <c>\w+|[^\w\s]+</c> 将词与标点分离。
/// </summary>
[TokenizerComponent("BertPreTokenizer")]
public sealed partial class BertPreTokenizer : IPreTokenizer
{
    /// <summary>
    /// 单例实例，方便使用。
    /// </summary>
    public static readonly BertPreTokenizer Instance = new();

#if NET7_0_OR_GREATER
    [GeneratedRegex(@"\w+|[^\w\s]+")]
    private static partial Regex SplitRegex();
#else
    private static Regex SplitRegex() => s_regex;
    private static readonly Regex s_regex = new(@"\w+|[^\w\s]+", RegexOptions.Compiled);
#endif

    /// <inheritdoc />
    public void PreTokenize(PreTokenizedString pretokenized)
    {
        var pattern = new RegexPattern(SplitRegex());
        pretokenized.Split((_, normalized) =>
        {
            var matches = pattern.FindMatches(normalized.GetSpan());
            return matches
                .Where(m => m.IsMatch)
                .Select(m => normalized.Slice(m.Start, m.End - m.Start));
        });
    }
}
