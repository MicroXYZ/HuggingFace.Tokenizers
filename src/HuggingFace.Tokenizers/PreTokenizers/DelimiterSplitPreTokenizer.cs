using HuggingFace.Tokenizers.Internal;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.PreTokenizers;

/// <summary>
/// 按指定分隔符字符拆分文本的预分词器。
/// </summary>
[TokenizerComponent("CharDelimiterSplit")]
public sealed class DelimiterSplitPreTokenizer : IPreTokenizer
{
    /// <summary>
    /// 用于拆分的分隔符字符。
    /// </summary>
    public char Delimiter { get; }

    /// <summary>
    /// 初始化新的 <see cref="DelimiterSplitPreTokenizer"/>.
    /// </summary>
    /// <param name="delimiter">The character to split on.</param>
    public DelimiterSplitPreTokenizer(char delimiter)
    {
        Delimiter = delimiter;
    }

    /// <inheritdoc />
    public void PreTokenize(PreTokenizedString pretokenized)
    {
        var pattern = new CharPattern(Delimiter);
        pretokenized.Split((_, normalized) =>
        {
            var matches = pattern.FindMatches(normalized.GetSpan());
            return matches
                .Where(m => !m.IsMatch) // Keep non-delimiter parts
                .Select(m => normalized.Slice(m.Start, m.End - m.Start));
        });
    }
}
