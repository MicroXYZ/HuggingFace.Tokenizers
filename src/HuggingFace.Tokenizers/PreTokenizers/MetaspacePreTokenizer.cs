using System.Text;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.PreTokenizers;

/// <summary>
/// SentencePiece 风格预分词器，将空格替换为特殊替换字符（默认 ▁, U+2581）。
/// </summary>
[TokenizerComponent("Metaspace")]
public sealed class MetaspacePreTokenizer : IPreTokenizer
{
    /// <summary>默认选项的单例实例。</summary>
    public static readonly MetaspacePreTokenizer Instance = new();

    private readonly string _replacement;
    private readonly char _replacementChar;
    private readonly bool _addPrefixSpace;
    private readonly PrependScheme _prependScheme;
    private readonly bool _split;

    /// <summary>替换字符。</summary>
    [JsonKey("replacement")]
    public char ReplacementChar => _replacementChar;

    /// <summary>是否添加前导替换字符。</summary>
    public bool AddPrefixSpace => _addPrefixSpace;

    /// <summary>前缀方案。</summary>
    [JsonKey("prepend_scheme")]
    public PrependScheme PrependSchemeValue => _prependScheme;

    /// <summary>
    /// 初始化新的 <see cref="MetaspacePreTokenizer"/>。
    /// </summary>
    /// <param name="replacement">替换字符，默认 ▁ (U+2581)。</param>
    /// <param name="addPrefixSpace">是否添加前导替换字符。</param>
    /// <param name="prependScheme">前缀方案。</param>
    /// <param name="split">是否在替换字符处拆分。与 Rust split 参数一致，默认 true。</param>
    public MetaspacePreTokenizer(
        char replacement = '\u2581',
        bool addPrefixSpace = true,
        PrependScheme prependScheme = PrependScheme.Always,
        bool split = true)
    {
        _replacement = replacement.ToString();
        _replacementChar = replacement;
        _addPrefixSpace = addPrefixSpace;
        _prependScheme = prependScheme;
        _split = split;
    }

    /// <inheritdoc />
    public void PreTokenize(PreTokenizedString pretokenized)
    {
        pretokenized.Split((index, normalized) =>
        {
            // 第一步：将空格替换为 replacement 字符
            normalized.Replace(" ", _replacement);

            // 第二步：根据 prepend scheme 决定是否添加前缀
            // 与 Rust 一致：使用原始偏移判断是否为文本起始位置
            bool shouldPrepend = _prependScheme switch
            {
                PrependScheme.Always => true,
                PrependScheme.First => normalized.OffsetsOriginal().Start == 0 && _addPrefixSpace,
                PrependScheme.Never => false,
                _ => false
            };

            if (shouldPrepend && (normalized.Length == 0 || normalized.GetSpan()[0] != _replacementChar))
                normalized.Prepend(_replacement);

            // split=false 时仅替换空格，不拆分（与 Rust 一致）
            if (!_split)
                return new[] { normalized };

            var text = normalized.GetSpan();
            if (text.IsEmpty)
                return Enumerable.Empty<NormalizedString>();

            // 使用 NormalizedString.Split 的 MergedWithNext 行为
            var splitResults = normalized.Split(
                SplitDelimiterBehavior.MergedWithNext,
                (ReadOnlySpan<char> str) => FindCharMatches(str, _replacementChar));

            return splitResults.Select(r => r.Part);
        });
    }

    /// <summary>
    /// 查找字符串中所有匹配字符的位置。
    /// </summary>
    private static IReadOnlyList<(int Start, int End, bool IsMatch)> FindCharMatches(ReadOnlySpan<char> text, char target)
    {
        var matches = new List<(int Start, int End, bool IsMatch)>();
        int lastEnd = 0;

        for (int i = 0; i < text.Length; i++)
        {
            if (text[i] == target)
            {
                if (i > lastEnd)
                    matches.Add((lastEnd, i, false));
                matches.Add((i, i + 1, true));
                lastEnd = i + 1;
            }
        }

        if (lastEnd < text.Length)
            matches.Add((lastEnd, text.Length, false));

        return matches;
    }
}
