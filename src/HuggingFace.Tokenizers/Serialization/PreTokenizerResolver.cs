using System.Collections.Concurrent;
using HuggingFace.Tokenizers.Internal;
using System.Text.Json;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.PreTokenizers;
using static HuggingFace.Tokenizers.Serialization.JsonElementHelper;

namespace HuggingFace.Tokenizers.Serialization;

/// <summary>
/// 解析 <see cref="PreTokenizerJsonModel"/> 反序列化为 concrete <see cref="IPreTokenizer"/> 实例。
/// </summary>
public static class PreTokenizerResolver
{
    // 运行时 Regex 模式缓存，避免重复构造
    private static readonly ConcurrentDictionary<string, System.Text.RegularExpressions.Regex> RegexCache = new();

    private static System.Text.RegularExpressions.Regex CreateRegex(string pattern)
        => HuggingFace.Tokenizers.Internal.RegexHelper.CreateRegex(pattern);

    /// <summary>
    /// 根据 JSON 模型解析具体的预分词器实例。
    /// </summary>
    public static IPreTokenizer Resolve(PreTokenizerJsonModel model)
    {
        ArgumentNullException.ThrowIfNull(model);

        return model.Type switch
        {
            "BertPreTokenizer" => new BertPreTokenizer(),
            "ByteLevel" => ResolveByteLevelPreTokenizer(model),
            "Whitespace" => new WhitespacePreTokenizer(),
            "WhitespaceSplit" => new WhitespaceSplitPreTokenizer(),
            "Metaspace" => ResolveMetaspacePreTokenizer(model),
            "Digits" => ResolveDigitsPreTokenizer(model),
            "Punctuation" => ResolvePunctuationPreTokenizer(model),
            "Delimiter" or "CharDelimiterSplit" => ResolveDelimiterSplitPreTokenizer(model),
            "Sequence" => ResolveSequencePreTokenizer(model),
            "Split" => ResolveSplitPreTokenizer(model),
            "UnicodeScripts" => new UnicodeScriptsPreTokenizer(),
            "FixedLength" => ResolveFixedLengthPreTokenizer(model),
            _ => throw new NotSupportedException($"Unsupported pre-tokenizer type: '{model.Type}'.")
        };
    }

    private static ByteLevelPreTokenizer ResolveByteLevelPreTokenizer(PreTokenizerJsonModel model)
    {
        var data = model.AdditionalData;
        return new ByteLevelPreTokenizer(
            addPrefixSpace: GetBool(data, "add_prefix_space", true),
            useRegex: GetBool(data, "use_regex", true),
            trimOffsets: GetBool(data, "trim_offsets", true));
    }

    private static MetaspacePreTokenizer ResolveMetaspacePreTokenizer(PreTokenizerJsonModel model)
    {
        var data = model.AdditionalData;
        char replacement = '\u2581';
        if (GetString(data, "replacement") is { Length: > 0 } repStr)
            replacement = repStr[0];

        var prependSchemeStr = GetString(data, "prepend_scheme");
        var addPrefixSpace = GetOptionalBool(data, "add_prefix_space");
        var prependScheme = PrependSchemeHelper.ResolveWithCompatibility(prependSchemeStr, addPrefixSpace);

        var split = GetBool(data, "split", true);

        return new MetaspacePreTokenizer(replacement, addPrefixSpace ?? true, prependScheme, split);
    }

    private static DigitsPreTokenizer ResolveDigitsPreTokenizer(PreTokenizerJsonModel model)
    {
        return new DigitsPreTokenizer(GetBool(model.AdditionalData, "individual_digits", false));
    }

    private static PunctuationPreTokenizer ResolvePunctuationPreTokenizer(PreTokenizerJsonModel model)
    {
        return new PunctuationPreTokenizer(
            ParseSplitDelimiterBehavior(GetString(model.AdditionalData, "behavior") ?? "isolated"));
    }

    private static DelimiterSplitPreTokenizer ResolveDelimiterSplitPreTokenizer(PreTokenizerJsonModel model)
    {
        var delimiterStr = GetString(model.AdditionalData, "delimiter")
            ?? throw new ArgumentException("DelimiterSplitPreTokenizer requires a 'delimiter' string.");
        if (delimiterStr.Length == 0)
            throw new ArgumentException("DelimiterSplitPreTokenizer 'delimiter' must not be empty.");
        return new DelimiterSplitPreTokenizer(delimiterStr[0]);
    }

    private static SequencePreTokenizer ResolveSequencePreTokenizer(PreTokenizerJsonModel model)
    {
        var pretokenizersArray = GetArray(model.AdditionalData, "pretokenizers")
            ?? throw new ArgumentException("SequencePreTokenizer requires a 'pretokenizers' array.");
        var pretokenizers = new List<IPreTokenizer>(pretokenizersArray.Length);
        foreach (var element in pretokenizersArray)
        {
            var childModel = element.Deserialize(TokenizerJsonContext.Default.PreTokenizerJsonModel)
                ?? throw new ArgumentException("Failed to deserialize child pre-tokenizer.");
            pretokenizers.Add(Resolve(childModel));
        }
        return new SequencePreTokenizer(pretokenizers);
    }

    private static SplitPreTokenizer ResolveSplitPreTokenizer(PreTokenizerJsonModel model)
    {
        var data = model.AdditionalData;
        var patternStr = GetPatternString(data, "pattern")
            ?? throw new ArgumentException("SplitPreTokenizer requires a 'pattern'.");
        var behavior = ParseSplitDelimiterBehavior(GetString(data, "behavior") ?? "removed");
        var invert = GetBool(data, "invert", false);

        IPattern pattern = IsRegexPattern(data, "pattern")
            ? new RegexPattern(RegexCache.GetOrAdd(patternStr, CreateRegex))
            : new StringPattern(patternStr);

        return new SplitPreTokenizer(pattern, behavior, invert);
    }

    private static FixedLengthPreTokenizer ResolveFixedLengthPreTokenizer(PreTokenizerJsonModel model)
    {
        var length = GetInt(model.AdditionalData, "length")
            ?? throw new ArgumentException("FixedLengthPreTokenizer requires a 'length' integer.");
        return new FixedLengthPreTokenizer(length);
    }

    private static SplitDelimiterBehavior ParseSplitDelimiterBehavior(string value) => value switch
    {
        "removed" or "Removed" => SplitDelimiterBehavior.Removed,
        "isolated" or "Isolated" => SplitDelimiterBehavior.Isolated,
        "mergedwithprevious" or "merged_with_previous" or "MergedWithPrevious" => SplitDelimiterBehavior.MergedWithPrevious,
        "mergedwithnext" or "merged_with_next" or "MergedWithNext" => SplitDelimiterBehavior.MergedWithNext,
        "contiguous" or "Contiguous" => SplitDelimiterBehavior.Contiguous,
        _ => throw new ArgumentException($"Unsupported split delimiter behavior: '{value}'.")
    };
}
