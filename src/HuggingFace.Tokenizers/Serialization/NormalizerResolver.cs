using System.Collections.Concurrent;
using System.Text.Json;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Normalizers;
using static HuggingFace.Tokenizers.Serialization.JsonElementHelper;

namespace HuggingFace.Tokenizers.Serialization;

/// <summary>
/// 解析 <see cref="NormalizerJsonModel"/> 反序列化为 concrete <see cref="INormalizer"/> 实例。
/// </summary>
public static class NormalizerResolver
{
    // 运行时 Regex 模式缓存，避免重复构造
    private static readonly ConcurrentDictionary<string, System.Text.RegularExpressions.Regex> RegexCache = new();

    private static System.Text.RegularExpressions.Regex CreateRegex(string pattern)
        => HuggingFace.Tokenizers.Internal.RegexHelper.CreateRegex(pattern);

    /// <summary>
    /// 根据 JSON 模型解析具体的标准化器实例。
    /// </summary>
    public static INormalizer Resolve(NormalizerJsonModel model)
    {
        ArgumentNullException.ThrowIfNull(model);

        return model.Type switch
        {
            // 支持 Rust 结构体名（Legacy 路径）和 EnumType 名（Tagged 路径）
            "BertNormalizer" or "Bert" => ResolveBertNormalizer(model),
            "NFC" => new NfcNormalizer(),
            "NFD" => new NfdNormalizer(),
            "NFKC" => new NfkcNormalizer(),
            "NFKD" => new NfkdNormalizer(),
            "Nmt" => new NmtNormalizer(),
            "Lowercase" => new LowercaseNormalizer(),
            "Strip" => ResolveStripNormalizer(model),
            "StripAccents" => new StripAccentsNormalizer(),
            "Prepend" => ResolvePrependNormalizer(model),
            "Replace" => ResolveReplaceNormalizer(model),
            "ByteLevel" => new ByteLevelNormalizer(),
            "Sequence" => ResolveSequenceNormalizer(model),
            "Precompiled" => ResolvePrecompiledNormalizer(model),
            _ => throw new NotSupportedException($"Unsupported normalizer type: '{model.Type}'.")
        };
    }

    private static BertNormalizer ResolveBertNormalizer(NormalizerJsonModel model)
    {
        var data = model.AdditionalData;
        return new BertNormalizer(
            cleanText: GetBool(data, "clean_text", true),
            handleChineseChars: GetBool(data, "handle_chinese_chars", true),
            stripAccents: GetOptionalBool(data, "strip_accents"),
            lowercase: GetBool(data, "lowercase", true));
    }

    private static StripNormalizer ResolveStripNormalizer(NormalizerJsonModel model)
    {
        var data = model.AdditionalData;
        return new StripNormalizer(
            stripLeft: GetBool(data, "strip_left", true),
            stripRight: GetBool(data, "strip_right", true));
    }

    private static PrependNormalizer ResolvePrependNormalizer(NormalizerJsonModel model)
    {
        var prepend = GetString(model.AdditionalData, "prepend")
            ?? throw new ArgumentException("PrependNormalizer requires a 'prepend' string.");
        return new PrependNormalizer(prepend);
    }

    private static ReplaceNormalizer ResolveReplaceNormalizer(NormalizerJsonModel model)
    {
        var data = model.AdditionalData;
        var patternStr = GetPatternString(data, "pattern")
            ?? throw new ArgumentException("ReplaceNormalizer requires a 'pattern'.");
        var isRegex = IsRegexPattern(data, "pattern");
        var replacement = GetString(data, "content")
            ?? throw new ArgumentException("ReplaceNormalizer requires a 'content' string.");

        if (isRegex)
        {
            var regex = RegexCache.GetOrAdd(patternStr, CreateRegex);
            return new ReplaceNormalizer(regex, replacement);
        }
        return new ReplaceNormalizer(patternStr, replacement);
    }

    private static SequenceNormalizer ResolveSequenceNormalizer(NormalizerJsonModel model)
    {
        var normalizersArray = GetArray(model.AdditionalData, "normalizers")
            ?? throw new ArgumentException("SequenceNormalizer requires a 'normalizers' array.");
        var normalizers = new List<INormalizer>(normalizersArray.Length);
        foreach (var element in normalizersArray)
        {
            var childModel = element.Deserialize(TokenizerJsonContext.Default.NormalizerJsonModel)
                ?? throw new ArgumentException("Failed to deserialize child normalizer.");
            normalizers.Add(Resolve(childModel));
        }
        return new SequenceNormalizer(normalizers);
    }

    private static PrecompiledNormalizer ResolvePrecompiledNormalizer(NormalizerJsonModel model)
    {
        var charsmap = GetString(model.AdditionalData, "charsmap")
            ?? throw new ArgumentException("PrecompiledNormalizer requires a 'charsmap' string.");
        return new PrecompiledNormalizer(Convert.FromBase64String(charsmap));
    }
}
