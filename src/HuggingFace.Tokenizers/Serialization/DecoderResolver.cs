using System.Text.Json;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Decoders;
using HuggingFace.Tokenizers.Internal;
using static HuggingFace.Tokenizers.Serialization.JsonElementHelper;

namespace HuggingFace.Tokenizers.Serialization;

/// <summary>
/// 解析 <see cref="DecoderJsonModel"/> 反序列化为 concrete <see cref="IDecoder"/> 实例。
/// </summary>
public static class DecoderResolver
{
    /// <summary>
    /// 根据 JSON 模型解析具体的解码器实例。
    /// </summary>
    public static IDecoder Resolve(DecoderJsonModel model)
    {
        ArgumentNullException.ThrowIfNull(model);

        return model.Type switch
        {
            "BPEDecoder" => ResolveBpeDecoder(model),
            "ByteLevel" => new ByteLevelDecoder(),
            "WordPiece" => ResolveWordPieceDecoder(model),
            "Metaspace" => ResolveMetaspaceDecoder(model),
            "CTC" => ResolveCtcDecoder(model),
            "ByteFallback" => new ByteFallbackDecoder(),
            "Fuse" => new FuseDecoder(),
            "Strip" => ResolveStripDecoder(model),
            "Replace" => ResolveReplaceDecoder(model),
            "Sequence" => ResolveSequenceDecoder(model),
            _ => throw new NotSupportedException($"Unsupported decoder type: '{model.Type}'.")
        };
    }

    private static BpeDecoder ResolveBpeDecoder(DecoderJsonModel model)
    {
        return new BpeDecoder(GetString(model.AdditionalData, "suffix", ""));
    }

    private static WordPieceDecoder ResolveWordPieceDecoder(DecoderJsonModel model)
    {
        return new WordPieceDecoder(
            GetString(model.AdditionalData, "prefix", "##"),
            GetBool(model.AdditionalData, "cleanup", true));
    }

    private static MetaspaceDecoder ResolveMetaspaceDecoder(DecoderJsonModel model)
    {
        var data = model.AdditionalData;
        var prependSchemeStr = GetString(data, "prepend_scheme");
        var addPrefixSpace = GetOptionalBool(data, "add_prefix_space");
        var prependScheme = PrependSchemeHelper.ResolveWithCompatibility(prependSchemeStr, addPrefixSpace);

        var split = GetBool(data, "split", true);
        return new MetaspaceDecoder(GetChar(data, "replacement", '\u2581'), prependScheme, split);
    }

    private static CtcDecoder ResolveCtcDecoder(DecoderJsonModel model)
    {
        var data = model.AdditionalData;
        // Rust CTC 字段名：pad_token (String), word_delimiter_token (String), cleanup (bool)
        var padToken = GetString(data, "pad_token", "<pad>");
        var wordDelimiter = GetString(data, "word_delimiter_token", "|");
        var cleanup = GetBool(data, "cleanup", true);
        return new CtcDecoder(padToken, wordDelimiter, cleanup);
    }

    private static StripDecoder ResolveStripDecoder(DecoderJsonModel model)
    {
        var data = model.AdditionalData;
        return new StripDecoder(GetChar(data, "content", ' '), GetInt(data, "start", 1), GetInt(data, "stop", 1));
    }

    private static ReplaceDecoder ResolveReplaceDecoder(DecoderJsonModel model)
    {
        var data = model.AdditionalData;
        var pattern = GetPatternString(data, "pattern")
            ?? throw new ArgumentException("ReplaceDecoder requires a 'pattern'.");
        var isRegex = IsRegexPattern(data, "pattern");
        // Rust Replace 结构体字段名为 "content"，兼容旧的 "replacement" 写法
        var replacement = GetString(data, "content") ?? GetString(data, "replacement")
            ?? throw new ArgumentException("ReplaceDecoder requires a 'content' string.");
        return new ReplaceDecoder(pattern, replacement, isRegex ? ReplacePatternType.Regex : ReplacePatternType.String);
    }

    private static SequenceDecoder ResolveSequenceDecoder(DecoderJsonModel model)
    {
        var decodersArray = GetArray(model.AdditionalData, "decoders")
            ?? throw new ArgumentException("SequenceDecoder requires a 'decoders' array.");
        var decoders = new List<IDecoder>(decodersArray.Length);
        foreach (var element in decodersArray)
        {
            var childModel = element.Deserialize(TokenizerJsonContext.Default.DecoderJsonModel)
                ?? throw new ArgumentException("Failed to deserialize child decoder.");
            decoders.Add(Resolve(childModel));
        }
        return new SequenceDecoder(decoders);
    }
}
