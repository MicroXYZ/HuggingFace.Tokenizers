using System.Text.Json;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Processors;
using static HuggingFace.Tokenizers.Serialization.JsonElementHelper;

namespace HuggingFace.Tokenizers.Serialization;

/// <summary>
/// 解析 <see cref="PostProcessorJsonModel"/> 反序列化为 concrete <see cref="IPostProcessor"/> 实例。
/// 将 JSON "type" 标识符映射到对应的处理器类，
/// 并从 <see cref="PostProcessorJsonModel.AdditionalData"/> 读取属性。
/// </summary>
public static class PostProcessorResolver
{
    /// <summary>
    /// 构建 <see cref="IPostProcessor"/> 从给定的 JSON 模型。
    /// </summary>
    public static IPostProcessor Resolve(PostProcessorJsonModel model)
    {
        ArgumentNullException.ThrowIfNull(model);

        return model.Type switch
        {
            "BertProcessing" => ResolveBertProcessing(model),
            "RobertaProcessing" => ResolveRobertaProcessing(model),
            "TemplateProcessing" => ResolveTemplateProcessing(model),
            "ByteLevel" => new ByteLevelPostProcessor(),
            "Sequence" => ResolveSequenceProcessor(model),
            _ => throw new NotSupportedException($"Unsupported post-processor type: '{model.Type}'.")
        };
    }

    private static BertProcessing ResolveBertProcessing(PostProcessorJsonModel model)
    {
        var data = model.AdditionalData;
        return new BertProcessing(ReadTokenPair(data, "sep"), ReadTokenPair(data, "cls"));
    }

    private static RobertaProcessing ResolveRobertaProcessing(PostProcessorJsonModel model)
    {
        var data = model.AdditionalData;
        return new RobertaProcessing(
            ReadTokenPair(data, "sep"),
            ReadTokenPair(data, "cls"),
            GetBool(data, "trim_offsets", true),
            GetBool(data, "add_prefix_space", true));
    }

    private static TemplateProcessing ResolveTemplateProcessing(PostProcessorJsonModel model)
    {
        var data = model.AdditionalData
            ?? throw new ArgumentException("TemplateProcessing requires AdditionalData.");

        // 读取 special_tokens 字典（与 Rust Tokens 对应）：
        // { "<bos>": { "id": "<bos>", "ids": [1], "tokens": ["<bos>"] }, ... }
        var specialTokens = ReadSpecialTokens(data);

        var single = GetArray(data, "single")
            ?? throw new ArgumentException("TemplateProcessing requires a 'single' template array.");
        var pair = GetArray(data, "pair");

        return new TemplateProcessing(
            ParseTemplatePieces(single, specialTokens),
            pair is not null ? ParseTemplatePieces(pair, specialTokens) : null);
    }

    /// <summary>
    /// 解析 special_tokens 字典，key 为字符串 id（如 "&lt;bos&gt;"），
    /// value 包含 ids（数字数组）和 tokens（文本数组）。
    /// 与 Rust TemplateProcessing.special_tokens 字段对应。
    /// </summary>
    private static Dictionary<string, (uint[] Ids, string[] Tokens)> ReadSpecialTokens(
        Dictionary<string, JsonElement>? data)
    {
        var result = new Dictionary<string, (uint[] Ids, string[] Tokens)>(StringComparer.Ordinal);
        if (data is null || !data.TryGetValue("special_tokens", out var element)
            || element.ValueKind != JsonValueKind.Object)
            return result;

        foreach (var prop in element.EnumerateObject())
        {
            // 每个 special token 的 JSON 结构：
            // { "id": "<bos>", "ids": [1], "tokens": ["<bos>"] }
            if (prop.Value.ValueKind != JsonValueKind.Object) continue;

            if (!prop.Value.TryGetProperty("ids", out var idsElement)
                || idsElement.ValueKind != JsonValueKind.Array)
                continue;

            var ids = idsElement.EnumerateArray().Select(e => GetUInt32Loose(e)).ToArray();

            string[] tokens;
            if (prop.Value.TryGetProperty("tokens", out var tokensElement)
                && tokensElement.ValueKind == JsonValueKind.Array)
            {
                tokens = tokensElement.EnumerateArray()
                    .Select(e => e.GetString() ?? "")
                    .ToArray();
            }
            else
            {
                // tokens 缺失时，用 id 本身作为文本
                tokens = [prop.Name];
            }

            if (ids.Length != tokens.Length)
                throw new ArgumentException(
                    $"SpecialToken '{prop.Name}': ids ({ids.Length}) and tokens ({tokens.Length}) length mismatch.");

            result[prop.Name] = (ids, tokens);
        }
        return result;
    }

    /// <summary>
    /// 解析模板片段列表。
    /// Rust Piece::SpecialToken { id: String, type_id: u32 } 中 id 是字符串引用，
    /// 通过 special_tokens 字典查找实际的数字 ID 和文本。
    /// </summary>
    private static IReadOnlyList<TemplatePiece> ParseTemplatePieces(
        JsonElement[] pieces,
        Dictionary<string, (uint[] Ids, string[] Tokens)> specialTokens)
    {
        var result = new List<TemplatePiece>(pieces.Length);
        foreach (var piece in pieces)
        {
            if (piece.ValueKind != JsonValueKind.Object)
                throw new ArgumentException("Each template piece must be a JSON object.");

            if (piece.TryGetProperty("SpecialToken", out var specialToken))
            {
                // Rust Piece::SpecialToken { id: String, type_id: u32 } — id 是字符串引用
                // 但部分 tokenizer.json 中 id 也可能是数字，需兼容两种格式
                var idElement = specialToken.GetProperty("id");
                var id = idElement.ValueKind == JsonValueKind.String
                    ? idElement.GetString()!
                    : GetUInt32Loose(idElement).ToString();
                var typeId = GetUInt32Loose(specialToken.GetProperty("type_id"));

                // 通过字符串 id 查找 special_tokens 字典获取实际的 ids 和 tokens
                if (specialTokens.TryGetValue(id, out var tokenInfo))
                {
                    result.Add(Template.Special(tokenInfo.Ids, tokenInfo.Tokens, typeId));
                }
                else
                {
                    // 字典中找不到时，尝试将 id 直接解析为数字（向后兼容纯数字 id 的情况）
                    if (uint.TryParse(id, out var numericId))
                        result.Add(Template.Special(numericId, typeId));
                    else
                        throw new ArgumentException(
                            $"SpecialToken id '{id}' not found in special_tokens dictionary and is not a numeric id.");
                }
            }
            else if (piece.TryGetProperty("Sequence", out var sequence))
            {
                var id = sequence.GetProperty("id").GetString()
                    ?? throw new ArgumentException("Sequence piece 'id' cannot be null.");
                var typeId = GetUInt32Loose(sequence.GetProperty("type_id"));
                result.Add(id switch
                {
                    "A" => Template.A(typeId),
                    "B" => Template.B(typeId),
                    _ => throw new ArgumentException($"Unknown sequence id: '{id}'.")
                });
            }
            else
            {
                throw new ArgumentException("Template piece must contain 'SpecialToken' or 'Sequence' key.");
            }
        }
        return result;
    }

    private static SequenceProcessor ResolveSequenceProcessor(PostProcessorJsonModel model)
    {
        var processorsArray = GetArray(model.AdditionalData, "processors")
            ?? throw new ArgumentException("SequenceProcessor requires a 'processors' array.");
        var processors = new List<IPostProcessor>(processorsArray.Length);
        foreach (var element in processorsArray)
        {
            var childModel = element.Deserialize(TokenizerJsonContext.Default.PostProcessorJsonModel)
                ?? throw new ArgumentException("Failed to deserialize child post-processor.");
            processors.Add(Resolve(childModel));
        }
        return new SequenceProcessor(processors);
    }
}
