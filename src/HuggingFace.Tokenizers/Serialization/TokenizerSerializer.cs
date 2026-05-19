using System.Text.Json;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Models.Unigram;

namespace HuggingFace.Tokenizers.Serialization;

/// <summary>
/// AOT 兼容的 tokenizer.json 格式序列化器。
/// 使用 Utf8JsonWriter/JsonDocument 代替基于反射的序列化。
/// </summary>
public static class TokenizerSerializer
{

    /// <summary>
    /// 将分词器序列化为 JSON 字符串 (tokenizer.json format).
    /// </summary>
    /// <param name="tokenizer">要序列化的分词器。</param>
    /// <param name="pretty">是否缩进 JSON 输出。</param>
    public static string Serialize(Tokenizer tokenizer, bool pretty = false)
    {
        using var stream = new MemoryStream();
        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions
        {
            Indented = pretty,
            Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
        });

        writer.WriteStartObject();

        // version
        writer.WriteString("version", "1.0");

        // normalizer
        WriteNormalizer(writer, tokenizer.Normalizer);

        // pre_tokenizer
        WritePreTokenizer(writer, tokenizer.PreTokenizer);

        // model
        WriteModel(writer, tokenizer.Model);

        // added_tokens
        WriteAddedTokens(writer, tokenizer.AddedVocabulary);

        // post_processor
        WritePostProcessor(writer, tokenizer.PostProcessor);

        // decoder
        WriteDecoder(writer, tokenizer.Decoder);

        // truncation
        WriteTruncation(writer, tokenizer.Truncation);

        // padding
        WritePadding(writer, tokenizer.Padding);

        writer.WriteEndObject();
        writer.Flush();

        return global::System.Text.Encoding.UTF8.GetString(stream.ToArray());
    }

    /// <summary>
    /// 将 JSON 字符串反序列化为 Tokenizer。
    /// 注意：这仅创建基本结构。完整的模型加载需要 vocab/merge 文件。
    /// </summary>
    public static TokenizerJsonModel DeserializeToModel(string json)
    {
        return JsonSerializer.Deserialize(json, TokenizerJsonContext.Default.TokenizerJsonModel)
            ?? throw new JsonException("Failed to deserialize tokenizer JSON.");
    }

    private static void WriteNormalizer(Utf8JsonWriter writer, INormalizer? normalizer)
    {
        if (normalizer is null) return;

        writer.WritePropertyName("normalizer");
        writer.WriteStartObject();

        var typeName = global::HuggingFace.Tokenizers.Generated.TokenizerComponentFactory.GetNormalizerTypeName(normalizer);
        writer.WriteString("type", typeName);

        WriteNormalizerProperties(writer, normalizer);

        writer.WriteEndObject();
    }

    private static void WritePreTokenizer(Utf8JsonWriter writer, IPreTokenizer? preTokenizer)
    {
        if (preTokenizer is null) return;

        writer.WritePropertyName("pre_tokenizer");
        writer.WriteStartObject();

        var typeName = global::HuggingFace.Tokenizers.Generated.TokenizerComponentFactory.GetPreTokenizerTypeName(preTokenizer);
        writer.WriteString("type", typeName);

        WritePreTokenizerProperties(writer, preTokenizer);

        writer.WriteEndObject();
    }

    private static void WriteModel(Utf8JsonWriter writer, IModel model)
    {
        writer.WritePropertyName("model");
        writer.WriteStartObject();

        var typeName = global::HuggingFace.Tokenizers.Generated.TokenizerComponentFactory.GetModelTypeName(model);
        writer.WriteString("type", typeName);

        WriteModelProperties(writer, model);

        writer.WriteEndObject();
    }

    private static void WritePostProcessor(Utf8JsonWriter writer, IPostProcessor? processor)
    {
        if (processor is null) return;

        writer.WritePropertyName("post_processor");
        writer.WriteStartObject();

        var typeName = global::HuggingFace.Tokenizers.Generated.TokenizerComponentFactory.GetPostProcessorTypeName(processor);
        writer.WriteString("type", typeName);

        WritePostProcessorProperties(writer, processor);

        writer.WriteEndObject();
    }

    private static void WriteDecoder(Utf8JsonWriter writer, IDecoder? decoder)
    {
        if (decoder is null) return;

        writer.WritePropertyName("decoder");
        writer.WriteStartObject();

        var typeName = global::HuggingFace.Tokenizers.Generated.TokenizerComponentFactory.GetDecoderTypeName(decoder);
        writer.WriteString("type", typeName);

        WriteDecoderProperties(writer, decoder);

        writer.WriteEndObject();
    }

    private static void WriteTruncation(Utf8JsonWriter writer, TruncationParams? truncation)
    {
        if (truncation is null) return;

        writer.WritePropertyName("truncation");
        writer.WriteStartObject();
        writer.WriteString("type", truncation.Strategy.ToString());
        writer.WriteNumber("max_length", truncation.MaxLength);
        writer.WriteNumber("stride", truncation.Stride);
        writer.WriteString("direction", truncation.Direction.ToString());
        writer.WriteEndObject();
    }

    private static void WritePadding(Utf8JsonWriter writer, PaddingParams? padding)
    {
        if (padding is null) return;

        writer.WritePropertyName("padding");
        writer.WriteStartObject();

        // Rust PaddingStrategy: BatchLongest 序列化为字符串，Fixed(size) 序列化为 {"Fixed": size}
        if (padding.Strategy == PaddingStrategy.Fixed && padding.MaxLength > 0)
        {
            writer.WritePropertyName("type");
            writer.WriteStartObject();
            writer.WriteNumber("Fixed", padding.MaxLength);
            writer.WriteEndObject();
        }
        else
        {
            writer.WriteString("type", "BatchLongest");
        }

        writer.WriteNumber("pad_id", padding.PadId);
        writer.WriteNumber("pad_type_id", padding.PadTypeId);
        writer.WriteString("pad_token", padding.PadToken);
        writer.WriteString("direction", padding.Direction.ToString());

        if (padding.PadToMultipleOf is { } ptmo && ptmo > 0)
            writer.WriteNumber("pad_to_multiple_of", ptmo);

        writer.WriteEndObject();
    }

    // --- Type name resolution ---
    // Now handled by source-generated TokenizerComponentFactory.GetXxxTypeName methods (AOT-compatible).

    // --- Property writers (write type-specific properties) ---
    // 简单属性由 ComponentPropertyWriter 自动生成。
    // 复杂属性（集合、模式、元组）在下方手动处理。

    private static void WriteNormalizerProperties(Utf8JsonWriter writer, INormalizer normalizer)
    {
        // 自动生成的简单属性（bool、string、char、int、uint、enum、nullable）
        global::HuggingFace.Tokenizers.Generated.ComponentPropertyWriter.WriteNormalizerProperties(normalizer, writer);

        // 复杂属性手动处理
        switch (normalizer)
        {
            case Normalizers.SequenceNormalizer sequence:
                writer.WritePropertyName("normalizers");
                writer.WriteStartArray();
                foreach (var child in sequence.Normalizers)
                {
                    writer.WriteStartObject();
                    var childType = global::HuggingFace.Tokenizers.Generated.TokenizerComponentFactory.GetNormalizerTypeName(child);
                    writer.WriteString("type", childType);
                    WriteNormalizerProperties(writer, child);
                    writer.WriteEndObject();
                }
                writer.WriteEndArray();
                break;

            case Normalizers.ReplaceNormalizer replace:
                WritePattern(writer, "pattern", replace.PatternString, replace.IsRegexPattern);
                break;

            case Normalizers.PrecompiledNormalizer precompiled:
                var rawMap = precompiled.RawCharsMap;
                if (rawMap is not null)
                    writer.WriteString("charsmap", Convert.ToBase64String(rawMap));
                break;
        }
    }

    private static void WritePreTokenizerProperties(Utf8JsonWriter writer, IPreTokenizer preTokenizer)
    {
        // Auto-generated simple properties
        global::HuggingFace.Tokenizers.Generated.ComponentPropertyWriter.WritePreTokenizerProperties(preTokenizer, writer);

        // 复杂属性手动处理
        switch (preTokenizer)
        {
            case PreTokenizers.SequencePreTokenizer sequence:
                writer.WritePropertyName("pretokenizers");
                writer.WriteStartArray();
                foreach (var child in sequence.PreTokenizers)
                {
                    writer.WriteStartObject();
                    var childType = global::HuggingFace.Tokenizers.Generated.TokenizerComponentFactory.GetPreTokenizerTypeName(child);
                    writer.WriteString("type", childType);
                    WritePreTokenizerProperties(writer, child);
                    writer.WriteEndObject();
                }
                writer.WriteEndArray();
                break;

            case PreTokenizers.SplitPreTokenizer split:
                WritePattern(writer, "pattern", GetSplitPatternString(split), split.Pattern is Internal.RegexPattern);
                break;
        }
    }

    private static void WriteModelProperties(Utf8JsonWriter writer, IModel model)
    {
        // Unigram 模型的 vocab 需要序列化为 [[token, score], ...] 格式（与 Rust/标准 tokenizer.json 兼容）
        if (model is UnigramModel unigramModel)
        {
            var vocabList = unigramModel.GetVocabList();
            if (vocabList.Count > 0)
            {
                writer.WritePropertyName("vocab");
                writer.WriteStartArray();
                foreach (var (token, score) in vocabList)
                {
                    writer.WriteStartArray();
                    writer.WriteStringValue(token);
                    writer.WriteNumberValue(score);
                    writer.WriteEndArray();
                }
                writer.WriteEndArray();
            }

            // unk_id（与 Rust 对齐：None 时写 null，Some 时写数字）
            if (unigramModel.UnkId is uint unkId)
                writer.WriteNumber("unk_id", unkId);
            else
                writer.WriteNull("unk_id");

            // byte_fallback（与 Rust 对齐：始终写入）
            writer.WriteBoolean("byte_fallback", unigramModel.ByteFallback);
        }
        else
        {
            var vocab = model.GetVocab();
            if (vocab.Count > 0)
            {
                writer.WritePropertyName("vocab");
                writer.WriteStartObject();
                foreach (var (token, id) in vocab)
                    writer.WriteNumber(token, id);
                writer.WriteEndObject();
            }
        }

        // merges (BPE models)
        var merges = model.GetMerges().ToList();
        if (merges.Count > 0)
        {
            writer.WritePropertyName("merges");
            writer.WriteStartArray();
            foreach (var merge in merges)
                writer.WriteStringValue(merge);
            writer.WriteEndArray();
        }

        // continuing_subword_prefix (BPE / WordPiece)
        if (!string.IsNullOrEmpty(model.ContinuingSubwordPrefix))
            writer.WriteString("continuing_subword_prefix", model.ContinuingSubwordPrefix);

        // end_of_word_suffix (BPE)
        if (!string.IsNullOrEmpty(model.EndOfWordSuffix))
            writer.WriteString("end_of_word_suffix", model.EndOfWordSuffix);
    }

    private static void WriteAddedTokens(Utf8JsonWriter writer, AddedVocabulary addedVocabulary)
    {
        var tokens = addedVocabulary.GetAddedTokens();
        if (tokens.Count == 0) return;

        var decoder = addedVocabulary.GetAddedTokensDecoder();

        writer.WritePropertyName("added_tokens");
        writer.WriteStartArray();

        foreach (var (content, id) in tokens.OrderBy(kv => kv.Value))
        {
            // 从解码器查找完整的 AddedToken 元数据
            AddedToken? addedToken = decoder.TryGetValue(id, out var at) ? at : null;

            writer.WriteStartObject();
            writer.WriteNumber("id", id);
            writer.WriteString("content", content);
            writer.WriteBoolean("single_word", addedToken?.SingleWord ?? false);
            writer.WriteBoolean("lstrip", addedToken?.LStrip ?? false);
            writer.WriteBoolean("rstrip", addedToken?.RStrip ?? false);
            writer.WriteBoolean("normalized", addedToken?.Normalized ?? false);
            writer.WriteBoolean("special", addedToken?.IsSpecial ?? false);
            writer.WriteEndObject();
        }

        writer.WriteEndArray();
    }

    private static void WritePostProcessorProperties(Utf8JsonWriter writer, IPostProcessor processor)
    {
        // 自动生成的简单属性（RobertaProcessing: trim_offsets, add_prefix_space）
        global::HuggingFace.Tokenizers.Generated.ComponentPropertyWriter.WritePostProcessorProperties(processor, writer);

        // 复杂属性手动处理
        switch (processor)
        {
            case Processors.BertProcessing bert:
                WriteTokenPair(writer, "sep", bert.Sep);
                WriteTokenPair(writer, "cls", bert.Cls);
                break;

            case Processors.RobertaProcessing roberta:
                WriteTokenPair(writer, "sep", roberta.Sep);
                WriteTokenPair(writer, "cls", roberta.Cls);
                break;

            case Processors.TemplateProcessing template:
                writer.WritePropertyName("single");
                WriteTemplatePieces(writer, template.SingleTemplate);
                if (template.PairTemplate is not null)
                {
                    writer.WritePropertyName("pair");
                    WriteTemplatePieces(writer, template.PairTemplate);
                }
                break;

            case Processors.SequenceProcessor sequence:
                writer.WritePropertyName("processors");
                writer.WriteStartArray();
                foreach (var child in sequence.Processors)
                {
                    writer.WriteStartObject();
                    var childType = global::HuggingFace.Tokenizers.Generated.TokenizerComponentFactory.GetPostProcessorTypeName(child);
                    writer.WriteString("type", childType);
                    WritePostProcessorProperties(writer, child);
                    writer.WriteEndObject();
                }
                writer.WriteEndArray();
                break;
        }
    }

    private static void WriteDecoderProperties(Utf8JsonWriter writer, IDecoder decoder)
    {
        // Auto-generated simple properties
        global::HuggingFace.Tokenizers.Generated.ComponentPropertyWriter.WriteDecoderProperties(decoder, writer);

        // 复杂属性手动处理
        switch (decoder)
        {
            case Decoders.ReplaceDecoder replace:
                WritePattern(writer, "pattern", replace.Pattern, replace.PatternType == Decoders.ReplacePatternType.Regex);
                writer.WriteString("content", replace.ReplacementValue);
                break;

            case Decoders.SequenceDecoder sequence:
                writer.WritePropertyName("decoders");
                writer.WriteStartArray();
                foreach (var child in sequence.Decoders)
                {
                    writer.WriteStartObject();
                    var childType = global::HuggingFace.Tokenizers.Generated.TokenizerComponentFactory.GetDecoderTypeName(child);
                    writer.WriteString("type", childType);
                    WriteDecoderProperties(writer, child);
                    writer.WriteEndObject();
                }
                writer.WriteEndArray();
                break;
        }
    }

    // ─── Pattern serialization helper ─────────────────────────────────────────────────

    /// <summary>
    /// 以 Rust 兼容格式写入模式属性： <c>{"String":"..."}</c> 或 <c>{"Regex":"..."}</c>.
    /// </summary>
    private static void WritePattern(Utf8JsonWriter writer, string propertyName, string patternValue, bool isRegex)
    {
        writer.WritePropertyName(propertyName);
        writer.WriteStartObject();
        writer.WriteString(isRegex ? "Regex" : "String", patternValue);
        writer.WriteEndObject();
    }

    /// <summary>
    /// 将 token 对写入 JSON 数组： <c>["token", id]</c>.
    /// </summary>
    private static void WriteTokenPair(Utf8JsonWriter writer, string propertyName, (string Token, uint Id) pair)
    {
        writer.WritePropertyName(propertyName);
        writer.WriteStartArray();
        writer.WriteStringValue(pair.Token);
        writer.WriteNumberValue(pair.Id);
        writer.WriteEndArray();
    }

    /// <summary>
    /// 将模板片段写入与 Rust 格式一致的 JSON 数组。
    /// 每个片段或 <c>{"SpecialToken":{"id":N,"type_id":T}}</c> 或 <c>{"Sequence":{"id":"A"/"B","type_id":T}}</c>.
    /// </summary>
    private static void WriteTemplatePieces(Utf8JsonWriter writer, IReadOnlyList<Processors.TemplatePiece> pieces)
    {
        writer.WriteStartArray();
        foreach (var piece in pieces)
        {
            writer.WriteStartObject();
            switch (piece)
            {
                case Processors.SpecialTokenPiece special:
                    writer.WritePropertyName("SpecialToken");
                    writer.WriteStartObject();
                    // Rust Piece::SpecialToken { id: String, type_id: u32 } — id 是字符串
                    writer.WriteString("id", special.TokenIds.Count > 0
                        ? special.TokenIds[0].ToString()
                        : "0");
                    writer.WriteNumber("type_id", piece.TypeId);
                    writer.WriteEndObject();
                    break;

                case Processors.SequencePiece seq:
                    writer.WritePropertyName("Sequence");
                    writer.WriteStartObject();
                    writer.WriteString("id", seq.SequenceIndex == 0 ? "A" : "B");
                    writer.WriteNumber("type_id", piece.TypeId);
                    writer.WriteEndObject();
                    break;
            }
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
    }

    // ─── Property accessors via internal reflection-free field access ─────────────────
    // 简单属性已由 ComponentPropertyWriter 自动生成。
    // 此处仅保留复杂属性访问器。

    // --- PreTokenizer field accessors ---

    private static string GetSplitPatternString(PreTokenizers.SplitPreTokenizer split)
    {
        return split.Pattern switch
        {
            Internal.StringPattern sp => sp.Pattern,
            Internal.RegexPattern rp => rp.Regex.ToString(),
            Internal.CharPattern cp => cp.Char.ToString(),
            _ => split.Pattern.ToString() ?? ""
        };
    }
}
