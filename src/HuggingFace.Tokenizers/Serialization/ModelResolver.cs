using System.Text.Json;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Models.BPE;
using HuggingFace.Tokenizers.Models.Unigram;
using HuggingFace.Tokenizers.Models.WordLevel;
using HuggingFace.Tokenizers.Models.WordPiece;
using static HuggingFace.Tokenizers.Serialization.JsonElementHelper;

namespace HuggingFace.Tokenizers.Serialization;

/// <summary>
/// 解析 <see cref="ModelJsonModel"/> 反序列化为 concrete <see cref="IModel"/> 实例。
/// </summary>
public static class ModelResolver
{
    /// <summary>
    /// 根据 JSON 模型解析具体的分词模型实例。
    /// </summary>
    public static IModel Resolve(ModelJsonModel model)
    {
        return model.Type switch
        {
            "BPE" => ResolveBpe(model),
            "WordPiece" => ResolveWordPiece(model),
            "WordLevel" => ResolveWordLevel(model),
            "Unigram" => ResolveUnigram(model),
            _ => throw new NotSupportedException($"Unsupported model type: '{model.Type}'.")
        };
    }

    private static BpeModel ResolveBpe(ModelJsonModel model)
    {
        var data = model.AdditionalData;
        var vocab = RequireProperty(data, "vocab", ReadVocab);
        var merges = TryGetProperty(data, "merges", ReadMerges) ?? new List<(string, string)>();

        var builder = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges(merges);

        if (GetString(data, "unk_token") is { } unkToken) builder.SetUnkToken(unkToken);
        if (GetString(data, "continuing_subword_prefix") is { } csp) builder.SetContinuingSubwordPrefix(csp);
        if (GetString(data, "end_of_word_suffix") is { } eows) builder.SetEndOfWordSuffix(eows);
        if (GetOptionalBool(data, "fuse_unk") is { } fuseUnk) builder.SetFuseUnk(fuseUnk);
        if (GetOptionalBool(data, "byte_fallback") is { } byteFallback) builder.SetByteFallback(byteFallback);
        if (GetOptionalBool(data, "ignore_merges") is { } ignoreMerges) builder.SetIgnoreMerges(ignoreMerges);
        if (GetOptionalFloat(data, "dropout") is { } dropout) builder.SetDropout(dropout);
        if (GetInt(data, "cache_capacity") is { } cacheCapacity) builder.SetCacheCapacity(cacheCapacity);

        return builder.Build();
    }

    private static WordPieceModel ResolveWordPiece(ModelJsonModel model)
    {
        var data = model.AdditionalData;
        var vocab = RequireProperty(data, "vocab", ReadVocab);

        var builder = new WordPieceModel.WordPieceBuilder().SetVocab(vocab);
        if (GetString(data, "unk_token") is { } unkToken) builder.SetUnkToken(unkToken);
        if (GetString(data, "continuing_subword_prefix") is { } csp) builder.SetContinuingSubwordPrefix(csp);
        if (GetInt(data, "max_input_chars_per_word") is { } maxChars) builder.SetMaxInputCharsPerWord(maxChars);

        return builder.Build();
    }

    private static WordLevelModel ResolveWordLevel(ModelJsonModel model)
    {
        var data = model.AdditionalData;
        var vocab = RequireProperty(data, "vocab", ReadVocab);

        var builder = new WordLevelModel.WordLevelBuilder().SetVocab(vocab);
        if (GetString(data, "unk_token") is { } unkToken) builder.SetUnkToken(unkToken);

        return builder.Build();
    }

    private static UnigramModel ResolveUnigram(ModelJsonModel model)
    {
        var data = model.AdditionalData;
        var vocab = RequireProperty(data, "vocab", ReadUnigramVocab);

        var builder = new UnigramModel.UnigramBuilder().SetVocab(vocab);
        if (GetInt(data, "unk_id") is { } unkId && unkId >= 0 && unkId < vocab.Count)
            builder.SetUnkToken(vocab[unkId].Token);
        if (GetOptionalBool(data, "byte_fallback") is { } byteFallback) builder.SetByteFallback(byteFallback);

        return builder.Build();
    }

    // ── Vocab readers ────────────────────────────────────────────────────────

    private static Dictionary<string, uint> ReadVocab(JsonElement element)
    {
        var vocab = new Dictionary<string, uint>(StringComparer.Ordinal);
        foreach (var prop in element.EnumerateObject())
            vocab[prop.Name] = JsonElementHelper.GetUInt32Loose(prop.Value);
        return vocab;
    }

    private static List<(string, string)> ReadMerges(JsonElement element)
    {
        var merges = new List<(string, string)>();
        foreach (var item in element.EnumerateArray())
        {
            if (item.ValueKind == JsonValueKind.Array)
            {
                var arr = item.EnumerateArray().ToArray();
                if (arr.Length >= 2)
                    merges.Add((arr[0].GetString()!, arr[1].GetString()!));
            }
            else if (item.ValueKind == JsonValueKind.String)
            {
                var parts = item.GetString()!.Split(' ', 2);
                if (parts.Length == 2)
                    merges.Add((parts[0], parts[1]));
            }
        }
        return merges;
    }

    private static List<(string Token, double LogProb)> ReadUnigramVocab(JsonElement element)
    {
        var vocab = new List<(string Token, double LogProb)>();
        foreach (var item in element.EnumerateArray())
        {
            var arr = item.EnumerateArray().ToArray();
            if (arr.Length >= 2)
                vocab.Add((arr[0].GetString()!, arr[1].GetDouble()));
        }
        return vocab;
    }
}
