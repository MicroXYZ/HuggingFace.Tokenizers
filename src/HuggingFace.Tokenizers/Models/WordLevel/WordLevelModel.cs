using System.Text.Json;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Models.WordLevel;

/// <summary>
/// 词级分词模型。
/// 将每个词（空格分隔）映射到其词表 ID。
/// 简单但对词边界清晰的语言有效。
/// </summary>
[TokenizerComponent("WordLevel")]
public sealed class WordLevelModel : DictionaryVocabModel
{
    private WordLevelModel(
        Dictionary<string, uint> vocab,
        string unkToken) : base(vocab, unkToken)
    {
    }

    /// <inheritdoc />
    public override List<Token> Tokenize(ReadOnlySpan<char> sequence)
    {
        if (sequence.IsEmpty)
            return [];

        // WordLevel 将整个输入作为单个 token 查找（与 Rust 一致）
        // 零分配查找：通过 AlternateLookup 直接用 span 查词表
        var vocabLookup = _vocab.GetAlternateLookup<ReadOnlySpan<char>>();
        if (vocabLookup.TryGetValue(sequence, out var id))
        {
            var seqStr = sequence.ToString();
            return [new Token(id, seqStr, 0, sequence.Length)];
        }

        if (_unkToken is not null && _vocab.TryGetValue(_unkToken, out var unkId))
        {
            return [new Token(unkId, _unkToken, 0, sequence.Length)];
        }

        return [];
    }

    /// <summary>
    /// 使用训练结果更新词表。
    /// </summary>
    internal void UpdateVocab(Dictionary<string, uint> newVocab, string unkToken)
    {
        RebuildVocab(newVocab);
        _unkToken = unkToken;
    }

    /// <inheritdoc />
    public override IReadOnlyList<string> Save(string folder, string? prefix = null)
    {
        Directory.CreateDirectory(folder);

        var prefixStr = prefix is not null ? $"{prefix}." : "";
        var vocabPath = Path.Combine(folder, $"{prefixStr}vocab.json");

        using (var fs = File.Create(vocabPath))
        using (var writer = new Utf8JsonWriter(fs, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            foreach (var kvp in _vocab.OrderBy(k => k.Value))
            {
                writer.WritePropertyName(kvp.Key);
                writer.WriteNumberValue(kvp.Value);
            }
            writer.WriteEndObject();
        }

        return new List<string> { vocabPath };
    }

    /// <inheritdoc />

    /// <summary>
    /// 从 vocab.json 文件加载 WordLevel 模型。
    /// </summary>
    public static WordLevelModel Load(string vocabPath, string unkToken = "<unk>")
    {
        var json = File.ReadAllText(vocabPath);
        var vocab = new Dictionary<string, uint>(StringComparer.Ordinal);

        using var doc = JsonDocument.Parse(json);
        foreach (var property in doc.RootElement.EnumerateObject())
        {
            vocab[property.Name] = property.Value.GetUInt32();
        }

        return new WordLevelModel(vocab, unkToken);
    }

    /// <summary>
    /// 构建器，用于构建 <see cref="WordLevelModel"/> 实例的工厂方法。
    /// </summary>
    public sealed class WordLevelBuilder
    {
        private Dictionary<string, uint>? _vocab;
        private string _unkToken = "<unk>";

        /// <summary>
        /// 设置词表。
        /// </summary>
        public WordLevelBuilder SetVocab(Dictionary<string, uint> vocab)
        {
            _vocab = vocab;
            return this;
        }

        /// <summary>
        /// 设置未知 token。
        /// </summary>
        public WordLevelBuilder SetUnkToken(string unkToken)
        {
            _unkToken = unkToken;
            return this;
        }

        /// <summary>
        /// 构建 <see cref="WordLevelModel"/> 实例。
        /// </summary>
        public WordLevelModel Build()
        {
            var vocab = _vocab ?? throw new InvalidOperationException("Vocabulary must be set before building.");
            return new WordLevelModel(vocab, _unkToken);
        }
    }
}
