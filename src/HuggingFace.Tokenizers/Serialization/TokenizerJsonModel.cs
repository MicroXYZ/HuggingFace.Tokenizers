using System.Text.Json.Serialization;

namespace HuggingFace.Tokenizers.Serialization;

/// <summary>
/// 顶层 tokenizer.json 结构的 JSON 模型。
/// </summary>
public sealed class TokenizerJsonModel
{
    /// <summary>tokenizer.json 格式版本。</summary>
    [JsonPropertyName("version")]
    public string Version { get; set; } = "1.0";

    /// <summary>截断配置。</summary>
    [JsonPropertyName("truncation")]
    public TruncationJsonModel? Truncation { get; set; }

    /// <summary>填充配置。</summary>
    [JsonPropertyName("padding")]
    public PaddingJsonModel? Padding { get; set; }

    /// <summary>已添加的特殊 token 列表。</summary>
    [JsonPropertyName("added_tokens")]
    public List<AddedTokenJsonModel>? AddedTokens { get; set; }

    /// <summary>标准化器配置。</summary>
    [JsonPropertyName("normalizer")]
    public NormalizerJsonModel? Normalizer { get; set; }

    /// <summary>预分词器配置。</summary>
    [JsonPropertyName("pre_tokenizer")]
    public PreTokenizerJsonModel? PreTokenizer { get; set; }

    /// <summary>分词模型配置。</summary>
    [JsonPropertyName("model")]
    public ModelJsonModel? Model { get; set; }

    /// <summary>后处理器配置。</summary>
    [JsonPropertyName("post_processor")]
    public PostProcessorJsonModel? PostProcessor { get; set; }

    /// <summary>解码器配置。</summary>
    [JsonPropertyName("decoder")]
    public DecoderJsonModel? Decoder { get; set; }
}

/// <summary>
/// 已添加 token 的 JSON 模型。
/// </summary>
public sealed class AddedTokenJsonModel
{
    /// <summary>token 的词汇表 ID。</summary>
    [JsonPropertyName("id")]
    public uint Id { get; set; }

    /// <summary>token 的字符串内容。</summary>
    [JsonPropertyName("content")]
    public string Content { get; set; } = "";

    /// <summary>是否仅匹配完整单词。</summary>
    [JsonPropertyName("single_word")]
    public bool SingleWord { get; set; }

    /// <summary>是否去除左侧空白。</summary>
    [JsonPropertyName("lstrip")]
    public bool LStrip { get; set; }

    /// <summary>是否去除右侧空白。</summary>
    [JsonPropertyName("rstrip")]
    public bool RStrip { get; set; }

    /// <summary>是否进行标准化。</summary>
    [JsonPropertyName("normalized")]
    public bool Normalized { get; set; }

    /// <summary>是否为特殊 token。</summary>
    [JsonPropertyName("special")]
    public bool Special { get; set; }
}

/// <summary>
/// 截断配置的 JSON 模型。
/// </summary>
public sealed class TruncationJsonModel
{
    /// <summary>截断策略类型。</summary>
    [JsonPropertyName("type")]
    public string Type { get; set; } = "LongestFirst";

    /// <summary>最大长度。</summary>
    [JsonPropertyName("max_length")]
    public int MaxLength { get; set; }

    /// <summary>滑动窗口步长。</summary>
    [JsonPropertyName("stride")]
    public int Stride { get; set; }

    /// <summary>截断方向。</summary>
    [JsonPropertyName("direction")]
    public string Direction { get; set; } = "Right";
}

/// <summary>
/// 填充配置的 JSON 模型。
/// </summary>
public sealed class PaddingJsonModel
{
    /// <summary>填充策略类型。Rust PaddingStrategy 可能是字符串 "BatchLongest" 或对象 {"Fixed": size}。</summary>
    [JsonPropertyName("type")]
    public System.Text.Json.JsonElement TypeElement { get; set; }

    /// <summary>
    /// 解析填充策略类型字符串。
    /// 支持 Rust serde 格式：字符串 "BatchLongest" 或对象 {"Fixed": size}。
    /// </summary>
    [System.Text.Json.Serialization.JsonIgnore]
    public string Type
    {
        get
        {
            if (TypeElement.ValueKind == System.Text.Json.JsonValueKind.String)
                return TypeElement.GetString() ?? "BatchLongest";
            if (TypeElement.ValueKind == System.Text.Json.JsonValueKind.Object
                && TypeElement.TryGetProperty("Fixed", out _))
                return "Fixed";
            return "BatchLongest";
        }
    }

    /// <summary>
    /// 当 Type 为 "Fixed" 时，提取 Fixed 变体的值。
    /// </summary>
    [System.Text.Json.Serialization.JsonIgnore]
    public int? FixedLength
    {
        get
        {
            if (TypeElement.ValueKind == System.Text.Json.JsonValueKind.Object
                && TypeElement.TryGetProperty("Fixed", out var fixedEl)
                && fixedEl.ValueKind == System.Text.Json.JsonValueKind.Number)
                return fixedEl.GetInt32();
            return null;
        }
    }

    /// <summary>固定填充时的长度。</summary>
    [JsonPropertyName("length")]
    public int? Length { get; set; }

    /// <summary>填充 token 的 ID。</summary>
    [JsonPropertyName("pad_id")]
    public uint PadId { get; set; }

    /// <summary>填充 token 的类型 ID。</summary>
    [JsonPropertyName("pad_type_id")]
    public uint PadTypeId { get; set; }

    /// <summary>填充 token 的字符串表示。</summary>
    [JsonPropertyName("pad_token")]
    public string PadToken { get; set; } = "[PAD]";

    /// <summary>填充方向。</summary>
    [JsonPropertyName("direction")]
    public string Direction { get; set; } = "Right";

    /// <summary>填充到此倍数。</summary>
    [JsonPropertyName("pad_to_multiple_of")]
    public int? PadToMultipleOf { get; set; }
}
