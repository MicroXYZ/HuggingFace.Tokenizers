using System.Text.Json;
using System.Text.Json.Serialization;

namespace HuggingFace.Tokenizers.Serialization;

/// <summary>
/// 组件 JSON 模型基类。
/// Normalizer、PreTokenizer、Model、PostProcessor、Decoder 五个组件
/// 的 JSON 结构完全相同（type + 扩展数据），共用此基类消除重复。
/// </summary>
public class ComponentJsonModel
{
    /// <summary>组件类型名称（如 "BPE"、"WordPiece" 等）。</summary>
    [JsonPropertyName("type")]
    public string Type { get; set; } = "";

    /// <summary>
    /// 以原始 JSON 存储的附加属性，用于灵活反序列化。
    /// </summary>
    [JsonExtensionData]
    public Dictionary<string, JsonElement>? AdditionalData { get; set; }
}
