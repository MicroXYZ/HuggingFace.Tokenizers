namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 枚举属性的序列化命名约定。
/// 匹配 Rust serde 行为：snake_case 使用小写，PascalCase 保留原始名称。
/// </summary>
public enum EnumNamingConvention
{
    /// <summary>
    /// 枚举值序列化为小写（如 "always"、"isolated"）。
    /// 匹配 Rust <c>#[serde(rename_all = "snake_case")]</c>。
    /// </summary>
    SnakeCase,

    /// <summary>
    /// 枚举值序列化为 PascalCase（如 "Removed"、"Isolated"）。
    /// 匹配 Rust 默认 serde 行为（无 rename_all）。
    /// </summary>
    PascalCase
}
