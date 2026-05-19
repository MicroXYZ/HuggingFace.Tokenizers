using System;

namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 标记分词器组件类的 JSON 类型名称，用于序列化。
/// 由源代码生成器使用，生成 AOT 兼容的类型名称映射，
/// 消除硬编码字典和反射。
/// </summary>
[AttributeUsage(AttributeTargets.Class, Inherited = false, AllowMultiple = false)]
public sealed class TokenizerComponentAttribute : Attribute
{
    /// <summary>
    /// tokenizer.json 格式中 JSON "type" 字段的值。
    /// 必须与 Rust serde EnumType 变体名称完全匹配。
    /// </summary>
    public string JsonTypeName { get; }

    /// <summary>
    /// 此组件上枚举属性的命名约定。
    /// 默认为 <see cref="EnumNamingConvention.SnakeCase"/>。
    /// 对于 Rust 中没有 <c>#[serde(rename_all = "snake_case")]</c> 的枚举，
    /// 设置为 <see cref="EnumNamingConvention.PascalCase"/>。
    /// </summary>
    public EnumNamingConvention EnumNaming { get; set; } = EnumNamingConvention.SnakeCase;

    public TokenizerComponentAttribute(string jsonTypeName)
    {
        JsonTypeName = jsonTypeName ?? throw new ArgumentNullException(nameof(jsonTypeName));
    }
}
