using System;

namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 指定 JSON 属性名称，用于序列化时与 C# 属性名的 snake_case 形式不同的场景。
/// 由源代码生成的属性写入器使用。
/// </summary>
[AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
public sealed class JsonKeyAttribute : Attribute
{
    /// <summary>
    /// 序列化时使用的 JSON 属性名称。
    /// </summary>
    public string Name { get; }

    public JsonKeyAttribute(string name)
    {
        Name = name ?? throw new ArgumentNullException(nameof(name));
    }
}
