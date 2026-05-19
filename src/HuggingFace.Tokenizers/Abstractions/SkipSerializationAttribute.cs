using System;

namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 标记属性应被源代码生成的属性写入器跳过。
/// 用于需要自定义序列化逻辑（模式、集合等）或在序列化器中手动处理的属性。
/// </summary>
[AttributeUsage(AttributeTargets.Property, AllowMultiple = false, Inherited = false)]
public sealed class SkipSerializationAttribute : Attribute
{
}
