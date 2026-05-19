using System.Text.Json.Serialization;

namespace HuggingFace.Tokenizers.Benchmarks;

/// <summary>
/// System.Text.Json 源生成上下文（AOT 兼容）
/// </summary>
[JsonSourceGenerationOptions(WriteIndented = true)]
[JsonSerializable(typeof(BenchmarkResults))]
[JsonSerializable(typeof(List<ConsistencyEntry>))]
internal partial class BenchmarkJsonContext : JsonSerializerContext;
