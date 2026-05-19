namespace HuggingFace.Tokenizers.Benchmarks;

/// <summary>
/// 完整基准测试结果集合
/// </summary>
internal class BenchmarkResults
{
    public string Mode { get; set; } = "";
    public string DotNetVersion { get; set; } = "";
    public string Platform { get; set; } = "";
    public string Timestamp { get; set; } = "";
    public BenchResult? BpeEncodeSingle { get; set; }
    public BenchResult? BpeEncodeBatch { get; set; }
    public BenchResult? BpeEncodeFast { get; set; }
    public BenchResult? UnigramEncodeSingle { get; set; }
    public BenchResult? UnigramEncodeBatch { get; set; }
    public BenchResult? WordPieceEncodeSingle { get; set; }
    public BenchResult? WordPieceEncodeBatch { get; set; }
    public BenchResult? WordPieceEncodeFast { get; set; }
    public BenchResult? WordLevelEncodeSingle { get; set; }
    public BenchResult? WordLevelEncodeBatch { get; set; }
    public BenchResult? Qwen25EncodeSingle { get; set; }
    public BenchResult? Qwen25EncodeBatch { get; set; }
    public BenchResult? SerializationLoad { get; set; }
    public BenchResult? SerializationSave { get; set; }
    public BenchResult? SerializationDeserialize { get; set; }
    public BenchResult? BpeConcurrent4T { get; set; }
    public BenchResult? BpeTruncation512 { get; set; }
    public BenchResult? BpeTruncation128 { get; set; }
    public BenchResult? BpeEncodeCharOffsets { get; set; }
    public BenchResult? UnigramEncodeCharOffsets { get; set; }

    // === 补齐 Rust 版本覆盖但 C# 缺失的基准测试 ===
    public BenchResult? BpeEncodeNoCacheSingle { get; set; }
    public BenchResult? BpeEncodeNoCacheBatch { get; set; }
    public BenchResult? Llama3EncodeSingle { get; set; }
    public BenchResult? Llama3EncodeBatch { get; set; }
    public BenchResult? Llama3EncodeFast { get; set; }
    public BenchResult? Llama3EncodeCharOffsets { get; set; }
    public BenchResult? Llama3Concurrent1T { get; set; }
    public BenchResult? Llama3Concurrent2T { get; set; }
    public BenchResult? Llama3Concurrent4T { get; set; }
    public BenchResult? Llama3Concurrent8T { get; set; }
    public BenchResult? BertPipelineEncode { get; set; }
    public BenchResult? BertPipelineEncodeBatch { get; set; }
    public BenchResult? TruncationScalingByInput1K { get; set; }
    public BenchResult? TruncationScalingByInput10K { get; set; }
    public BenchResult? TruncationScalingByInput100K { get; set; }
    public BenchResult? TruncationScalingByInput500K { get; set; }
    public BenchResult? TruncationScalingByMaxLen128 { get; set; }
    public BenchResult? TruncationScalingByMaxLen512 { get; set; }
    public BenchResult? TruncationScalingByMaxLen2048 { get; set; }
    public BenchResult? TruncationScalingByMaxLen8192 { get; set; }
    public BenchResult? TruncationDirectionLeft { get; set; }
    public BenchResult? TruncationDirectionRight { get; set; }
    public BenchResult? TemplateProcessingSingle { get; set; }
    public BenchResult? TemplateProcessingPair { get; set; }
    public BenchResult? AddedVocabDeserialize100K { get; set; }
    public BenchResult? AddedVocabDeserialize400K { get; set; }
    public BenchResult? AddedVocabDeserialize100KNFKC { get; set; }
    public BenchResult? AddedVocabDeserialize400KNFKC { get; set; }
    public BenchResult? BpeConcurrent1T { get; set; }
    public BenchResult? BpeConcurrent2T { get; set; }
    public BenchResult? BpeConcurrent8T { get; set; }
    public BenchResult? SerializationLoadRoberta { get; set; }
    public BenchResult? SerializationLoadLlama3 { get; set; }
    public BenchResult? SerializationLoadAlbert { get; set; }
}
