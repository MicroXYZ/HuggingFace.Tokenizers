namespace HuggingFace.Tokenizers.Benchmarks;

/// <summary>
/// 编码+解码一致性条目（与 Rust ConsistencyEntry 对齐）
/// </summary>
internal class ConsistencyEntry
{
    public string Input { get; set; } = "";
    public uint[] Ids { get; set; } = [];
    public string[] Tokens { get; set; } = [];
    public string Decoded { get; set; } = "";
}
