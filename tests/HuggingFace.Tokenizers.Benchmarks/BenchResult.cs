namespace HuggingFace.Tokenizers.Benchmarks;

/// <summary>
/// 单项基准测试结果
/// </summary>
internal class BenchResult
{
    public int TotalOps { get; set; }
    public double ElapsedMs { get; set; }
    public double AvgMs { get; set; }
    public double OpsPerSec { get; set; }
    public double UsPerOp { get; set; }
    public double ThroughputMBps { get; set; }
}
