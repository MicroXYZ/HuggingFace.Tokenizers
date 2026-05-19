using System.Diagnostics;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Serialization;

namespace HuggingFace.Tokenizers.Benchmarks;

/// <summary>
/// 单算法基准测试 — 开发时快速验证性能变化。
/// 用法：
///   dotnet run -- --per-algo --bpe
///   dotnet run -- --per-algo --unigram
///   dotnet run -- --per-algo --wordpiece
///   dotnet run -- --per-algo --wordlevel
///   dotnet run -- --per-algo --all
///   dotnet run -- --per-algo --all --json
///   dotnet run -- --per-algo --all --iterations 50
/// </summary>
public static class PerAlgorithmBenchmarks
{
    private static string DataDir => Path.Combine(AppContext.BaseDirectory, "data");

    /// <summary>
    /// 入口方法，返回 0 表示成功。
    /// </summary>
    public static int Run(string[] args)
    {
        bool jsonOutput = args.Contains("--json");
        bool runAll = args.Contains("--all");
        bool runBpe = args.Contains("--bpe") || runAll;
        bool runUnigram = args.Contains("--unigram") || runAll;
        bool runWordPiece = args.Contains("--wordpiece") || runAll;
        bool runWordLevel = args.Contains("--wordlevel") || runAll;

        // 解析迭代次数
        int singleIter = 100;
        int batchIter = 20;
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == "--iterations" && int.TryParse(args[i + 1], out var n))
            {
                singleIter = n;
                batchIter = Math.Max(5, n / 5);
            }
        }

        if (!runBpe && !runUnigram && !runWordPiece && !runWordLevel)
        {
            Console.WriteLine("用法: dotnet run -- --per-algo [--bpe|--unigram|--wordpiece|--wordlevel|--all] [--json] [--iterations N]");
            return 1;
        }

        Console.WriteLine("╔══════════════════════════════════════════════╗");
        Console.WriteLine("║   Per-Algorithm Benchmarks (快速验证)       ║");
        Console.WriteLine("╠══════════════════════════════════════════════╣");
        Console.WriteLine($"║  Runtime:  .NET {Environment.Version}, {IntPtr.Size * 8}-bit");
        Console.WriteLine($"║  CPU:      {Environment.ProcessorCount} cores");
        Console.WriteLine($"║  Iter:     single={singleIter}, batch={batchIter}");
        Console.WriteLine("╚══════════════════════════════════════════════╝");
        Console.WriteLine();

        // 加载测试数据
        var smallPath = Path.Combine(DataDir, "small.txt");
        if (!File.Exists(smallPath))
        {
            Console.WriteLine($"❌ 测试数据未找到: {smallPath}");
            Console.WriteLine("   请先运行完整基准测试生成测试数据: dotnet run -- --bench");
            return 1;
        }

        var smallLines = File.ReadAllLines(smallPath);
        var testLines = smallLines.Take(1000).ToArray();
        Console.WriteLine($"测试数据: {smallLines.Length} 行, 取前 {testLines.Length} 行");
        Console.WriteLine();

        var allResults = new Dictionary<string, Dictionary<string, BenchResult>>();

        if (runBpe)
        {
            var results = RunBpeBenchmark(testLines, smallLines, singleIter, batchIter);
            if (jsonOutput) allResults["bpe"] = results;
        }

        if (runUnigram)
        {
            var results = RunUnigramBenchmark(testLines, smallLines, singleIter, batchIter);
            if (jsonOutput) allResults["unigram"] = results;
        }

        if (runWordPiece)
        {
            var results = RunWordPieceBenchmark(testLines, smallLines, singleIter, batchIter);
            if (jsonOutput) allResults["wordpiece"] = results;
        }

        if (runWordLevel)
        {
            var results = RunWordLevelBenchmark(testLines, smallLines, singleIter, batchIter);
            if (jsonOutput) allResults["wordlevel"] = results;
        }

        if (jsonOutput && allResults.Count > 0)
        {
            // 手动构建 JSON 输出（AOT 安全）
            var sb = new System.Text.StringBuilder();
            sb.AppendLine("{");
            bool firstAlgo = true;
            foreach (var (algo, results) in allResults)
            {
                if (!firstAlgo) sb.AppendLine(",");
                firstAlgo = false;
                sb.AppendLine($"  \"{algo}\": {{");
                bool first = true;
                foreach (var (key, val) in results)
                {
                    if (!first) sb.AppendLine(",");
                    first = false;
                    sb.Append($"    \"{key}\": {{");
                    sb.Append($"\"OpsPerSec\":{val.OpsPerSec:F0},");
                    sb.Append($"\"UsPerOp\":{val.UsPerOp:F2},");
                    sb.Append($"\"TotalOps\":{val.TotalOps},");
                    sb.Append($"\"ElapsedMs\":{val.ElapsedMs:F1}");
                    sb.Append("}");
                }
                sb.AppendLine();
                sb.Append("  }");
            }
            sb.AppendLine();
            sb.AppendLine("}");
            var outPath = Path.Combine(DataDir, "per-algo-bench.json");
            File.WriteAllText(outPath, sb.ToString());
            Console.WriteLine($"JSON 结果已保存: {outPath}");
        }

        return 0;
    }

    // ════════════════════════════════════════════════
    //  BPE
    // ════════════════════════════════════════════════

    private static Dictionary<string, BenchResult> RunBpeBenchmark(
        string[] testLines, string[] allLines, int singleIter, int batchIter)
    {
        Console.WriteLine("═══════════════════════════════════");
        Console.WriteLine(" BPE 基准测试");
        Console.WriteLine("═══════════════════════════════════");

        var tokenizer = LoadTokenizer("tokenizer-rust-bpe.json", "BPE");
        if (tokenizer is null) return new();

        var results = new Dictionary<string, BenchResult>();

        // 单条 Encode
        results["EncodeSingle"] = BenchmarkSingle(tokenizer, testLines, singleIter, "BPE Encode (single)");

        // 批量 Encode
        results["EncodeBatch"] = BenchmarkBatch(tokenizer, allLines, batchIter, "BPE Encode (batch)");

        // EncodeFast
        results["EncodeFast"] = BenchmarkEncodeFast(tokenizer, testLines, singleIter, "BPE EncodeFast");

        // EncodeCharOffsets
        results["EncodeCharOffsets"] = BenchmarkCharOffsets(tokenizer, testLines, singleIter, "BPE EncodeCharOffsets");

        // 序列化
        results["SerializationLoad"] = BenchmarkSerializationLoad(tokenizer, "BPE Serialization Load");
        results["SerializationSave"] = BenchmarkSerializationSave(tokenizer, "BPE Serialization Save");

        // 并发
        results["Concurrent4T"] = BenchmarkConcurrent(tokenizer, testLines, 4, "BPE Concurrent (4T)");

        Console.WriteLine();
        return results;
    }

    // ════════════════════════════════════════════════
    //  Unigram
    // ════════════════════════════════════════════════

    private static Dictionary<string, BenchResult> RunUnigramBenchmark(
        string[] testLines, string[] allLines, int singleIter, int batchIter)
    {
        Console.WriteLine("═══════════════════════════════════");
        Console.WriteLine(" Unigram 基准测试");
        Console.WriteLine("═══════════════════════════════════");

        var tokenizer = LoadTokenizer("tokenizer-rust-unigram.json", "Unigram");
        if (tokenizer is null) return new();

        var results = new Dictionary<string, BenchResult>();

        results["EncodeSingle"] = BenchmarkSingle(tokenizer, testLines, singleIter, "Unigram Encode (single)");
        results["EncodeBatch"] = BenchmarkBatch(tokenizer, allLines, batchIter, "Unigram Encode (batch)");
        results["EncodeCharOffsets"] = BenchmarkCharOffsets(tokenizer, testLines, singleIter, "Unigram EncodeCharOffsets");
        results["SerializationLoad"] = BenchmarkSerializationLoad(tokenizer, "Unigram Serialization Load");
        results["SerializationSave"] = BenchmarkSerializationSave(tokenizer, "Unigram Serialization Save");

        Console.WriteLine();
        return results;
    }

    // ════════════════════════════════════════════════
    //  WordPiece
    // ════════════════════════════════════════════════

    private static Dictionary<string, BenchResult> RunWordPieceBenchmark(
        string[] testLines, string[] allLines, int singleIter, int batchIter)
    {
        Console.WriteLine("═══════════════════════════════════");
        Console.WriteLine(" WordPiece 基准测试");
        Console.WriteLine("═══════════════════════════════════");

        var tokenizer = LoadTokenizer("tokenizer-rust-wordpiece.json", "WordPiece");
        if (tokenizer is null) return new();

        var results = new Dictionary<string, BenchResult>();

        results["EncodeSingle"] = BenchmarkSingle(tokenizer, testLines, singleIter, "WordPiece Encode (single)");
        results["EncodeBatch"] = BenchmarkBatch(tokenizer, allLines, batchIter, "WordPiece Encode (batch)");
        results["EncodeFast"] = BenchmarkEncodeFast(tokenizer, testLines, singleIter, "WordPiece EncodeFast");
        results["SerializationLoad"] = BenchmarkSerializationLoad(tokenizer, "WordPiece Serialization Load");
        results["SerializationSave"] = BenchmarkSerializationSave(tokenizer, "WordPiece Serialization Save");

        Console.WriteLine();
        return results;
    }

    // ════════════════════════════════════════════════
    //  WordLevel
    // ════════════════════════════════════════════════

    private static Dictionary<string, BenchResult> RunWordLevelBenchmark(
        string[] testLines, string[] allLines, int singleIter, int batchIter)
    {
        Console.WriteLine("═══════════════════════════════════");
        Console.WriteLine(" WordLevel 基准测试");
        Console.WriteLine("═══════════════════════════════════");

        var tokenizer = LoadTokenizer("tokenizer-rust-wordlevel.json", "WordLevel");
        if (tokenizer is null) return new();

        var results = new Dictionary<string, BenchResult>();

        results["EncodeSingle"] = BenchmarkSingle(tokenizer, testLines, singleIter, "WordLevel Encode (single)");
        results["EncodeBatch"] = BenchmarkBatch(tokenizer, allLines, batchIter, "WordLevel Encode (batch)");
        results["SerializationLoad"] = BenchmarkSerializationLoad(tokenizer, "WordLevel Serialization Load");
        results["SerializationSave"] = BenchmarkSerializationSave(tokenizer, "WordLevel Serialization Save");

        Console.WriteLine();
        return results;
    }

    // ════════════════════════════════════════════════
    //  通用基准测试方法
    // ════════════════════════════════════════════════

    private static BenchResult BenchmarkSingle(Tokenizer tokenizer, string[] lines, int iterations, string label)
    {
        Console.WriteLine($"  --- {label} ({iterations} × {lines.Length} 行) ---");

        // Warmup
        foreach (var line in lines.Take(100)) tokenizer.Encode(line);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            foreach (var line in lines)
                tokenizer.Encode(line);
        sw.Stop();

        int total = iterations * lines.Length;
        var result = MakeResult(total, sw);
        PrintBenchResult(result, total, sw);
        return result;
    }

    private static BenchResult BenchmarkBatch(Tokenizer tokenizer, string[] allLines, int iterations, string label)
    {
        int batchSize = 1000;
        Console.WriteLine($"  --- {label} ({iterations} × batch={batchSize}) ---");

        var batches = new List<string[]>();
        for (int i = 0; i < allLines.Length; i += batchSize)
        {
            var remaining = Math.Min(batchSize, allLines.Length - i);
            if (remaining > 0) batches.Add(allLines[i..(i + remaining)]);
        }

        // Warmup
        foreach (var batch in batches.Take(2))
        {
            tokenizer.EncodeBatch(batch);
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            foreach (var batch in batches)
            {
                tokenizer.EncodeBatch(batch);
            }
        sw.Stop();

        int totalLines = iterations * allLines.Length;
        var result = MakeResult(totalLines, sw);
        PrintBenchResult(result, totalLines, sw);
        return result;
    }

    private static BenchResult BenchmarkEncodeFast(Tokenizer tokenizer, string[] lines, int iterations, string label)
    {
        Console.WriteLine($"  --- {label} ({iterations} × {lines.Length} 行) ---");

        // Warmup
        foreach (var line in lines.Take(100)) { tokenizer.Encode(line); tokenizer.EncodeFast(line); }

        // Encode (基线)
        var swBaseline = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            foreach (var line in lines)
                tokenizer.Encode(line);
        swBaseline.Stop();

        // EncodeFast
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            foreach (var line in lines)
                tokenizer.EncodeFast(line);
        sw.Stop();

        int total = iterations * lines.Length;
        var result = MakeResult(total, sw);
        Console.WriteLine($"    Encode:     {swBaseline.Elapsed.TotalMilliseconds:F1}ms ({total / swBaseline.Elapsed.TotalSeconds:F0} ops/sec)");
        Console.WriteLine($"    EncodeFast: {sw.Elapsed.TotalMilliseconds:F1}ms ({result.OpsPerSec:F0} ops/sec)");
        Console.WriteLine($"    加速比:     {swBaseline.Elapsed.TotalMilliseconds / sw.Elapsed.TotalMilliseconds:F2}x");
        Console.WriteLine();
        return result;
    }

    private static BenchResult BenchmarkCharOffsets(Tokenizer tokenizer, string[] lines, int iterations, string label)
    {
        Console.WriteLine($"  --- {label} ({iterations} × {lines.Length} 行) ---");

        // Warmup
        tokenizer.EncodeBatchCharOffsets(lines.Take(100).ToArray());

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            tokenizer.EncodeBatchCharOffsets(lines);
        }
        sw.Stop();

        int total = iterations * lines.Length;
        var result = MakeResult(total, sw);
        PrintBenchResult(result, total, sw);
        return result;
    }

    private static BenchResult BenchmarkSerializationLoad(Tokenizer tokenizer, string label)
    {
        Console.WriteLine($"  --- {label} ---");
        var json = tokenizer.ToJson();

        // Warmup
        for (int i = 0; i < 3; i++) TokenizerLoader.FromJson(json);

        const int iterations = 100;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            TokenizerLoader.FromJson(json);
        sw.Stop();

        var result = MakeResult(iterations, sw);
        Console.WriteLine($"    {iterations} 次: {sw.Elapsed.TotalMilliseconds:F1}ms, 平均: {result.AvgMs:F2}ms/次");
        Console.WriteLine();
        return result;
    }

    private static BenchResult BenchmarkSerializationSave(Tokenizer tokenizer, string label)
    {
        Console.WriteLine($"  --- {label} ---");

        // Warmup
        for (int i = 0; i < 3; i++) tokenizer.ToJson();

        const int iterations = 100;
        string? lastJson = null;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            lastJson = tokenizer.ToJson();
        sw.Stop();

        var result = MakeResult(iterations, sw);
        Console.WriteLine($"    {iterations} 次: {sw.Elapsed.TotalMilliseconds:F1}ms, 平均: {result.AvgMs:F2}ms/次, JSON: {(lastJson?.Length ?? 0) / 1024}KB");
        Console.WriteLine();
        return result;
    }

    private static BenchResult BenchmarkConcurrent(Tokenizer tokenizer, string[] lines, int numThreads, string label)
    {
        Console.WriteLine($"  --- {label} ({numThreads} threads) ---");

        int linesPerThread = lines.Length / numThreads;
        var inputs = new string[numThreads];
        for (int t = 0; t < numThreads; t++)
            inputs[t] = string.Join("\n", lines.AsSpan(t * linesPerThread, linesPerThread).ToArray());

        // Warmup
        Parallel.ForEach(inputs, input => tokenizer.Encode(input));

        const int iterations = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            Parallel.ForEach(inputs, new ParallelOptions { MaxDegreeOfParallelism = numThreads },
                input => tokenizer.Encode(input));
        sw.Stop();

        int totalOps = iterations * numThreads;
        long totalBytes = inputs.Sum(s => (long)s.Length);
        var result = new BenchResult
        {
            TotalOps = totalOps,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = totalOps / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / totalOps,
            ThroughputMBps = (totalBytes / (1024.0 * 1024.0)) / (sw.Elapsed.TotalSeconds / iterations)
        };
        Console.WriteLine($"    {totalOps} ops: {sw.Elapsed.TotalMilliseconds:F1}ms, {result.OpsPerSec:F0} ops/sec, {result.ThroughputMBps:F1} MiB/s");
        Console.WriteLine();
        return result;
    }

    // ════════════════════════════════════════════════
    //  辅助方法
    // ════════════════════════════════════════════════

    private static Tokenizer? LoadTokenizer(string fileName, string algoName)
    {
        var path = Path.Combine(DataDir, fileName);
        if (!File.Exists(path))
        {
            Console.WriteLine($"  ⚠ 模型文件未找到: {fileName}");
            Console.WriteLine($"    请先运行完整基准测试生成模型: dotnet run -- --bench");
            Console.WriteLine();
            return null;
        }

        Console.WriteLine($"  ✅ 加载模型: {fileName}");
        var tokenizer = TokenizerLoader.FromFile(path);
        Console.WriteLine($"    词表大小: {tokenizer.GetVocabSizeWithAddedTokens():N0}");
        Console.WriteLine();
        return tokenizer;
    }

    private static BenchResult MakeResult(int totalOps, Stopwatch sw) => new()
    {
        TotalOps = totalOps,
        ElapsedMs = sw.Elapsed.TotalMilliseconds,
        AvgMs = sw.Elapsed.TotalMilliseconds / totalOps,
        OpsPerSec = totalOps / sw.Elapsed.TotalSeconds,
        UsPerOp = sw.Elapsed.TotalMicroseconds / totalOps
    };

    private static void PrintBenchResult(BenchResult result, int totalOps, Stopwatch sw)
    {
        Console.WriteLine($"    {totalOps:N0} ops: {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"    吞吐量: {result.OpsPerSec:F0} ops/sec, 延迟: {result.UsPerOp:F2} µs/op");
        Console.WriteLine();
    }
}
