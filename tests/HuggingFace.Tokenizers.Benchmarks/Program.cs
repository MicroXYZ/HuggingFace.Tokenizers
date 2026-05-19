using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Decoders;
using HuggingFace.Tokenizers.Internal;
using HuggingFace.Tokenizers.Models.BPE;
using HuggingFace.Tokenizers.Models.Unigram;
using HuggingFace.Tokenizers.Models.WordPiece;
using HuggingFace.Tokenizers.Models.WordLevel;
using HuggingFace.Tokenizers.Normalizers;
using HuggingFace.Tokenizers.PreTokenizers;
using HuggingFace.Tokenizers.Processors;
using HuggingFace.Tokenizers.Serialization;

namespace HuggingFace.Tokenizers.Benchmarks;

/// <summary>
/// 全面基准测试 — 与 Rust criterion 结果对齐。
/// 覆盖：BPE/Unigram/WordPiece 训练与编码、批量编码、AOT/JIT 对比、数据一致性验证。
/// </summary>
public static class Program
{
    private static string DataDir => Path.Combine(AppContext.BaseDirectory, "data");

    public static async Task<int> Main(string[] args)
    {
        var mode = DetectRuntimeMode(args);
        var runConsistency = args.Contains("--consistency") || args.Contains("--all");
        var runBench = args.Contains("--bench") || args.Contains("--all") || args.Length == 0;
        var runNormalization = args.Contains("--normalization");
        var runPerAlgo = args.Contains("--per-algo");
        var runMicro = args.Contains("--micro");
        var runProfile = args.Contains("--profile");
        var runNfcProfile = args.Contains("--nfc-profile");
        var jsonOutput = args.Contains("--json");

        if (runNormalization)
        {
            NormalizationBenchmarks.Run();
            return 0;
        }

        // ── 单算法快速基准测试 ──
        if (runPerAlgo)
        {
            return PerAlgorithmBenchmarks.Run(args);
        }

        // ── 微基准测试（Phase 4/5 验证） ──
        if (runMicro)
        {
            return MicroBenchmarks.Run(args);
        }

        // ── 并发 Profile ──
        if (runProfile)
        {
            ConcurrencyProfile.Run();
            return 0;
        }

        // ── NFC Profile ──
        if (runNfcProfile)
        {
            NfcProfile.Run();
            return 0;
        }

        // ── --run-id 参数（用于一致性测试，指定运行编号 1/2/3） ──
        int runId = 0;
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--run-id" && i + 1 < args.Length && int.TryParse(args[i + 1], out var rid))
                runId = rid;
            else if (args[i].StartsWith("--run-id=") && int.TryParse(args[i]["--run-id=".Length..], out var rid2))
                runId = rid2;
        }

        // ── 从环境变量读取资源限制（由 run.cs 设置） ──
        string? resourceProfile = Environment.GetEnvironmentVariable("BENCH_RESOURCE_PROFILE");
        int maxConcurrent = int.TryParse(Environment.GetEnvironmentVariable("BENCH_MAX_CONCURRENT"), out var mc) ? mc : Environment.ProcessorCount;
        bool skipLargeDownloads = Environment.GetEnvironmentVariable("BENCH_SKIP_LARGE_DOWNLOADS") == "1";
        bool skip400KVocab = Environment.GetEnvironmentVariable("BENCH_SKIP_400K_VOCAB") == "1";
        bool isMinimal = resourceProfile == "minimal";

        Console.WriteLine($"=== Tokenizers.NET Comprehensive Benchmarks [{mode}] ===");
        Console.WriteLine($"Runtime: .NET {Environment.Version}, {IntPtr.Size * 8}-bit");
        Console.WriteLine($"OS: {RuntimeInformation.OSDescription}");
        Console.WriteLine($"CPU: {Environment.ProcessorCount} cores");
        if (resourceProfile is not null)
            Console.WriteLine($"Resource: {resourceProfile} (maxConcurrent={maxConcurrent}, skipLargeDL={skipLargeDownloads})");
        Console.WriteLine();

        var smallPath = Path.Combine(DataDir, "small.txt");
        var bigPath = Path.Combine(DataDir, "big.txt");

        if (!File.Exists(smallPath) || !File.Exists(bigPath))
        {
            Console.WriteLine($"ERROR: Test data not found in {DataDir}");
            return 1;
        }

        var smallData = File.ReadAllText(smallPath);
        var bigData = File.ReadAllText(bigPath);
        var smallLines = File.ReadAllLines(smallPath);
        var bigLines = File.ReadAllLines(bigPath);

        Console.WriteLine($"small.txt: {smallData.Length:N0} chars ({smallData.Length / 1024} KB)");
        Console.WriteLine($"big.txt:   {bigData.Length:N0} chars ({bigData.Length / 1024} KB)");
        Console.WriteLine();

        var results = new BenchmarkResults
        {
            Mode = mode,
            DotNetVersion = Environment.Version.ToString(),
            Platform = $"{IntPtr.Size * 8}-bit",
            Timestamp = DateTime.UtcNow.ToString("o")
        };

        if (runBench)
        {
            // === BPE 推理 ===
            results.BpeEncodeSingle = BenchmarkBpeEncodeSingle(smallLines);
            results.BpeEncodeBatch = BenchmarkBpeEncodeBatch(smallLines, 1000);

            // === Unigram 推理 ===
            results.UnigramEncodeSingle = BenchmarkUnigramEncodeSingle(smallLines);
            results.UnigramEncodeBatch = BenchmarkUnigramEncodeBatch(smallLines, 1000);

            // === WordPiece 推理 ===
            results.WordPieceEncodeSingle = BenchmarkWordPieceEncodeSingle(smallLines);
            results.WordPieceEncodeBatch = BenchmarkWordPieceEncodeBatch(smallLines, 1000);

            // === WordLevel 推理 ===
            results.WordLevelEncodeSingle = BenchmarkWordLevelEncodeSingle(smallLines);
            results.WordLevelEncodeBatch = BenchmarkWordLevelEncodeBatch(smallLines, 1000);

            // === EncodeFast vs Encode ===
            results.BpeEncodeFast = BenchmarkBpeEncodeFast(smallLines);
            results.WordPieceEncodeFast = BenchmarkWordPieceEncodeFast(smallLines);

            // === Qwen2.5 ===
            if (!skipLargeDownloads)
            {
                results.Qwen25EncodeSingle = await BenchmarkTokenizerFromJson("Qwen/Qwen2.5-7B-Instruct", bigLines);
                results.Qwen25EncodeBatch = await BenchmarkTokenizerFromJsonBatch("Qwen/Qwen2.5-7B-Instruct", bigLines, 1000);
            }
            else
            {
                Console.WriteLine("⏭ 跳过 Qwen2.5 (资源受限，跳过大型 tokenizer 下载)");
            }

            // === Serialization ===
            results.SerializationLoad = BenchmarkSerializationLoad();
            results.SerializationSave = BenchmarkSerializationSave();
            results.SerializationDeserialize = BenchmarkSerializationDeserialize();

            // === Concurrent (4 threads, 受资源限制) ===
            results.BpeConcurrent4T = BenchmarkBpeConcurrent(smallLines, Math.Min(4, maxConcurrent));

            // === Truncation ===
            results.BpeTruncation512 = BenchmarkBpeTruncation(smallLines, 512);
            results.BpeTruncation128 = BenchmarkBpeTruncation(smallLines, 128);

            // === EncodeCharOffsets ===
            results.BpeEncodeCharOffsets = BenchmarkBpeEncodeCharOffsets(smallLines);
            results.UnigramEncodeCharOffsets = BenchmarkUnigramEncodeCharOffsets(smallLines);

            // === 新增：补齐 Rust 版本覆盖但 C# 缺失的基准测试 ===

            // BPE Encode No Cache（C# 不支持 cache_capacity(0)，跳过）
            var (noCacheSingle, noCacheBatch) = BenchmarkBpeEncodeNoCache(smallLines);
            results.BpeEncodeNoCacheSingle = noCacheSingle;
            results.BpeEncodeNoCacheBatch = noCacheBatch;

            // Llama3 系列（需要从 HF 下载，资源受限时跳过）
            if (!skipLargeDownloads)
            {
                var (llama3Single, llama3Batch, llama3Fast, llama3Char, llama3C1T, llama3C2T, llama3C4T, llama3C8T) =
                    await BenchmarkLlama3(bigLines);
                results.Llama3EncodeSingle = llama3Single;
                results.Llama3EncodeBatch = llama3Batch;
                results.Llama3EncodeFast = llama3Fast;
                results.Llama3EncodeCharOffsets = llama3Char;
                results.Llama3Concurrent1T = llama3C1T;
                results.Llama3Concurrent2T = llama3C2T;
                results.Llama3Concurrent4T = llama3C4T;
                results.Llama3Concurrent8T = llama3C8T;
            }
            else
            {
                Console.WriteLine("⏭ 跳过 Llama3 (资源受限)");
            }

            // BERT Pipeline
            var (bertSingle, bertBatch) = BenchmarkBertPipeline(smallLines);
            results.BertPipelineEncode = bertSingle;
            results.BertPipelineEncodeBatch = bertBatch;

            // 截断扩展测试
            var (truncInput1K, truncInput10K, truncInput100K, truncInput500K,
                 truncMaxLen128, truncMaxLen512, truncMaxLen2048, truncMaxLen8192,
                 truncDirLeft, truncDirRight) = BenchmarkTruncationScaling(bigData, smallLines);
            results.TruncationScalingByInput1K = truncInput1K;
            results.TruncationScalingByInput10K = truncInput10K;
            results.TruncationScalingByInput100K = truncInput100K;
            results.TruncationScalingByInput500K = truncInput500K;
            results.TruncationScalingByMaxLen128 = truncMaxLen128;
            results.TruncationScalingByMaxLen512 = truncMaxLen512;
            results.TruncationScalingByMaxLen2048 = truncMaxLen2048;
            results.TruncationScalingByMaxLen8192 = truncMaxLen8192;
            results.TruncationDirectionLeft = truncDirLeft;
            results.TruncationDirectionRight = truncDirRight;

            // Template Processing
            var (tplSingle, tplPair) = BenchmarkTemplateProcessing(smallLines);
            results.TemplateProcessingSingle = tplSingle;
            results.TemplateProcessingPair = tplPair;

            // Added Vocab 反序列化（400K 在低内存下跳过）
            {
                var (avd100K, avd400K, avd100KNFKC, avd400KNFKC) = BenchmarkAddedVocabDeserialize(skip400K: skip400KVocab);
                results.AddedVocabDeserialize100K = avd100K;
                results.AddedVocabDeserialize400K = avd400K;
                results.AddedVocabDeserialize100KNFKC = avd100KNFKC;
                results.AddedVocabDeserialize400KNFKC = avd400KNFKC;
            }

            // BPE 并发扩展（1/2/8 线程，受 maxConcurrent 限制）
            var (bpeC1T, bpeC2T, bpeC8T) = BenchmarkBpeConcurrentScaling(smallLines, maxConcurrent);
            results.BpeConcurrent1T = bpeC1T;
            results.BpeConcurrent2T = bpeC2T;
            results.BpeConcurrent8T = bpeC8T;

            // 多 tokenizer 序列化加载（需要从 HF 下载，资源受限时跳过）
            if (!skipLargeDownloads)
            {
                var (serialRoberta, serialLlama3, serialAlbert) = await BenchmarkMultiTokenizerSerialization();
                results.SerializationLoadRoberta = serialRoberta;
                results.SerializationLoadLlama3 = serialLlama3;
                results.SerializationLoadAlbert = serialAlbert;
            }
            else
            {
                Console.WriteLine("⏭ 跳过 MultiTokenizerSerialization (资源受限)");
            }

            // === Throughput summary ===
            PrintThroughputSummary(results);
        }

        if (runConsistency)
        {
            Console.WriteLine("\n=== Data Consistency Verification ===");
            await RunConsistencyChecks(smallLines, bigLines);
        }

        if (jsonOutput)
        {
            var json = JsonSerializer.Serialize(results, BenchmarkJsonContext.Default.BenchmarkResults);
            var outPath = Path.Combine(DataDir, $"benchmark-{mode.ToLower()}.json");
            File.WriteAllText(outPath, json);
            Console.WriteLine($"\nJSON results saved to: {outPath}");
        }

        return 0;
    }

    private static string DetectRuntimeMode(string[] args)
    {
        if (!System.Runtime.CompilerServices.RuntimeFeature.IsDynamicCodeSupported)
            return "AOT";
        if (args.Contains("--r2r") || Environment.GetEnvironmentVariable("DOTNET_READY_TO_RUN") == "1")
            return "R2R";
        return "JIT";
    }

    // ──────────────────────────────────────────────
    //  BPE Benchmarks
    // ──────────────────────────────────────────────



    private static BenchResult BenchmarkBpeEncodeSingle(string[] lines)
    {
        Console.WriteLine("--- BPE Encode (single, trained on small.txt) ---");
        var tokenizer = CreateTrainedBpeTokenizer();
        var testLines = lines.Take(1000).ToArray();

        // Warmup
        foreach (var line in testLines) tokenizer.Encode(line);

        const int iterations = 100;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            foreach (var line in testLines)
                tokenizer.Encode(line);
        sw.Stop();

        int total = iterations * testLines.Length;
        var result = new BenchResult
        {
            TotalOps = total,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            AvgMs = sw.Elapsed.TotalMilliseconds / total,
            OpsPerSec = total / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / total
        };
        Console.WriteLine($"  {total:N0} ops: {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Throughput: {result.OpsPerSec:F0} ops/sec, Latency: {result.UsPerOp:F2} µs/op");
        Console.WriteLine();
        return result;
    }

    private static BenchResult BenchmarkBpeEncodeBatch(string[] lines, int batchSize)
    {
        Console.WriteLine($"--- BPE Encode (batch={batchSize}) ---");
        var tokenizer = CreateTrainedBpeTokenizer();
        var batches = new List<string[]>();
        for (int i = 0; i < lines.Length; i += batchSize)
        {
            var remaining = Math.Min(batchSize, lines.Length - i);
            if (remaining > 0) batches.Add(lines[i..(i + remaining)]);
        }

        // Warmup
        foreach (var batch in batches.Take(2))
        {
            tokenizer.EncodeBatch(batch);
        }

        const int iterations = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            foreach (var batch in batches)
            {
                tokenizer.EncodeBatch(batch);
            }
        sw.Stop();

        int totalLines = iterations * lines.Length;
        var result = new BenchResult
        {
            TotalOps = totalLines,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = totalLines / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / totalLines
        };
        Console.WriteLine($"  {totalLines:N0} lines in {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Throughput: {result.OpsPerSec:F0} lines/sec, Latency: {result.UsPerOp:F2} µs/line");
        Console.WriteLine();
        return result;
    }

    private static BenchResult BenchmarkBpeEncodeFast(string[] lines)
    {
        Console.WriteLine("--- BPE EncodeFast vs Encode ---");
        var tokenizer = CreateTrainedBpeTokenizer();
        var testLines = lines.Take(1000).ToArray();

        // Warmup
        foreach (var line in testLines) { tokenizer.Encode(line); tokenizer.EncodeFast(line); }

        const int iterations = 100;

        // Encode
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            foreach (var line in testLines)
                tokenizer.Encode(line);
        sw.Stop();
        var encodeMs = sw.Elapsed.TotalMilliseconds;

        // EncodeFast
        sw.Restart();
        for (int i = 0; i < iterations; i++)
            foreach (var line in testLines)
                tokenizer.EncodeFast(line);
        sw.Stop();
        var encodeFastMs = sw.Elapsed.TotalMilliseconds;

        int total = iterations * testLines.Length;
        Console.WriteLine($"  Encode:     {encodeMs:F1}ms ({total / (encodeMs / 1000):F0} ops/sec)");
        Console.WriteLine($"  EncodeFast: {encodeFastMs:F1}ms ({total / (encodeFastMs / 1000):F0} ops/sec)");
        Console.WriteLine($"  Speedup:    {encodeMs / encodeFastMs:F2}x");
        Console.WriteLine();

        return new BenchResult
        {
            TotalOps = total,
            ElapsedMs = encodeFastMs,
            OpsPerSec = total / (encodeFastMs / 1000),
            UsPerOp = (encodeFastMs * 1000) / total
        };
    }

    // ──────────────────────────────────────────────
    //  Unigram Benchmarks
    // ──────────────────────────────────────────────


    private static BenchResult BenchmarkUnigramEncodeSingle(string[] lines)
    {
        Console.WriteLine("--- Unigram Encode (single) ---");
        var tokenizer = CreateTrainedUnigramTokenizer();
        var testLines = lines.Take(1000).ToArray();

        // Warmup
        foreach (var line in testLines) tokenizer.Encode(line);

        const int iterations = 100;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            foreach (var line in testLines)
                tokenizer.Encode(line);
        sw.Stop();

        int total = iterations * testLines.Length;
        var result = new BenchResult
        {
            TotalOps = total,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = total / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / total
        };
        Console.WriteLine($"  {total:N0} ops: {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Throughput: {result.OpsPerSec:F0} ops/sec, Latency: {result.UsPerOp:F2} µs/op");
        Console.WriteLine();
        return result;
    }

    private static BenchResult BenchmarkUnigramEncodeBatch(string[] lines, int batchSize)
    {
        Console.WriteLine($"--- Unigram Encode (batch={batchSize}) ---");
        var tokenizer = CreateTrainedUnigramTokenizer();
        var batches = new List<string[]>();
        for (int i = 0; i < lines.Length; i += batchSize)
        {
            var remaining = Math.Min(batchSize, lines.Length - i);
            if (remaining > 0) batches.Add(lines[i..(i + remaining)]);
        }

        // Warmup
        foreach (var batch in batches.Take(2))
        {
            tokenizer.EncodeBatch(batch);
        }

        const int iterations = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            foreach (var batch in batches)
            {
                tokenizer.EncodeBatch(batch);
            }
        sw.Stop();

        int totalLines = iterations * lines.Length;
        var result = new BenchResult
        {
            TotalOps = totalLines,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = totalLines / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / totalLines
        };
        Console.WriteLine($"  {totalLines:N0} lines in {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Throughput: {result.OpsPerSec:F0} lines/sec, Latency: {result.UsPerOp:F2} µs/line");
        Console.WriteLine();
        return result;
    }

    // ──────────────────────────────────────────────
    //  WordPiece Benchmarks
    // ──────────────────────────────────────────────


    // ──────────────────────────────────────────────
    //  WordLevel Benchmarks
    // ──────────────────────────────────────────────


    private static BenchResult BenchmarkWordLevelEncodeSingle(string[] lines)
    {
        Console.WriteLine("--- WordLevel Encode (single) ---");
        var tokenizer = CreateTrainedWordLevelTokenizer();
        var testLines = lines.Take(1000).ToArray();

        // Warmup
        foreach (var line in testLines) tokenizer.Encode(line);

        const int iterations = 100;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            foreach (var line in testLines)
                tokenizer.Encode(line);
        sw.Stop();

        int total = iterations * testLines.Length;
        var result = new BenchResult
        {
            TotalOps = total,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = total / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / total
        };
        Console.WriteLine($"  {total:N0} ops: {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Throughput: {result.OpsPerSec:F0} ops/sec, Latency: {result.UsPerOp:F2} µs/op");
        Console.WriteLine();
        return result;
    }

    private static BenchResult BenchmarkWordLevelEncodeBatch(string[] lines, int batchSize)
    {
        Console.WriteLine($"--- WordLevel Encode (batch={batchSize}) ---");
        var tokenizer = CreateTrainedWordLevelTokenizer();
        var batches = new List<string[]>();
        for (int i = 0; i < lines.Length; i += batchSize)
        {
            var remaining = Math.Min(batchSize, lines.Length - i);
            if (remaining > 0) batches.Add(lines[i..(i + remaining)]);
        }

        // Warmup
        foreach (var batch in batches.Take(2))
        {
            tokenizer.EncodeBatch(batch);
        }

        const int iterations = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            foreach (var batch in batches)
            {
                tokenizer.EncodeBatch(batch);
            }
        sw.Stop();

        int totalLines = iterations * lines.Length;
        var result = new BenchResult
        {
            TotalOps = totalLines,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = totalLines / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / totalLines
        };
        Console.WriteLine($"  {totalLines:N0} lines in {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Throughput: {result.OpsPerSec:F0} lines/sec, Latency: {result.UsPerOp:F2} µs/line");
        Console.WriteLine();
        return result;
    }

    // ──────────────────────────────────────────────
    //  Save Trained Model (一致性测试用)
    // ──────────────────────────────────────────────

    /// <summary>
    /// 训练模型并保存 JSON，用于跨平台一致性比对。
    /// 与 Rust cross_lang_train.rs 使用相同参数：
    private static BenchResult BenchmarkWordPieceEncodeSingle(string[] lines)
    {
        Console.WriteLine("--- WordPiece Encode (single) ---");
        var tokenizer = CreateTrainedWordPieceTokenizer();
        var testLines = lines.Take(1000).ToArray();

        // Warmup
        foreach (var line in testLines) tokenizer.Encode(line);

        const int iterations = 100;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            foreach (var line in testLines)
                tokenizer.Encode(line);
        sw.Stop();

        int total = iterations * testLines.Length;
        var result = new BenchResult
        {
            TotalOps = total,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = total / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / total
        };
        Console.WriteLine($"  {total:N0} ops: {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Throughput: {result.OpsPerSec:F0} ops/sec, Latency: {result.UsPerOp:F2} µs/op");
        Console.WriteLine();
        return result;
    }

    private static BenchResult BenchmarkWordPieceEncodeBatch(string[] lines, int batchSize)
    {
        Console.WriteLine($"--- WordPiece Encode (batch={batchSize}) ---");
        var tokenizer = CreateTrainedWordPieceTokenizer();
        var batches = new List<string[]>();
        for (int i = 0; i < lines.Length; i += batchSize)
        {
            var remaining = Math.Min(batchSize, lines.Length - i);
            if (remaining > 0) batches.Add(lines[i..(i + remaining)]);
        }

        // Warmup
        foreach (var batch in batches.Take(2))
        {
            tokenizer.EncodeBatch(batch);
        }

        const int iterations = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            foreach (var batch in batches)
            {
                tokenizer.EncodeBatch(batch);
            }
        sw.Stop();

        int totalLines = iterations * lines.Length;
        var result = new BenchResult
        {
            TotalOps = totalLines,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = totalLines / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / totalLines
        };
        Console.WriteLine($"  {totalLines:N0} lines in {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Throughput: {result.OpsPerSec:F0} lines/sec, Latency: {result.UsPerOp:F2} µs/line");
        Console.WriteLine();
        return result;
    }

    private static BenchResult BenchmarkWordPieceEncodeFast(string[] lines)
    {
        Console.WriteLine("--- WordPiece EncodeFast vs Encode ---");
        var tokenizer = CreateTrainedWordPieceTokenizer();
        var testLines = lines.Take(1000).ToArray();

        // Warmup
        foreach (var line in testLines) { tokenizer.Encode(line); tokenizer.EncodeFast(line); }

        const int iterations = 100;

        // Encode
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            foreach (var line in testLines)
                tokenizer.Encode(line);
        sw.Stop();
        var encodeMs = sw.Elapsed.TotalMilliseconds;

        // EncodeFast
        sw.Restart();
        for (int i = 0; i < iterations; i++)
            foreach (var line in testLines)
                tokenizer.EncodeFast(line);
        sw.Stop();
        var encodeFastMs = sw.Elapsed.TotalMilliseconds;

        int total = iterations * testLines.Length;
        Console.WriteLine($"  Encode:     {encodeMs:F1}ms ({total / (encodeMs / 1000):F0} ops/sec)");
        Console.WriteLine($"  EncodeFast: {encodeFastMs:F1}ms ({total / (encodeFastMs / 1000):F0} ops/sec)");
        Console.WriteLine($"  Speedup:    {encodeMs / encodeFastMs:F2}x");
        Console.WriteLine();

        return new BenchResult
        {
            TotalOps = total,
            ElapsedMs = encodeFastMs,
            OpsPerSec = total / (encodeFastMs / 1000),
            UsPerOp = (encodeFastMs * 1000) / total
        };
    }

    // ──────────────────────────────────────────────
    //  Qwen2.5 (Real-world Tokenizer from JSON)
    // ──────────────────────────────────────────────

    private static async Task<BenchResult> BenchmarkTokenizerFromJson(string modelId, string[] lines)
    {
        Console.WriteLine($"--- {modelId} Encode (single, from JSON) ---");
        // 优先加载 Phase 3 下载的本地文件，避免重复从 HF 下载
        var localQwenPath = Path.Combine(DataDir, "qwen2.5-tokenizer.json");
        Tokenizer tokenizer;
        if (modelId.Contains("Qwen") && File.Exists(localQwenPath))
        {
            tokenizer = TokenizerLoader.FromFile(localQwenPath);
            Console.WriteLine($"  ✅ 从本地加载: {localQwenPath}");
        }
        else
        {
            tokenizer = await TokenizerLoader.FromPretrainedAsync(modelId);
        }
        var testLines = lines.Take(1000).ToArray();

        // Warmup
        foreach (var line in testLines.Take(100)) tokenizer.Encode(line);

        const int iterations = 100;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            foreach (var line in testLines)
                tokenizer.Encode(line);
        sw.Stop();

        int total = iterations * testLines.Length;
        var result = new BenchResult
        {
            TotalOps = total,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = total / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / total
        };
        Console.WriteLine($"  {total:N0} ops: {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Throughput: {result.OpsPerSec:F0} ops/sec, Latency: {result.UsPerOp:F2} µs/op");
        Console.WriteLine();
        return result;
    }

    private static async Task<BenchResult> BenchmarkTokenizerFromJsonBatch(string modelId, string[] lines, int batchSize)
    {
        Console.WriteLine($"--- {modelId} Encode (batch={batchSize}, from JSON) ---");
        // 优先加载 Phase 3 下载的本地文件，避免重复从 HF 下载
        var localQwenPath = Path.Combine(DataDir, "qwen2.5-tokenizer.json");
        Tokenizer tokenizer;
        if (modelId.Contains("Qwen") && File.Exists(localQwenPath))
        {
            tokenizer = TokenizerLoader.FromFile(localQwenPath);
            Console.WriteLine($"  ✅ 从本地加载: {localQwenPath}");
        }
        else
        {
            tokenizer = await TokenizerLoader.FromPretrainedAsync(modelId);
        }

        var batches = new List<string[]>();
        for (int i = 0; i < lines.Length; i += batchSize)
        {
            var remaining = Math.Min(batchSize, lines.Length - i);
            if (remaining > 0) batches.Add(lines[i..(i + remaining)]);
        }

        // Warmup
        foreach (var batch in batches.Take(2))
        {
            tokenizer.EncodeBatch(batch);
        }

        const int iterations = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            foreach (var batch in batches)
            {
                tokenizer.EncodeBatch(batch);
            }
        sw.Stop();

        int totalLines = iterations * lines.Length;
        var result = new BenchResult
        {
            TotalOps = totalLines,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = totalLines / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / totalLines
        };
        Console.WriteLine($"  {totalLines:N0} lines in {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Throughput: {result.OpsPerSec:F0} lines/sec, Latency: {result.UsPerOp:F2} µs/line");
        Console.WriteLine();
        return result;
    }

    // ──────────────────────────────────────────────
    //  Serialization Benchmarks
    // ──────────────────────────────────────────────

    private static BenchResult BenchmarkSerializationLoad()
    {
        Console.WriteLine("--- Serialization: FromJson (in-memory) ---");
        var tokenizer = CreateTrainedBpeTokenizer();
        var json = tokenizer.ToJson();

        // Warmup — 纯内存反序列化，与 Rust Tokenizer::from_str 对齐
        for (int i = 0; i < 3; i++) TokenizerLoader.FromJson(json);

        const int iterations = 100;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            TokenizerLoader.FromJson(json);
        sw.Stop();

        return PrintResult("Serialization Load", "from-json", iterations, sw.Elapsed, 0);
    }

    private static BenchResult BenchmarkSerializationSave()
    {
        Console.WriteLine("--- Serialization: ToJson (in-memory) ---");
        var tokenizer = CreateTrainedBpeTokenizer();

        // Warmup — 纯内存序列化，与 Rust tokenizer.to_string() 对齐
        for (int i = 0; i < 3; i++) tokenizer.ToJson();

        const int iterations = 100;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            tokenizer.ToJson();
        sw.Stop();

        var result = PrintResult("Serialization Save", "to-json", iterations, sw.Elapsed, 0);
        Console.WriteLine($"  JSON size: {tokenizer.ToJson().Length / 1024:N0} KB");
        Console.WriteLine();
        return result;
    }

    private static BenchResult BenchmarkSerializationDeserialize()
    {
        Console.WriteLine("--- Serialization: FromJson ---");
        var tokenizer = CreateTrainedBpeTokenizer();
        var json = tokenizer.ToJson();

        // Warmup
        for (int i = 0; i < 3; i++) TokenizerLoader.FromJson(json);

        const int iterations = 100;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            TokenizerLoader.FromJson(json);
        sw.Stop();

        return PrintResult("Serialization Deserialize", "from-json", iterations, sw.Elapsed, 0);
    }

    // ──────────────────────────────────────────────
    //  Concurrent Benchmarks
    // ──────────────────────────────────────────────

    private static BenchResult BenchmarkBpeConcurrent(string[] lines, int numThreads)
    {
        Console.WriteLine($"--- BPE Concurrent ({numThreads} threads, batch-per-thread) ---");
        var tokenizer = CreateTrainedBpeTokenizer();
        var testLines = lines.Take(1000).ToArray();

        // 与 Rust 对齐：每线程分一块，join 成一个大字符串，每线程 encode 一次
        int linesPerThread = testLines.Length / numThreads;
        var inputs = new string[numThreads];
        for (int t = 0; t < numThreads; t++)
            inputs[t] = string.Join("\n", testLines.AsSpan(t * linesPerThread, linesPerThread).ToArray());

        // Warmup
        Parallel.ForEach(inputs, input => tokenizer.Encode(input));

        // ops = iterations * numThreads（与 Rust 一致：每次并行 encode = 1 op）
        const int iterations = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            Parallel.ForEach(inputs, new ParallelOptions { MaxDegreeOfParallelism = numThreads },
                input => tokenizer.Encode(input));
        }
        sw.Stop();

        int totalOps = iterations * numThreads;
        long totalBytes = inputs.Sum(s => (long)s.Length);
        var result = new BenchResult
        {
            TotalOps = totalOps,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            AvgMs = sw.Elapsed.TotalMilliseconds / totalOps,
            OpsPerSec = totalOps / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / totalOps,
            ThroughputMBps = (totalBytes / (1024.0 * 1024.0)) / (sw.Elapsed.TotalSeconds / iterations)
        };
        Console.WriteLine($"  {totalOps:N0} ops ({numThreads} threads): {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Throughput: {result.OpsPerSec:F0} ops/sec, {result.ThroughputMBps:F1} MiB/s");
        Console.WriteLine();
        return result;
    }

    // ──────────────────────────────────────────────
    //  Truncation Benchmarks
    // ──────────────────────────────────────────────

    private static BenchResult BenchmarkBpeTruncation(string[] lines, int maxLength)
    {
        // 与 Rust 对齐：单个大字符串（50K 字符 from big.txt），重复 encode
        Console.WriteLine($"--- BPE Truncation (maxLength={maxLength}, single 50K input) ---");
        var tokenizer = CreateTrainedBpeTokenizer();
        tokenizer.Truncation = new TruncationParams { MaxLength = maxLength, Strategy = TruncationStrategy.LongestFirst };

        var bigPath = Path.Combine(DataDir, "big.txt");
        var bigData = File.ReadAllText(bigPath);
        var input = bigData.Length >= 50_000 ? bigData[..50_000] : bigData;

        // Warmup
        tokenizer.Encode(input);

        const int iterations = 100;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            tokenizer.Encode(input);
        sw.Stop();

        tokenizer.Truncation = null;

        long totalBytes = input.Length;
        var result = new BenchResult
        {
            TotalOps = iterations,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            AvgMs = sw.Elapsed.TotalMilliseconds / iterations,
            OpsPerSec = iterations / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / iterations,
            ThroughputMBps = (totalBytes / (1024.0 * 1024.0)) / (sw.Elapsed.TotalSeconds / iterations)
        };
        Console.WriteLine($"  {iterations} ops: {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Throughput: {result.OpsPerSec:F0} ops/sec, {result.ThroughputMBps:F1} MiB/s (input={totalBytes / 1024}KB)");
        Console.WriteLine();
        return result;
    }

    // ──────────────────────────────────────────────
    //  EncodeCharOffsets Benchmarks
    // ──────────────────────────────────────────────

    private static BenchResult BenchmarkBpeEncodeCharOffsets(string[] lines)
    {
        Console.WriteLine("--- BPE EncodeCharOffsets (batch) ---");
        var tokenizer = CreateTrainedBpeTokenizer();
        var testLines = lines.Take(1000).ToArray();

        // Warmup — batch 模式，与 Rust encode_batch_char_offsets 对齐
        tokenizer.EncodeBatchCharOffsets(testLines);

        const int iterations = 100;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            tokenizer.EncodeBatchCharOffsets(testLines);
        sw.Stop();

        int total = iterations * testLines.Length;
        var result = new BenchResult
        {
            TotalOps = total,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = total / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / total
        };
        Console.WriteLine($"  {total:N0} ops ({iterations} batches × {testLines.Length} lines): {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Throughput: {result.OpsPerSec:F0} lines/sec, Latency: {result.UsPerOp:F2} µs/line");
        Console.WriteLine();
        return result;
    }

    private static BenchResult BenchmarkUnigramEncodeCharOffsets(string[] lines)
    {
        Console.WriteLine("--- Unigram EncodeCharOffsets (batch) ---");
        var tokenizer = CreateTrainedUnigramTokenizer();
        var testLines = lines.Take(1000).ToArray();

        // Warmup — batch 模式，与 Rust encode_batch_char_offsets 对齐
        tokenizer.EncodeBatchCharOffsets(testLines);

        const int iterations = 100;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            tokenizer.EncodeBatchCharOffsets(testLines);
        sw.Stop();

        int total = iterations * testLines.Length;
        var result = new BenchResult
        {
            TotalOps = total,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = total / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / total
        };
        Console.WriteLine($"  {total:N0} ops ({iterations} batches × {testLines.Length} lines): {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Throughput: {result.OpsPerSec:F0} lines/sec, Latency: {result.UsPerOp:F2} µs/line");
        Console.WriteLine();
        return result;
    }

    // ──────────────────────────────────────────────
    //  BPE Encode No Cache（无缓存 BPE 编码）
    // ──────────────────────────────────────────────

    /// <summary>
    /// 无缓存 BPE 编码测试。C# 绑定没有 cache_capacity(0) 的概念，跳过此测试。
    /// </summary>
    private static (BenchResult? single, BenchResult? batch) BenchmarkBpeEncodeNoCache(string[] lines)
    {
        Console.WriteLine("--- BPE Encode No Cache ---");
        Console.WriteLine("  ⚠️ C# 绑定不支持 cache_capacity(0)，跳过此测试");
        Console.WriteLine();
        return (null, null);
    }

    // ──────────────────────────────────────────────
    //  Llama3 Benchmarks
    // ──────────────────────────────────────────────

    /// <summary>
    /// Llama3 系列基准测试。需要从 HuggingFace 下载 meta-llama/Meta-Llama-3-8B tokenizer。
    /// </summary>
    private static async Task<(BenchResult? encodeSingle, BenchResult? encodeBatch, BenchResult? encodeFast,
        BenchResult? encodeCharOffsets, BenchResult? c1t, BenchResult? c2t, BenchResult? c4t, BenchResult? c8t)>
        BenchmarkLlama3(string[] lines)
    {
        Console.WriteLine("--- Llama3 Tokenizer (meta-llama/Meta-Llama-3-8B) ---");
        Tokenizer tokenizer;
        try
        {
            tokenizer = await TokenizerLoader.FromPretrainedAsync("meta-llama/Meta-Llama-3-8B");
            Console.WriteLine("  ✅ Llama3 tokenizer 加载成功");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  ⚠️ Llama3 tokenizer 加载失败，跳过: {ex.Message}");
            Console.WriteLine();
            return (null, null, null, null, null, null, null, null);
        }

        var testLines = lines.Take(1000).ToArray();

        // === 单条编码 ===
        Console.WriteLine("  --- Llama3 Encode (single) ---");
        foreach (var line in testLines) tokenizer.Encode(line); // warmup
        const int singleIter = 100;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < singleIter; i++)
            foreach (var line in testLines)
                tokenizer.Encode(line);
        sw.Stop();
        int totalSingle = singleIter * testLines.Length;
        var encodeSingle = new BenchResult
        {
            TotalOps = totalSingle,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = totalSingle / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / totalSingle
        };
        Console.WriteLine($"    {totalSingle:N0} ops: {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"    Throughput: {encodeSingle.OpsPerSec:F0} ops/sec, Latency: {encodeSingle.UsPerOp:F2} µs/op");

        // === 批量编码 ===
        Console.WriteLine("  --- Llama3 Encode (batch=1000) ---");
        var batches = new List<string[]>();
        for (int i = 0; i < lines.Length; i += 1000)
        {
            var remaining = Math.Min(1000, lines.Length - i);
            if (remaining > 0) batches.Add(lines[i..(i + remaining)]);
        }
        foreach (var batch in batches.Take(2))
        {
            tokenizer.EncodeBatch(batch);
        } // warmup
        const int batchIter = 20;
        sw.Restart();
        for (int i = 0; i < batchIter; i++)
            foreach (var batch in batches)
            {
                tokenizer.EncodeBatch(batch);
            }
        sw.Stop();
        int totalBatch = batchIter * lines.Length;
        var encodeBatch = new BenchResult
        {
            TotalOps = totalBatch,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = totalBatch / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / totalBatch
        };
        Console.WriteLine($"    {totalBatch:N0} lines: {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"    Throughput: {encodeBatch.OpsPerSec:F0} lines/sec, Latency: {encodeBatch.UsPerOp:F2} µs/line");

        // === EncodeFast ===
        Console.WriteLine("  --- Llama3 EncodeFast ---");
        foreach (var line in testLines) { tokenizer.Encode(line); tokenizer.EncodeFast(line); } // warmup
        sw.Restart();
        for (int i = 0; i < singleIter; i++)
            foreach (var line in testLines)
                tokenizer.EncodeFast(line);
        sw.Stop();
        int totalFast = singleIter * testLines.Length;
        var encodeFast = new BenchResult
        {
            TotalOps = totalFast,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = totalFast / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / totalFast
        };
        Console.WriteLine($"    {totalFast:N0} ops: {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"    Throughput: {encodeFast.OpsPerSec:F0} ops/sec, Latency: {encodeFast.UsPerOp:F2} µs/op");

        // === EncodeCharOffsets ===
        Console.WriteLine("  --- Llama3 EncodeCharOffsets ---");
        foreach (var line in testLines) tokenizer.EncodeCharOffsets(line); // warmup
        sw.Restart();
        for (int i = 0; i < singleIter; i++)
            foreach (var line in testLines)
                tokenizer.EncodeCharOffsets(line);
        sw.Stop();
        int totalChar = singleIter * testLines.Length;
        var encodeCharOffsets = new BenchResult
        {
            TotalOps = totalChar,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = totalChar / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / totalChar
        };
        Console.WriteLine($"    {totalChar:N0} ops: {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"    Throughput: {encodeCharOffsets.OpsPerSec:F0} ops/sec, Latency: {encodeCharOffsets.UsPerOp:F2} µs/op");

        // === 并发编码 1/2/4/8 线程（受 maxConcurrent 限制） ===
        BenchResult? c1t = null, c2t = null, c4t = null, c8t = null;
        int llama3MaxConc = int.TryParse(Environment.GetEnvironmentVariable("BENCH_MAX_CONCURRENT"), out var lmc) ? lmc : 8;
        foreach (var (threads, label) in new[] { (1, "1T"), (2, "2T"), (4, "4T"), (8, "8T") })
        {
            if (threads > llama3MaxConc)
            {
                Console.WriteLine($"  --- Llama3 Concurrent ({label}): ⏭ 跳过 ---");
                continue;
            }
            Console.WriteLine($"  --- Llama3 Concurrent ({label}, batch-per-thread) ---");
            // 与 Rust 对齐：每线程分一块，join 成一个大字符串
            int linesPerThread = testLines.Length / threads;
            var inputs = new string[threads];
            for (int t = 0; t < threads; t++)
                inputs[t] = string.Join("\n", testLines.AsSpan(t * linesPerThread, linesPerThread).ToArray());

            Parallel.ForEach(inputs, input => tokenizer.Encode(input)); // warmup
            const int cIter = 20;
            sw.Restart();
            for (int i = 0; i < cIter; i++)
                Parallel.ForEach(inputs, new ParallelOptions { MaxDegreeOfParallelism = threads },
                    input => tokenizer.Encode(input));
            sw.Stop();
            int totalOps = cIter * threads;
            long totalBytes = inputs.Sum(s => (long)s.Length);
            var cResult = new BenchResult
            {
                TotalOps = totalOps,
                ElapsedMs = sw.Elapsed.TotalMilliseconds,
                OpsPerSec = totalOps / sw.Elapsed.TotalSeconds,
                UsPerOp = sw.Elapsed.TotalMicroseconds / totalOps,
                ThroughputMBps = (totalBytes / (1024.0 * 1024.0)) / (sw.Elapsed.TotalSeconds / cIter)
            };
            Console.WriteLine($"    {totalOps:N0} ops ({label}): {sw.Elapsed.TotalMilliseconds:F1}ms");
            Console.WriteLine($"    Throughput: {cResult.OpsPerSec:F0} ops/sec, {cResult.ThroughputMBps:F1} MiB/s");

            switch (threads)
            {
                case 1: c1t = cResult; break;
                case 2: c2t = cResult; break;
                case 4: c4t = cResult; break;
                case 8: c8t = cResult; break;
            }
        }

        Console.WriteLine();
        return (encodeSingle, encodeBatch, encodeFast, encodeCharOffsets, c1t, c2t, c4t, c8t);
    }

    // ──────────────────────────────────────────────
    //  BERT Pipeline Benchmarks
    // ──────────────────────────────────────────────

    /// <summary>
    /// 完整 BERT pipeline 基准测试。使用自训练 BPE 模型转 WordPiece，配置 BertNormalizer + BertPreTokenizer + BertProcessing + WordPieceDecoder。
    /// </summary>
    private static (BenchResult? single, BenchResult? batch) BenchmarkBertPipeline(string[] lines)
    {
        Console.WriteLine("--- BERT Pipeline Encode ---");
        var tokenizer = CreateTrainedWordPieceTokenizer();
        // 配置完整 BERT pipeline
        tokenizer.Normalizer = new BertNormalizer(lowercase: true);
        tokenizer.PreTokenizer = new BertPreTokenizer();
        tokenizer.PostProcessor = new BertProcessing(
            sep: ("[SEP]", 99),
            cls: ("[CLS]", 98));
        tokenizer.Decoder = new WordPieceDecoder(prefix: "##");

        var testLines = lines.Take(1000).ToArray();

        // 单条编码
        Console.WriteLine("  --- BERT Pipeline Encode (single) ---");
        foreach (var line in testLines) tokenizer.Encode(line); // warmup
        const int singleIter = 100;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < singleIter; i++)
            foreach (var line in testLines)
                tokenizer.Encode(line);
        sw.Stop();
        int totalSingle = singleIter * testLines.Length;
        var encodeSingle = new BenchResult
        {
            TotalOps = totalSingle,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = totalSingle / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / totalSingle
        };
        Console.WriteLine($"    {totalSingle:N0} ops: {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"    Throughput: {encodeSingle.OpsPerSec:F0} ops/sec, Latency: {encodeSingle.UsPerOp:F2} µs/op");

        // 批量编码
        Console.WriteLine("  --- BERT Pipeline Encode (batch=1000) ---");
        var batches = new List<string[]>();
        for (int i = 0; i < lines.Length; i += 1000)
        {
            var remaining = Math.Min(1000, lines.Length - i);
            if (remaining > 0) batches.Add(lines[i..(i + remaining)]);
        }
        foreach (var batch in batches.Take(2))
        {
            tokenizer.EncodeBatch(batch);
        } // warmup
        const int batchIter = 20;
        sw.Restart();
        for (int i = 0; i < batchIter; i++)
            foreach (var batch in batches)
            {
                tokenizer.EncodeBatch(batch);
            }
        sw.Stop();
        int totalBatch = batchIter * lines.Length;
        var encodeBatch = new BenchResult
        {
            TotalOps = totalBatch,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = totalBatch / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / totalBatch
        };
        Console.WriteLine($"    {totalBatch:N0} lines: {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"    Throughput: {encodeBatch.OpsPerSec:F0} lines/sec, Latency: {encodeBatch.UsPerOp:F2} µs/line");
        Console.WriteLine();

        return (encodeSingle, encodeBatch);
    }

    // ──────────────────────────────────────────────
    //  Truncation Scaling Benchmarks
    // ──────────────────────────────────────────────

    /// <summary>
    /// 截断性能按输入规模和 max_length 的扩展测试。
    /// </summary>
    private static (BenchResult? byInput1K, BenchResult? byInput10K, BenchResult? byInput100K, BenchResult? byInput500K,
        BenchResult? byMaxLen128, BenchResult? byMaxLen512, BenchResult? byMaxLen2048, BenchResult? byMaxLen8192,
        BenchResult? dirLeft, BenchResult? dirRight) BenchmarkTruncationScaling(string bigData, string[] lines)
    {
        Console.WriteLine("--- Truncation Scaling ---");
        var tokenizer = CreateTrainedBpeTokenizer();

        BenchResult MakeResult(int iterations, Stopwatch sw) => new()
        {
            TotalOps = iterations,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = iterations / (sw.Elapsed.TotalMilliseconds / 1000.0),
            UsPerOp = sw.Elapsed.TotalMicroseconds / iterations
        };

        // === 按输入规模：maxLength=512，不同输入长度 ===
        var inputSizes = new[] { 1_000, 10_000, 100_000, 500_000 };
        BenchResult?[] byInput = new BenchResult?[4];
        for (int idx = 0; idx < inputSizes.Length; idx++)
        {
            var size = inputSizes[idx];
            var text = bigData.Length >= size ? bigData[..size] : bigData;
            tokenizer.Truncation = new TruncationParams { MaxLength = 512, Strategy = TruncationStrategy.LongestFirst };
            Console.WriteLine($"  --- Truncation by Input Size ({size / 1000}K chars, maxLength=512) ---");
            for (int w = 0; w < 3; w++) tokenizer.Encode(text); // warmup
            const int iterations = 20;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
                tokenizer.Encode(text);
            sw.Stop();
            tokenizer.Truncation = null;
            byInput[idx] = MakeResult(iterations, sw);
            Console.WriteLine($"    {iterations} ops: {sw.Elapsed.TotalMilliseconds:F1}ms, {byInput[idx]!.OpsPerSec:F0} ops/sec, {byInput[idx]!.UsPerOp:F2} µs/op");
        }

        // === 按 max_length：输入 50K 字符，不同 maxLength ===
        var maxLens = new[] { 128, 512, 2048, 8192 };
        BenchResult?[] byMaxLen = new BenchResult?[4];
        var text50K = bigData.Length >= 50_000 ? bigData[..50_000] : bigData;
        for (int idx = 0; idx < maxLens.Length; idx++)
        {
            var ml = maxLens[idx];
            tokenizer.Truncation = new TruncationParams { MaxLength = ml, Strategy = TruncationStrategy.LongestFirst };
            Console.WriteLine($"  --- Truncation by MaxLen (50K input, maxLength={ml}) ---");
            for (int w = 0; w < 3; w++) tokenizer.Encode(text50K); // warmup
            const int iterations = 20;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
                tokenizer.Encode(text50K);
            sw.Stop();
            tokenizer.Truncation = null;
            byMaxLen[idx] = MakeResult(iterations, sw);
            Console.WriteLine($"    {iterations} ops: {sw.Elapsed.TotalMilliseconds:F1}ms, {byMaxLen[idx]!.OpsPerSec:F0} ops/sec, {byMaxLen[idx]!.UsPerOp:F2} µs/op");
        }

        // === 按方向：Left vs Right ===
        var text10K = bigData.Length >= 10_000 ? bigData[..10_000] : bigData;
        BenchResult? dirLeft = null, dirRight = null;
        foreach (var direction in new[] { TruncationDirection.Left, TruncationDirection.Right })
        {
            tokenizer.Truncation = new TruncationParams { MaxLength = 512, Strategy = TruncationStrategy.LongestFirst, Direction = direction };
            Console.WriteLine($"  --- Truncation Direction ({direction}, 10K input, maxLength=512) ---");
            for (int w = 0; w < 3; w++) tokenizer.Encode(text10K); // warmup
            const int iterations = 20;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
                tokenizer.Encode(text10K);
            sw.Stop();
            tokenizer.Truncation = null;
            var result = MakeResult(iterations, sw);
            Console.WriteLine($"    {iterations} ops: {sw.Elapsed.TotalMilliseconds:F1}ms, {result.OpsPerSec:F0} ops/sec, {result.UsPerOp:F2} µs/op");
            if (direction == TruncationDirection.Left) dirLeft = result; else dirRight = result;
        }
        Console.WriteLine();

        return (byInput[0], byInput[1], byInput[2], byInput[3],
                byMaxLen[0], byMaxLen[1], byMaxLen[2], byMaxLen[3],
                dirLeft, dirRight);
    }

    // ──────────────────────────────────────────────
    //  Template Processing Benchmarks
    // ──────────────────────────────────────────────

    /// <summary>
    /// 后处理器 TemplateProcessing 性能测试。
    /// </summary>
    private static (BenchResult? single, BenchResult? pair) BenchmarkTemplateProcessing(string[] lines)
    {
        Console.WriteLine("--- Template Processing ---");
        var tokenizer = CreateTrainedBpeTokenizer();
        var testLines = lines.Take(100).ToArray();

        // 创建 single template: [CLS]:0 $A:0 [SEP]:0
        // 创建 pair template:   [CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1
        var singleProcessor = new TemplateProcessing(
            singleTemplate: [Template.Special(98, "[CLS]", 0), Template.A(0), Template.Special(99, "[SEP]", 0)],
            pairTemplate: [Template.Special(98, "[CLS]", 0), Template.A(0), Template.Special(99, "[SEP]", 0), Template.B(1), Template.Special(99, "[SEP]", 1)]);

        // 预编码得到 Encoding 对象
        var encodings = testLines.Select(l => tokenizer.Encode(l)).ToArray();

        // === Single ===
        Console.WriteLine("  --- Template Processing (single) ---");
        foreach (var enc in encodings.Take(10)) singleProcessor.Process(enc, null, true); // warmup
        const int singleIter = 100;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < singleIter; i++)
            foreach (var enc in encodings)
                singleProcessor.Process(enc, null, true);
        sw.Stop();
        int totalSingle = singleIter * encodings.Length;
        var singleResult = new BenchResult
        {
            TotalOps = totalSingle,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = totalSingle / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / totalSingle
        };
        Console.WriteLine($"    {totalSingle:N0} ops: {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"    Throughput: {singleResult.OpsPerSec:F0} ops/sec, Latency: {singleResult.UsPerOp:F2} µs/op");

        // === Pair ===
        Console.WriteLine("  --- Template Processing (pair) ---");
        foreach (var enc in encodings.Take(10)) singleProcessor.Process(enc, enc, true); // warmup
        const int pairIter = 100;
        sw.Restart();
        for (int i = 0; i < pairIter; i++)
            foreach (var enc in encodings)
                singleProcessor.Process(enc, enc, true);
        sw.Stop();
        int totalPair = pairIter * encodings.Length;
        var pairResult = new BenchResult
        {
            TotalOps = totalPair,
            ElapsedMs = sw.Elapsed.TotalMilliseconds,
            OpsPerSec = totalPair / sw.Elapsed.TotalSeconds,
            UsPerOp = sw.Elapsed.TotalMicroseconds / totalPair
        };
        Console.WriteLine($"    {totalPair:N0} ops: {sw.Elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"    Throughput: {pairResult.OpsPerSec:F0} ops/sec, Latency: {pairResult.UsPerOp:F2} µs/op");
        Console.WriteLine();

        return (singleResult, pairResult);
    }

    // ──────────────────────────────────────────────
    //  Added Vocab Deserialize Benchmarks
    // ──────────────────────────────────────────────

    /// <summary>
    /// 大词表反序列化性能测试。分别测试 100K/400K 个 AddedToken，以及有无 NFKC normalizer。
    /// </summary>
    private static (BenchResult? r100K, BenchResult? r400K, BenchResult? r100KNFKC, BenchResult? r400KNFKC)
        BenchmarkAddedVocabDeserialize(bool skip400K = false)
    {
        Console.WriteLine("--- Added Vocab Deserialize ---");
        var benchResults = new List<(int count, bool nfkc, string label)>
        {
            (100_000, false, "100K"),
            (100_000, true, "100K+NFKC"),
        };
        if (!skip400K)
        {
            benchResults.Add((400_000, false, "400K"));
            benchResults.Add((400_000, true, "400K+NFKC"));
        }

        BenchResult? r100K = null, r400K = null, r100KNFKC = null, r400KNFKC = null;

        foreach (var (count, nfkc, label) in benchResults)
        {
            Console.WriteLine($"  --- Added Vocab Deserialize ({label}, special=true) ---");
            try
            {
                var tokenizer = CreateTrainedBpeTokenizer();
                // 添加大量 AddedToken
                for (int i = 0; i < count; i++)
                    tokenizer.AddToken(new AddedToken($"tok{i}", isSpecial: true));

                if (nfkc)
                    tokenizer.Normalizer = new NfkcNormalizer();

                // 保存到临时文件
                var tmpPath = Path.Combine(Path.GetTempPath(), $"bench-added-vocab-{count}-{nfkc}.json");
                tokenizer.Save(tmpPath);

                // warmup
                for (int w = 0; w < 3; w++) TokenizerLoader.FromFile(tmpPath);

                const int iterations = 10;
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < iterations; i++)
                    TokenizerLoader.FromFile(tmpPath);
                sw.Stop();

                File.Delete(tmpPath);

                var result = new BenchResult
                {
                    TotalOps = iterations,
                    ElapsedMs = sw.Elapsed.TotalMilliseconds,
                    AvgMs = sw.Elapsed.TotalMilliseconds / iterations,
                    OpsPerSec = iterations / (sw.Elapsed.TotalMilliseconds / 1000.0),
                    UsPerOp = sw.Elapsed.TotalMicroseconds / iterations
                };
                Console.WriteLine($"    {iterations} runs: {sw.Elapsed.TotalMilliseconds:F1}ms, Avg: {result.AvgMs:F1}ms/run");

                switch ((count, nfkc))
                {
                    case (100_000, false): r100K = result; break;
                    case (400_000, false): r400K = result; break;
                    case (100_000, true): r100KNFKC = result; break;
                    case (400_000, true): r400KNFKC = result; break;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"    ⚠️ 测试失败，跳过: {ex.Message}");
            }
        }
        Console.WriteLine();
        return (r100K, r400K, r100KNFKC, r400KNFKC);
    }

    // ──────────────────────────────────────────────
    //  BPE Concurrent Scaling Benchmarks
    // ──────────────────────────────────────────────

    /// <summary>
    /// BPE 并发扩展性测试（1/2/8 线程，4 线程已有）。
    /// </summary>
    private static (BenchResult? c1t, BenchResult? c2t, BenchResult? c8t) BenchmarkBpeConcurrentScaling(string[] lines, int maxConcurrent = 8)
    {
        Console.WriteLine("--- BPE Concurrent Scaling ---");
        var tokenizer = CreateTrainedBpeTokenizer();
        var testLines = lines.Take(1000).ToArray();

        BenchResult? c1t = null, c2t = null, c8t = null;
        foreach (var (threads, label) in new[] { (1, "1T"), (2, "2T"), (8, "8T") })
        {
            if (threads > maxConcurrent)
            {
                Console.WriteLine($"  --- BPE Concurrent ({label}): ⏭ 跳过 (maxConcurrent={maxConcurrent}) ---");
                continue;
            }
            Console.WriteLine($"  --- BPE Concurrent ({label}, batch-per-thread) ---");

            // 与 Rust 对齐：每线程分一块，join 成一个大字符串
            int linesPerThread = testLines.Length / threads;
            var inputs = new string[threads];
            for (int t = 0; t < threads; t++)
                inputs[t] = string.Join("\n", testLines.AsSpan(t * linesPerThread, linesPerThread).ToArray());

            Parallel.ForEach(inputs, input => tokenizer.Encode(input)); // warmup
            const int iterations = 20;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
                Parallel.ForEach(inputs, new ParallelOptions { MaxDegreeOfParallelism = threads },
                    input => tokenizer.Encode(input));
            sw.Stop();

            int totalOps = iterations * threads;
            long totalBytes = inputs.Sum(s => (long)s.Length);
            var result = new BenchResult
            {
                TotalOps = totalOps,
                ElapsedMs = sw.Elapsed.TotalMilliseconds,
                AvgMs = sw.Elapsed.TotalMilliseconds / totalOps,
                OpsPerSec = totalOps / sw.Elapsed.TotalSeconds,
                UsPerOp = sw.Elapsed.TotalMicroseconds / totalOps,
                ThroughputMBps = (totalBytes / (1024.0 * 1024.0)) / (sw.Elapsed.TotalSeconds / iterations)
            };
            Console.WriteLine($"    {totalOps:N0} ops ({label}): {sw.Elapsed.TotalMilliseconds:F1}ms");
            Console.WriteLine($"    Throughput: {result.OpsPerSec:F0} ops/sec, {result.ThroughputMBps:F1} MiB/s");

            switch (threads)
            {
                case 1: c1t = result; break;
                case 2: c2t = result; break;
                case 8: c8t = result; break;
            }
        }
        Console.WriteLine();
        return (c1t, c2t, c8t);
    }

    // ──────────────────────────────────────────────
    //  Multi-Tokenizer Serialization Benchmarks
    // ──────────────────────────────────────────────

    /// <summary>
    /// 多 tokenizer 序列化加载性能测试。尝试下载 roberta-base, llama3, albert-base-v1。
    /// </summary>
    private static async Task<(BenchResult? roberta, BenchResult? llama3, BenchResult? albert)>
        BenchmarkMultiTokenizerSerialization()
    {
        Console.WriteLine("--- Multi-Tokenizer Serialization Load ---");
        BenchResult? roberta = null, llama3 = null, albert = null;

        var targets = new (string modelId, string label)[]
        {
            ("roberta-base", "Roberta"),
            ("meta-llama/Meta-Llama-3-8B", "Llama3"),
            ("albert/albert-base-v1", "Albert"),
        };

        foreach (var (modelId, label) in targets)
        {
            Console.WriteLine($"  --- Serialization Load ({label}: {modelId}) ---");
            try
            {
                var tokenizer = await TokenizerLoader.FromPretrainedAsync(modelId);
                var tmpPath = Path.Combine(Path.GetTempPath(), $"bench-serial-{label}.json");
                tokenizer.Save(tmpPath);

                // warmup
                for (int w = 0; w < 3; w++) TokenizerLoader.FromFile(tmpPath);

                const int iterations = 50;
                var sw = Stopwatch.StartNew();
                for (int i = 0; i < iterations; i++)
                    TokenizerLoader.FromFile(tmpPath);
                sw.Stop();

                File.Delete(tmpPath);

                var result = new BenchResult
                {
                    TotalOps = iterations,
                    ElapsedMs = sw.Elapsed.TotalMilliseconds,
                    AvgMs = sw.Elapsed.TotalMilliseconds / iterations,
                    OpsPerSec = iterations / (sw.Elapsed.TotalMilliseconds / 1000.0),
                    UsPerOp = sw.Elapsed.TotalMicroseconds / iterations
                };
                Console.WriteLine($"    {iterations} runs: {sw.Elapsed.TotalMilliseconds:F1}ms, Avg: {result.AvgMs:F1}ms/run");

                switch (label)
                {
                    case "Roberta": roberta = result; break;
                    case "Llama3": llama3 = result; break;
                    case "Albert": albert = result; break;
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"    ⚠️ {label} 加载失败，跳过: {ex.Message}");
            }
        }
        Console.WriteLine();
        return (roberta, llama3, albert);
    }

    // ──────────────────────────────────────────────
    //  Data Consistency
    // ──────────────────────────────────────────────

    private static async Task RunConsistencyChecks(string[] smallLines, string[] bigLines)
    {
        // 一致性验证使用全部数据（small + big），确保跨端编码完全一致
        var testLines = smallLines.Concat(bigLines).ToArray();
        Console.WriteLine($"  一致性数据: small={smallLines.Length} + big={bigLines.Length} = {testLines.Length} 行");

        // ── BPE 一致性（加载 Rust 训练的 tokenizer） ──
        Console.WriteLine("--- BPE Consistency ---");
        var bpeTokenizer = CreateTrainedBpeTokenizer();
        SaveConsistencyData("bpe", testLines, bpeTokenizer);

        // ── Unigram 一致性（加载 Rust 训练的 tokenizer） ──
        Console.WriteLine("\n--- Unigram Consistency ---");
        var unigramTokenizer = CreateTrainedUnigramTokenizer();
        SaveConsistencyData("unigram", testLines, unigramTokenizer);

        // ── WordPiece 一致性（加载 Rust 训练的 tokenizer） ──
        Console.WriteLine("\n--- WordPiece Consistency ---");
        var wpTokenizer = CreateTrainedWordPieceTokenizer();
        SaveConsistencyData("wordpiece", testLines, wpTokenizer);

        // ── WordLevel 一致性（加载 Rust 训练的 tokenizer） ──
        Console.WriteLine("\n--- WordLevel Consistency ---");
        var wordLevelTokenizer = CreateTrainedWordLevelTokenizer();
        SaveConsistencyData("wordlevel", testLines, wordLevelTokenizer);

        // ── Qwen2.5 一致性（优先加载本地文件） ──
        Console.WriteLine("\n--- Qwen2.5 Consistency ---");
        var localQwenPath = Path.Combine(DataDir, "qwen2.5-tokenizer.json");
        Tokenizer qwenTokenizer;
        if (File.Exists(localQwenPath))
        {
            qwenTokenizer = TokenizerLoader.FromFile(localQwenPath);
            Console.WriteLine($"  ✅ 从本地加载: {localQwenPath}");
        }
        else
        {
            qwenTokenizer = await TokenizerLoader.FromPretrainedAsync("Qwen/Qwen2.5-7B-Instruct");
        }
        SaveConsistencyData("qwen2.5", testLines, qwenTokenizer);

        // ── 内部一致性检查 ──
        Console.WriteLine("\n--- Encode vs EncodeFast ID Consistency (BPE) ---");
        int fastOk = 0, fastFail = 0;
        foreach (var line in testLines)
        {
            var full = bpeTokenizer.Encode(line);
            var fast = bpeTokenizer.EncodeFast(line);
            var fullIds = full.GetIds();
            if (fullIds.Length == fast.Length && fullIds.SequenceEqual(fast))
                fastOk++;
            else
                fastFail++;
        }
        Console.WriteLine($"  EncodeFast consistency: {fastOk} OK, {fastFail} FAIL");

        Console.WriteLine("\n--- Batch vs Single Encode Consistency (BPE) ---");
        int batchOk = 0, batchFail = 0;
        var batchResult = bpeTokenizer.EncodeBatch(testLines);
        for (int i = 0; i < testLines.Length; i++)
        {
            var single = bpeTokenizer.Encode(testLines[i]);
            var singleIds = single.GetIds();
            var batchIds = batchResult[i].GetIds();
            if (singleIds.Length == batchIds.Length && singleIds.SequenceEqual(batchIds))
                batchOk++;
            else
                batchFail++;
        }
        Console.WriteLine($"  Batch consistency: {batchOk} OK, {batchFail} FAIL");

        Console.WriteLine("\n--- Cross-Tokenizer Consistency (Rust-trained) ---");
        // 对比 Rust 训练的模型和 C# 加载同一模型的编码结果
        var rustBpePath = Path.Combine(DataDir, "tokenizer-rust-bpe.json");
        if (File.Exists(rustBpePath))
        {
            var rustConsistencyPath = Path.Combine(DataDir, "consistency-rust-bpe.json");
            if (File.Exists(rustConsistencyPath))
            {
                var dotnetEntries = BuildConsistencyEntries(testLines, bpeTokenizer);
                var (crossMatch, crossTotal) = CompareConsistencyEntries(dotnetEntries, rustConsistencyPath);
                Console.WriteLine($"  BPE Rust↔C# consistency: {crossMatch}/{crossTotal}");
            }
        }

        Console.WriteLine("\n--- Multi-Run Determinism (Rust-trained BPE) ---");
        // 验证加载同一 Rust 模型多次，编码结果是否确定
        var run1Ids = new List<uint[]>();
        var run2Ids = new List<uint[]>();
        var bpeTokenizer2 = CreateTrainedBpeTokenizer();
        foreach (var line in testLines)
        {
            run1Ids.Add(bpeTokenizer.Encode(line).GetIds());
            run2Ids.Add(bpeTokenizer2.Encode(line).GetIds());
        }
        bool deterministic = true;
        for (int i = 0; i < testLines.Length; i++)
        {
            if (!run1Ids[i].SequenceEqual(run2Ids[i]))
            {
                deterministic = false;
                break;
            }
        }
        Console.WriteLine($"  Deterministic: {(deterministic ? "YES ✓" : "NO ✗")}");
    }

    private static List<ConsistencyEntry> BuildConsistencyEntries(string[] lines, Tokenizer tokenizer)
    {
        var entries = new List<ConsistencyEntry>();
        foreach (var line in lines)
        {
            var encoded = tokenizer.Encode(line);
            entries.Add(new ConsistencyEntry
            {
                Input = line,
                Ids = encoded.GetIds(),
                Tokens = encoded.GetTokens(),
                Decoded = tokenizer.Decode(encoded.GetIds(), false)
            });
        }
        return entries;
    }

    private static (int match, int total) CompareConsistencyEntries(List<ConsistencyEntry> dotnetEntries, string rustJsonPath)
    {
        var rustJson = File.ReadAllText(rustJsonPath);
        var rustEntries = JsonSerializer.Deserialize(rustJson, BenchmarkJsonContext.Default.ListConsistencyEntry);
        if (rustEntries is null || rustEntries.Count != dotnetEntries.Count)
            return (0, Math.Max(rustEntries?.Count ?? 0, dotnetEntries.Count));

        int match = 0;
        for (int i = 0; i < dotnetEntries.Count; i++)
        {
            if (dotnetEntries[i].Input == rustEntries[i].Input &&
                dotnetEntries[i].Decoded == rustEntries[i].Decoded &&
                dotnetEntries[i].Tokens.SequenceEqual(rustEntries[i].Tokens))
                match++;
        }
        return (match, dotnetEntries.Count);
    }

    private static void SaveConsistencyData(string tag, string[] lines, Tokenizer tokenizer)
    {
        Console.WriteLine($"  Saving {tag}...");
        var entries = new List<ConsistencyEntry>();
        foreach (var line in lines)
        {
            var encoded = tokenizer.Encode(line);
            var ids = encoded.GetIds();
            var decoded = tokenizer.Decode(ids, false);
            entries.Add(new ConsistencyEntry
            {
                Input = line,
                Ids = ids,
                Tokens = encoded.GetTokens(),
                Decoded = decoded
            });
        }

        var json = JsonSerializer.Serialize(entries, BenchmarkJsonContext.Default.ListConsistencyEntry);
        var outPath = Path.Combine(DataDir, $"consistency-dotnet-{tag}.json");
        File.WriteAllText(outPath, json);
        Console.WriteLine($"  ✅ {tag}: {entries.Count} entries → consistency-dotnet-{tag}.json");
    }

    // ──────────────────────────────────────────────
    //  Helpers
    // ──────────────────────────────────────────────

    /// <summary>
    /// 使用 Whitespace 预分词器拆分文本（正则 \w+|[^\w\s]+），与 Rust Whitespace 预分词器一致。
    /// </summary>
    private static IReadOnlyList<string> WhitespaceSplit(string text)
    {
        var pts = new PreTokenizedString(text);
        WhitespacePreTokenizer.Instance.PreTokenize(pts);
        return pts.GetSplits().Select(s => s.Normalized.Get()).ToList();
    }

    private static Tokenizer CreateTrainedBpeTokenizer()
    {
        var jsonPath = Path.Combine(DataDir, "tokenizer-rust-bpe.json");
        if (File.Exists(jsonPath))
            return TokenizerLoader.FromFile(jsonPath);

        throw new FileNotFoundException(
            "Rust 训练的 BPE 模型未找到。请先运行 Phase 2（Rust 训练）生成 tokenizer-rust-bpe.json",
            jsonPath);
    }

    private static Tokenizer CreateTrainedWordPieceTokenizer()
    {
        var jsonPath = Path.Combine(DataDir, "tokenizer-rust-wordpiece.json");
        if (File.Exists(jsonPath))
            return TokenizerLoader.FromFile(jsonPath);

        throw new FileNotFoundException(
            "Rust 训练的 WordPiece 模型未找到。请先运行 Phase 2（Rust 训练）生成 tokenizer-rust-wordpiece.json",
            jsonPath);
    }

    private static Tokenizer CreateTrainedUnigramTokenizer()
    {
        var jsonPath = Path.Combine(DataDir, "tokenizer-rust-unigram.json");
        if (File.Exists(jsonPath))
            return TokenizerLoader.FromFile(jsonPath);

        throw new FileNotFoundException(
            "Rust 训练的 Unigram 模型未找到。请先运行 Phase 2（Rust 训练）生成 tokenizer-rust-unigram.json",
            jsonPath);
    }

    private static Tokenizer CreateTrainedWordLevelTokenizer()
    {
        var jsonPath = Path.Combine(DataDir, "tokenizer-rust-wordlevel.json");
        if (File.Exists(jsonPath))
            return TokenizerLoader.FromFile(jsonPath);

        throw new FileNotFoundException(
            "Rust 训练的 WordLevel 模型未找到。请先运行 Phase 2（Rust 训练）生成 tokenizer-rust-wordlevel.json",
            jsonPath);
    }

    private static BenchResult PrintResult(string name, string label, int iterations, TimeSpan elapsed, int dataSize)
    {
        double avgMs = elapsed.TotalMilliseconds / iterations;
        double throughputMBps = dataSize > 0 ? (dataSize / (1024.0 * 1024.0)) / (avgMs / 1000.0) : 0;
        double opsPerSec = iterations / (elapsed.TotalMilliseconds / 1000.0);
        Console.WriteLine($"  {iterations} runs: {elapsed.TotalMilliseconds:F1}ms");
        Console.WriteLine($"  Avg: {avgMs:F1}ms/run, {throughputMBps:F1} MiB/s");
        Console.WriteLine();
        return new BenchResult
        {
            TotalOps = iterations,
            ElapsedMs = elapsed.TotalMilliseconds,
            AvgMs = avgMs,
            OpsPerSec = opsPerSec,
            UsPerOp = elapsed.TotalMicroseconds / iterations,
            ThroughputMBps = throughputMBps
        };
    }

    private static void PrintThroughputSummary(BenchmarkResults r)
    {
        Console.WriteLine("\n=== Throughput Summary ===");
        Console.WriteLine($"{"Benchmark",-35} {"Ops/sec",12} {"µs/op",10} {"MiB/s",10}");
        Console.WriteLine(new string('-', 67));

        PrintRow("BPE Encode (single)", r.BpeEncodeSingle);
        PrintRow("BPE Encode (batch)", r.BpeEncodeBatch);
        PrintRow("BPE EncodeFast", r.BpeEncodeFast);
        PrintRow("Unigram Encode (single)", r.UnigramEncodeSingle);
        PrintRow("Unigram Encode (batch)", r.UnigramEncodeBatch);
        PrintRow("WordPiece Encode (single)", r.WordPieceEncodeSingle);
        PrintRow("WordPiece Encode (batch)", r.WordPieceEncodeBatch);
        PrintRow("WordPiece EncodeFast", r.WordPieceEncodeFast);
        PrintRow("WordLevel Encode (single)", r.WordLevelEncodeSingle);
        PrintRow("WordLevel Encode (batch)", r.WordLevelEncodeBatch);
        PrintRow("Qwen2.5 Encode (single)", r.Qwen25EncodeSingle);
        PrintRow("Qwen2.5 Encode (batch)", r.Qwen25EncodeBatch);
        PrintRow("Serialization Load", r.SerializationLoad);
        PrintRow("Serialization Save", r.SerializationSave);
        PrintRow("Serialization Deserialize", r.SerializationDeserialize);
        PrintRow("BPE Concurrent (4T)", r.BpeConcurrent4T);
        PrintRow("BPE Truncation (512)", r.BpeTruncation512);
        PrintRow("BPE Truncation (128)", r.BpeTruncation128);
        PrintRow("BPE EncodeCharOffsets", r.BpeEncodeCharOffsets);
        PrintRow("Unigram EncodeCharOffsets", r.UnigramEncodeCharOffsets);

        // === 新增：补齐 Rust 版本覆盖但 C# 缺失的基准测试 ===
        PrintRow("Llama3 Encode (single)", r.Llama3EncodeSingle);
        PrintRow("Llama3 Encode (batch)", r.Llama3EncodeBatch);
        PrintRow("Llama3 EncodeFast", r.Llama3EncodeFast);
        PrintRow("Llama3 EncodeCharOffsets", r.Llama3EncodeCharOffsets);
        PrintRow("Llama3 Concurrent (1T)", r.Llama3Concurrent1T);
        PrintRow("Llama3 Concurrent (2T)", r.Llama3Concurrent2T);
        PrintRow("Llama3 Concurrent (4T)", r.Llama3Concurrent4T);
        PrintRow("Llama3 Concurrent (8T)", r.Llama3Concurrent8T);
        PrintRow("BERT Pipeline Encode", r.BertPipelineEncode);
        PrintRow("BERT Pipeline Encode (batch)", r.BertPipelineEncodeBatch);
        PrintRow("Truncation by Input (1K)", r.TruncationScalingByInput1K);
        PrintRow("Truncation by Input (10K)", r.TruncationScalingByInput10K);
        PrintRow("Truncation by Input (100K)", r.TruncationScalingByInput100K);
        PrintRow("Truncation by Input (500K)", r.TruncationScalingByInput500K);
        PrintRow("Truncation by MaxLen (128)", r.TruncationScalingByMaxLen128);
        PrintRow("Truncation by MaxLen (512)", r.TruncationScalingByMaxLen512);
        PrintRow("Truncation by MaxLen (2048)", r.TruncationScalingByMaxLen2048);
        PrintRow("Truncation by MaxLen (8192)", r.TruncationScalingByMaxLen8192);
        PrintRow("Truncation Direction (Left)", r.TruncationDirectionLeft);
        PrintRow("Truncation Direction (Right)", r.TruncationDirectionRight);
        PrintRow("Template Processing (single)", r.TemplateProcessingSingle);
        PrintRow("Template Processing (pair)", r.TemplateProcessingPair);
        PrintRow("Added Vocab Deserialize (100K)", r.AddedVocabDeserialize100K);
        PrintRow("Added Vocab Deserialize (400K)", r.AddedVocabDeserialize400K);
        PrintRow("Added Vocab Deserialize (100K+NFKC)", r.AddedVocabDeserialize100KNFKC);
        PrintRow("Added Vocab Deserialize (400K+NFKC)", r.AddedVocabDeserialize400KNFKC);
        PrintRow("BPE Concurrent (1T)", r.BpeConcurrent1T);
        PrintRow("BPE Concurrent (2T)", r.BpeConcurrent2T);
        PrintRow("BPE Concurrent (8T)", r.BpeConcurrent8T);
        PrintRow("Serialization Load (Roberta)", r.SerializationLoadRoberta);
        PrintRow("Serialization Load (Llama3)", r.SerializationLoadLlama3);
        PrintRow("Serialization Load (Albert)", r.SerializationLoadAlbert);
    }

    private static void PrintRow(string name, BenchResult? result)
    {
        if (result is null) return;
        string ops = result.OpsPerSec > 0 ? $"{result.OpsPerSec:F0}" : "-";
        string us = result.UsPerOp > 0 ? $"{result.UsPerOp:F2}" : "-";
        string mib = result.ThroughputMBps > 0 ? $"{result.ThroughputMBps:F1}" : "-";
        Console.WriteLine($"{name,-35} {ops,12} {us,10} {mib,10}");
    }
}


