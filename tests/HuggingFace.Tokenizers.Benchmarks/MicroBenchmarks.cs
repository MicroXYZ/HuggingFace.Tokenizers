using System.Diagnostics;
using System.Runtime.CompilerServices;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;
using HuggingFace.Tokenizers.Models.BPE;
using HuggingFace.Tokenizers.Normalizers;
using HuggingFace.Tokenizers.Serialization;

namespace HuggingFace.Tokenizers.Benchmarks;

/// <summary>
/// 微基准测试 — 验证 Phase 4 (AlignmentTracker) 和 Phase 5 (BPE MergeAll) 的性能特征。
/// 用法：
///   dotnet run -- --micro
///   dotnet run -- --micro --alignment
///   dotnet run -- --micro --mergeall
/// </summary>
public static class MicroBenchmarks
{
    public static int Run(string[] args)
    {
        bool runAlignment = args.Contains("--alignment") || args.Contains("--all") || args.Length <= 1;
        bool runMergeAll = args.Contains("--mergeall") || args.Contains("--all") || args.Length <= 1;
        bool jsonOutput = args.Contains("--json");

        Console.WriteLine("╔══════════════════════════════════════════════╗");
        Console.WriteLine("║   Micro Benchmarks (Phase 4/5 验证)         ║");
        Console.WriteLine("╠══════════════════════════════════════════════╣");
        Console.WriteLine($"║  Runtime:  .NET {Environment.Version}, {IntPtr.Size * 8}-bit");
        Console.WriteLine($"║  CPU:      {Environment.ProcessorCount} cores");
        Console.WriteLine("╚══════════════════════════════════════════════╝");
        Console.WriteLine();

        if (runAlignment)
            RunAlignmentBenchmarks();

        if (runMergeAll)
            RunMergeAllBenchmarks();

        return 0;
    }

    // ════════════════════════════════════════════════
    //  Phase 4: AlignmentTracker 微基准测试
    // ════════════════════════════════════════════════

    private static void RunAlignmentBenchmarks()
    {
        Console.WriteLine("═══════════════════════════════════════════════");
        Console.WriteLine(" Phase 4: AlignmentTracker 性能验证");
        Console.WriteLine("═══════════════════════════════════════════════");
        Console.WriteLine();

        // 测试数据
        var asciiText = GenerateAsciiText(50_000);
        var chineseText = GenerateChineseText(50_000);
        var mixedText = GenerateMixedText(50_000);
        var emojiText = GenerateEmojiText(10_000);

        Console.WriteLine($"测试数据: ASCII={asciiText.Length} chars, 中文={chineseText.Length} chars, 混合={mixedText.Length} chars, Emoji={emojiText.Length} chars");
        Console.WriteLine();

        // 1. ConvertOffsets 性能：纯 ASCII（1:1 对齐，应该最快）
        BenchmarkConvertOffsets("ASCII (1:1 对齐)", asciiText, 1000);

        // 2. ConvertOffsets 性能：中文（多字节 UTF-8，对齐列表更长）
        BenchmarkConvertOffsets("中文 (多字节 UTF-8)", chineseText, 1000);

        // 3. ConvertOffsets 性能：混合文本
        BenchmarkConvertOffsets("混合文本", mixedText, 1000);

        // 4. ConvertOffsets 性能：Emoji（4字节 UTF-16 surrogate pairs）
        BenchmarkConvertOffsets("Emoji (surrogate pairs)", emojiText, 1000);

        // 5. Transform 性能：小变换（100 个替换）
        BenchmarkTransform("小变换 (100 替换)", asciiText, 100, 1000);

        // 6. Transform 性能：中变换（1000 个替换）
        BenchmarkTransform("中变换 (1000 替换)", asciiText, 1000, 500);

        // 7. Transform 性能：大变换（NFC 标准化，全量变换）
        BenchmarkNfcTransform("NFC 标准化 (全量)", asciiText, 500);
        BenchmarkNfcTransform("NFC 标准化 (中文)", chineseText, 500);

        // 8. AlignmentTracker 构造 + 初始化性能
        BenchmarkAlignmentInit("ASCII 50K", asciiText, 1000);
        BenchmarkAlignmentInit("中文 50K", chineseText, 1000);
        BenchmarkAlignmentInit("Emoji 10K", emojiText, 1000);
    }

    private static void BenchmarkConvertOffsets(string label, string text, int iterations)
    {
        Console.WriteLine($"  --- ConvertOffsets: {label} ---");

        // 创建 NormalizedString
        var ns = new NormalizedString(text);

        // 预热
        for (int i = 0; i < 10; i++)
            ns.ConvertOffsets(OffsetReferential.Original, 0..Math.Min(100, text.Length));

        // 测试 Original → Normalized 方向（使用二分查找）
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            // 随机采样 100 个偏移转换
            for (int j = 0; j < 100; j++)
            {
                int start = Random.Shared.Next(0, text.Length - 1);
                int end = Random.Shared.Next(start + 1, Math.Min(start + 1000, text.Length));
                ns.ConvertOffsets(OffsetReferential.Original, start..end);
            }
        }
        sw.Stop();
        int totalOps = iterations * 100;
        Console.WriteLine($"    Original→Normalized: {totalOps:N0} ops in {sw.Elapsed.TotalMilliseconds:F1}ms, {totalOps / sw.Elapsed.TotalSeconds:F0} ops/sec, {sw.Elapsed.TotalMicroseconds / totalOps:F2} µs/op");

        // 测试 Normalized → Original 方向（使用 ExpandAlignments，O(1)）
        sw.Restart();
        for (int i = 0; i < iterations; i++)
        {
            for (int j = 0; j < 100; j++)
            {
                int start = Random.Shared.Next(0, text.Length - 1);
                int end = Random.Shared.Next(start + 1, Math.Min(start + 1000, text.Length));
                ns.ConvertOffsets(OffsetReferential.Normalized, start..end);
            }
        }
        sw.Stop();
        Console.WriteLine($"    Normalized→Original: {totalOps:N0} ops in {sw.Elapsed.TotalMilliseconds:F1}ms, {totalOps / sw.Elapsed.TotalSeconds:F0} ops/sec, {sw.Elapsed.TotalMicroseconds / totalOps:F2} µs/op");
        Console.WriteLine();
    }

    private static void BenchmarkTransform(string label, string text, int transformSize, int iterations)
    {
        Console.WriteLine($"  --- Transform: {label} ---");

        var ns = new NormalizedString(text);

        // 构造变换：替换 transformSize 个字符
        var transforms = new List<(char Char, int Change)>();
        for (int i = 0; i < transformSize && i < text.Length; i++)
            transforms.Add(('X', 0)); // 1:1 替换

        // 预热
        for (int i = 0; i < 5; i++)
        {
            var warmupNs = new NormalizedString(text);
            warmupNs.Transform(transforms, 0);
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            var testNs = new NormalizedString(text);
            testNs.Transform(transforms, 0);
        }
        sw.Stop();

        Console.WriteLine($"    {iterations:N0} ops in {sw.Elapsed.TotalMilliseconds:F1}ms, {iterations / sw.Elapsed.TotalSeconds:F0} ops/sec, {sw.Elapsed.TotalMicroseconds / iterations:F2} µs/op");
        Console.WriteLine();
    }

    private static void BenchmarkNfcTransform(string label, string text, int iterations)
    {
        Console.WriteLine($"  --- NFC Transform: {label} ---");

        // 预热
        for (int i = 0; i < 5; i++)
        {
            var warmupNs = new NormalizedString(text);
            warmupNs.Nfc();
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            var testNs = new NormalizedString(text);
            testNs.Nfc();
        }
        sw.Stop();

        Console.WriteLine($"    {iterations:N0} ops in {sw.Elapsed.TotalMilliseconds:F1}ms, {iterations / sw.Elapsed.TotalSeconds:F0} ops/sec, {sw.Elapsed.TotalMicroseconds / iterations:F2} µs/op");
        Console.WriteLine();
    }

    private static void BenchmarkAlignmentInit(string label, string text, int iterations)
    {
        Console.WriteLine($"  --- AlignmentTracker Init: {label} ---");

        // 预热
        for (int i = 0; i < 10; i++)
            _ = new NormalizedString(text);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            _ = new NormalizedString(text);
        sw.Stop();

        Console.WriteLine($"    {iterations:N0} ops in {sw.Elapsed.TotalMilliseconds:F1}ms, {iterations / sw.Elapsed.TotalSeconds:F0} ops/sec, {sw.Elapsed.TotalMicroseconds / iterations:F2} µs/op");
        Console.WriteLine();
    }

    // ════════════════════════════════════════════════
    //  Phase 5: BPE MergeAll 微基准测试
    // ════════════════════════════════════════════════

    private static void RunMergeAllBenchmarks()
    {
        Console.WriteLine("═══════════════════════════════════════════════");
        Console.WriteLine(" Phase 5: BPE MergeAll 性能验证");
        Console.WriteLine("═══════════════════════════════════════════════");
        Console.WriteLine();

        // 尝试加载预训练 BPE 模型，否则用代码构建
        Tokenizer? tokenizer = null;
        var dataDir = Path.Combine(AppContext.BaseDirectory, "data");
        var bpePath = Path.Combine(dataDir, "tokenizer-rust-bpe.json");
        if (File.Exists(bpePath))
        {
            tokenizer = TokenizerLoader.FromFile(bpePath);
            Console.WriteLine($"  ✅ 加载 BPE 模型，词表大小: {tokenizer.GetVocabSizeWithAddedTokens():N0}");
        }
        else
        {
            Console.WriteLine("  ⚠ 预训练 BPE 模型未找到，使用代码构建测试模型");
            tokenizer = CreateTestBpeTokenizer();
            Console.WriteLine($"  ✅ 构建 BPE 模型，词表大小: {tokenizer.GetVocabSizeWithAddedTokens():N0}");
        }
        Console.WriteLine();

        // 测试不同长度的输入
        var shortText = "Hello world";
        var mediumText = "The quick brown fox jumps over the lazy dog. " +
                         "This is a medium-length sentence for testing BPE merge performance.";
        var longText = string.Join(" ", Enumerable.Range(0, 100).Select(i =>
            "The quick brown fox jumps over the lazy dog."));

        Console.WriteLine($"测试数据: short={shortText.Length} chars, medium={mediumText.Length} chars, long={longText.Length} chars");
        Console.WriteLine();

        // 1. 单条 Encode 性能（包含完整管道）
        BenchmarkBpeEncode("短文本", shortText, 10000, tokenizer);
        BenchmarkBpeEncode("中等文本", mediumText, 5000, tokenizer);
        BenchmarkBpeEncode("长文本", longText, 500, tokenizer);

        // 2. EncodeFast 性能（跳过 offset 追踪）
        BenchmarkBpeEncodeFast("短文本", shortText, 10000, tokenizer);
        BenchmarkBpeEncodeFast("中等文本", mediumText, 5000, tokenizer);
        BenchmarkBpeEncodeFast("长文本", longText, 500, tokenizer);

        // 3. 不同截断长度的性能
        BenchmarkBpeTruncation("50K→512", 50_000, 512, 100, tokenizer);
        BenchmarkBpeTruncation("50K→2048", 50_000, 2048, 100, tokenizer);
        BenchmarkBpeTruncation("50K→8192", 50_000, 8192, 100, tokenizer);

        // 4. 并发性能
        BenchmarkBpeConcurrent("4 线程", 4, tokenizer);
    }

    private static void BenchmarkBpeEncode(string label, string text, int iterations, Tokenizer tokenizer)
    {
        Console.WriteLine($"  --- BPE Encode: {label} ({text.Length} chars) ---");

        // 预热
        for (int i = 0; i < 100; i++) tokenizer.Encode(text);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            tokenizer.Encode(text);
        sw.Stop();

        Console.WriteLine($"    {iterations:N0} ops: {sw.Elapsed.TotalMilliseconds:F1}ms, {iterations / sw.Elapsed.TotalSeconds:F0} ops/sec, {sw.Elapsed.TotalMicroseconds / iterations:F2} µs/op");
        Console.WriteLine();
    }

    private static void BenchmarkBpeEncodeFast(string label, string text, int iterations, Tokenizer tokenizer)
    {
        Console.WriteLine($"  --- BPE EncodeFast: {label} ({text.Length} chars) ---");

        // 预热
        for (int i = 0; i < 100; i++) tokenizer.EncodeFast(text);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            tokenizer.EncodeFast(text);
        sw.Stop();

        Console.WriteLine($"    {iterations:N0} ops: {sw.Elapsed.TotalMilliseconds:F1}ms, {iterations / sw.Elapsed.TotalSeconds:F0} ops/sec, {sw.Elapsed.TotalMicroseconds / iterations:F2} µs/op");
        Console.WriteLine();
    }

    private static void BenchmarkBpeTruncation(string label, int inputSize, int maxLength, int iterations, Tokenizer tokenizer)
    {
        Console.WriteLine($"  --- BPE Truncation: {label} ({inputSize} chars → {maxLength} tokens) ---");

        var text = GenerateAsciiText(inputSize);
        tokenizer.Truncation = new TruncationParams
        {
            MaxLength = maxLength,
            Strategy = TruncationStrategy.LongestFirst
        };

        // 预热
        tokenizer.Encode(text);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            tokenizer.Encode(text);
        sw.Stop();

        tokenizer.Truncation = null;

        Console.WriteLine($"    {iterations:N0} ops: {sw.Elapsed.TotalMilliseconds:F1}ms, {iterations / sw.Elapsed.TotalSeconds:F0} ops/sec, {sw.Elapsed.TotalMicroseconds / iterations:F2} µs/op");
        Console.WriteLine();
    }

    private static void BenchmarkBpeConcurrent(string label, int numThreads, Tokenizer tokenizer)
    {
        Console.WriteLine($"  --- BPE Concurrent: {label} ---");

        var testLines = new string[numThreads];
        for (int t = 0; t < numThreads; t++)
            testLines[t] = GenerateAsciiText(10_000);

        // 预热
        Parallel.ForEach(testLines, input => tokenizer.Encode(input));

        const int iterations = 20;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            Parallel.ForEach(testLines, new ParallelOptions { MaxDegreeOfParallelism = numThreads },
                input => tokenizer.Encode(input));
        sw.Stop();

        int totalOps = iterations * numThreads;
        Console.WriteLine($"    {totalOps:N0} ops ({numThreads} threads): {sw.Elapsed.TotalMilliseconds:F1}ms, {totalOps / sw.Elapsed.TotalSeconds:F0} ops/sec");
        Console.WriteLine();
    }

    /// <summary>
    /// 创建一个简单的 BPE 测试 tokenizer，用于 MergeAll 微基准测试。
    /// 构建一个小型词表（基础字符 + 常见合并），足够测试 MergeAll 性能。
    /// </summary>
    private static Tokenizer CreateTestBpeTokenizer()
    {
        // 基础字符词表
        var vocab = new Dictionary<string, uint>();
        uint id = 0;

        // 单字符 token (a-z, A-Z, 0-9, 空格, 标点)
        for (char c = 'a'; c <= 'z'; c++) vocab[c.ToString()] = id++;
        for (char c = 'A'; c <= 'Z'; c++) vocab[c.ToString()] = id++;
        for (char c = '0'; c <= '9'; c++) vocab[c.ToString()] = id++;
        vocab[" "] = id++;
        vocab["."] = id++;
        vocab[","] = id++;
        vocab["!"] = id++;
        vocab["?"] = id++;

        // 常见双字符合并
        var merges = new List<(string, string)>();
        string[] commonPairs = { "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
                                 "the", "ing", "tion", "ed", "es", "or", "te", "of", "it", "is" };
        foreach (var pair in commonPairs)
        {
            if (pair.Length == 2)
            {
                merges.Add((pair[0].ToString(), pair[1].ToString()));
                vocab[pair] = id++;
            }
            else if (pair.Length == 3)
            {
                // 合并前两个字符，再合并第三个
                var first = pair[..2];
                if (!vocab.ContainsKey(first)) vocab[first] = id++;
                merges.Add((first, pair[2].ToString()));
                vocab[pair] = id++;
            }
        }

        // 更多合并以增加词表大小
        for (char c = 'a'; c <= 'z'; c++)
        {
            var pair = $"{c}e";
            if (!vocab.ContainsKey(pair))
            {
                merges.Add((c.ToString(), "e"));
                vocab[pair] = id++;
            }
        }

        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges(merges)
            .Build();

        return new Tokenizer(model);
    }

    // ════════════════════════════════════════════════
    //  测试数据生成
    // ════════════════════════════════════════════════

    private static string GenerateAsciiText(int length)
    {
        var words = new[] { "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                           "hello", "world", "test", "data", "sample", "text", "string", "value" };
        var sb = new System.Text.StringBuilder(length);
        var rng = new Random(42);
        while (sb.Length < length)
        {
            if (sb.Length > 0) sb.Append(' ');
            sb.Append(words[rng.Next(words.Length)]);
        }
        return sb.ToString(0, length);
    }

    private static string GenerateChineseText(int length)
    {
        var chars = "的一是不了人我在有他这为之大来以个中上们到说国和地也子时道出会三要于下得可你年生";
        var sb = new System.Text.StringBuilder(length);
        var rng = new Random(42);
        while (sb.Length < length)
            sb.Append(chars[rng.Next(chars.Length)]);
        return sb.ToString(0, length);
    }

    private static string GenerateMixedText(int length)
    {
        var sb = new System.Text.StringBuilder(length);
        var rng = new Random(42);
        while (sb.Length < length)
        {
            int choice = rng.Next(4);
            switch (choice)
            {
                case 0: // ASCII word
                    sb.Append("hello");
                    break;
                case 1: // Chinese
                    sb.Append("你好");
                    break;
                case 2: // Arabic
                    sb.Append("مرحبا");
                    break;
                case 3: // Hindi
                    sb.Append("नमस्ते");
                    break;
            }
            if (sb.Length < length) sb.Append(' ');
        }
        return sb.ToString(0, Math.Min(sb.Length, length));
    }

    private static string GenerateEmojiText(int length)
    {
        var emojis = new[] { "😀", "🌍", "🎉", "👨‍👩‍👧‍👦", "🏳️‍🌈", "🇺🇸", "🔥", "💯" };
        var sb = new System.Text.StringBuilder(length * 4);
        var rng = new Random(42);
        while (sb.Length < length)
        {
            sb.Append(emojis[rng.Next(emojis.Length)]);
        }
        return sb.ToString(0, Math.Min(sb.Length, length));
    }
}
