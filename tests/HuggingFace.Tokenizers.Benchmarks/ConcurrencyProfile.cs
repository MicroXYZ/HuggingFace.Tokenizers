using System.Diagnostics;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;
using HuggingFace.Tokenizers.Models.BPE;
using HuggingFace.Tokenizers.Normalizers;
using HuggingFace.Tokenizers.PreTokenizers;
using HuggingFace.Tokenizers.Serialization;

namespace HuggingFace.Tokenizers.Benchmarks;

/// <summary>
/// 并发性能 Profile — 定位 Encode 管道各阶段耗时。
/// </summary>
public static class ConcurrencyProfile
{
    public static void Run()
    {
        Console.WriteLine("═══════════════════════════════════════════════");
        Console.WriteLine(" Encode 管道各阶段 Profile");
        Console.WriteLine("═══════════════════════════════════════════════");
        Console.WriteLine();

        // 尝试加载预训练模型，否则构建测试模型
        var dataDir = Path.Combine(AppContext.BaseDirectory, "data");
        var bpePath = Path.Combine(dataDir, "tokenizer-rust-bpe.json");
        Tokenizer tokenizer;
        if (File.Exists(bpePath))
        {
            tokenizer = TokenizerLoader.FromFile(bpePath);
            Console.WriteLine($"  ✅ 加载 BPE 模型，词表: {tokenizer.GetVocabSizeWithAddedTokens():N0}");
        }
        else
        {
            Console.WriteLine("  ⚠ 预训练模型未找到，构建测试 BPE 模型");
            tokenizer = CreateTestBpeTokenizer();
            Console.WriteLine($"  ✅ 构建 BPE 模型，词表: {tokenizer.GetVocabSizeWithAddedTokens():N0}");
        }

        // 测试数据：模拟并发基准的输入
        var lines = GenerateTestLines(1000, 25); // 1000 行，每行 ~25 字
        var bigText = string.Join("\n", lines);  // ~25K 字符
        Console.WriteLine($"  测试数据: {bigText.Length:N0} chars, {lines.Length} lines");
        Console.WriteLine();

        // === 1. 完整 Encode 耗时 ===
        ProfileFullEncode(tokenizer, bigText, 1000);

        // === 2. 各阶段拆分 ===
        ProfilePipelineStages(tokenizer, bigText, 1000);

        // === 3. EncodeBatch vs 单条循环 ===
        ProfileBatchVsSingle(tokenizer, lines, 100);

        // === 4. 并发扩展性 ===
        ProfileConcurrencyScaling(tokenizer, lines);
    }

    private static void ProfileFullEncode(Tokenizer tokenizer, string text, int iterations)
    {
        Console.WriteLine("  --- 完整 Encode ---");
        for (int w = 0; w < 50; w++) tokenizer.Encode(text);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            tokenizer.Encode(text);
        sw.Stop();
        Console.WriteLine($"    {iterations} ops: {sw.Elapsed.TotalMilliseconds:F1}ms, {iterations / sw.Elapsed.TotalSeconds:F0} ops/sec, {sw.Elapsed.TotalMicroseconds / iterations:F1} µs/op");
        Console.WriteLine();
    }

    private static void ProfilePipelineStages(Tokenizer tokenizer, string text, int iterations)
    {
        Console.WriteLine("  --- 管道各阶段拆分 ---");

        // 阶段 1: ExtractAndNormalize
        // 通过 EncodeCharOffsets 测量（包含 Normalize + PreTokenize + Tokenize + ToEncoding）
        // 通过 EncodeFast 测量（跳过 ToEncoding 的 offset 追踪）
        // 差值可估算 ToEncoding 开销

        // EncodeFast: 跳过 offset 追踪
        for (int w = 0; w < 50; w++) tokenizer.EncodeFast(text);
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            tokenizer.EncodeFast(text);
        sw.Stop();
        var encodeFastMs = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"    EncodeFast (跳过 offset): {iterations} ops in {encodeFastMs:F1}ms, {iterations / (encodeFastMs / 1000):F0} ops/sec");

        // Encode: 完整
        var sw2 = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            tokenizer.Encode(text);
        sw2.Stop();
        var encodeMs = sw2.Elapsed.TotalMilliseconds;
        Console.WriteLine($"    Encode (完整):             {iterations} ops in {encodeMs:F1}ms, {iterations / (encodeMs / 1000):F0} ops/sec");

        Console.WriteLine($"    ToEncoding 开销:           {encodeMs - encodeFastMs:F1}ms ({(encodeMs - encodeFastMs) / encodeMs * 100:F1}%)");
        Console.WriteLine();
    }

    private static void ProfileBatchVsSingle(Tokenizer tokenizer, string[] lines, int iterations)
    {
        Console.WriteLine("  --- 单条循环 vs EncodeBatch ---");

        // 单条循环
        for (int w = 0; w < 5; w++) foreach (var l in lines) tokenizer.Encode(l);
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            foreach (var l in lines)
                tokenizer.Encode(l);
        sw.Stop();
        var singleMs = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"    单条循环 ({iterations}×{lines.Length}): {singleMs:F1}ms, {iterations * lines.Length / (singleMs / 1000):F0} ops/sec");

        // EncodeBatch
        for (int w = 0; w < 5; w++) tokenizer.EncodeBatch(lines);
        sw.Restart();
        for (int i = 0; i < iterations; i++)
            tokenizer.EncodeBatch(lines);
        sw.Stop();
        var batchMs = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"    EncodeBatch ({iterations}×{lines.Length}): {batchMs:F1}ms, {iterations * lines.Length / (batchMs / 1000):F0} ops/sec");
        Console.WriteLine($"    Batch/Single 加速比: {singleMs / batchMs:F2}x");
        Console.WriteLine();
    }

    private static void ProfileConcurrencyScaling(Tokenizer tokenizer, string[] lines)
    {
        Console.WriteLine("  --- 并发扩展性 ---");

        var testLines = lines.Take(1000).ToArray();
        foreach (var threads in new[] { 1, 2, 4 })
        {
            int linesPerThread = testLines.Length / threads;
            var inputs = new string[threads];
            for (int t = 0; t < threads; t++)
                inputs[t] = string.Join("\n", testLines.AsSpan(t * linesPerThread, linesPerThread).ToArray());

            // 预热
            Parallel.ForEach(inputs, input => tokenizer.Encode(input));

            const int iterations = 20;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
                Parallel.ForEach(inputs, new ParallelOptions { MaxDegreeOfParallelism = threads },
                    input => tokenizer.Encode(input));
            sw.Stop();

            int totalOps = iterations * threads;
            Console.WriteLine($"    {threads}T: {totalOps} ops in {sw.Elapsed.TotalMilliseconds:F1}ms, {totalOps / sw.Elapsed.TotalSeconds:F0} ops/sec");
        }
        Console.WriteLine();
    }

    private static Tokenizer CreateTestBpeTokenizer()
    {
        var vocab = new Dictionary<string, uint>();
        uint id = 0;
        for (char c = 'a'; c <= 'z'; c++) vocab[c.ToString()] = id++;
        for (char c = 'A'; c <= 'Z'; c++) vocab[c.ToString()] = id++;
        for (char c = '0'; c <= '9'; c++) vocab[c.ToString()] = id++;
        vocab[" "] = id++; vocab["."] = id++; vocab[","] = id++; vocab["!"] = id++; vocab["?"] = id++;

        var merges = new List<(string, string)>();
        string[] commonPairs = { "th", "he", "in", "er", "an", "re", "on", "at", "en", "nd",
                                 "the", "ing", "tion", "ed", "es", "or", "te", "of", "it", "is" };
        foreach (var pair in commonPairs)
        {
            if (pair.Length == 2) { merges.Add((pair[0].ToString(), pair[1].ToString())); vocab[pair] = id++; }
            else if (pair.Length == 3)
            {
                var first = pair[..2];
                if (!vocab.ContainsKey(first)) vocab[first] = id++;
                merges.Add((first, pair[2].ToString()));
                vocab[pair] = id++;
            }
        }
        for (char c = 'a'; c <= 'z'; c++) { var p = $"{c}e"; if (!vocab.ContainsKey(p)) { merges.Add((c.ToString(), "e")); vocab[p] = id++; } }

        return new Tokenizer(new BpeModel.BpeBuilder().SetVocab(vocab).SetMerges(merges).Build());
    }

    private static string[] GenerateTestLines(int count, int avgWordsPerLine)
    {
        var words = new[] { "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                           "hello", "world", "test", "data", "sample", "text" };
        var rng = new Random(42);
        var lines = new string[count];
        for (int i = 0; i < count; i++)
        {
            int wordCount = avgWordsPerLine + rng.Next(-3, 4);
            var lineWords = new string[wordCount];
            for (int j = 0; j < wordCount; j++)
                lineWords[j] = words[rng.Next(words.Length)];
            lines[i] = string.Join(" ", lineWords);
        }
        return lines;
    }
}
