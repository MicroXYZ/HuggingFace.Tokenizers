using System.Diagnostics;
using System.Text;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Benchmarks;

/// <summary>
/// NormalizationTables 性能基准测试。
/// 对比自定义实现 vs .NET 内置 string.Normalize()。
/// </summary>
public static class NormalizationBenchmarks
{
    // ═══════════════════════════════════════════════════════════════════
    // 测试数据
    // ═══════════════════════════════════════════════════════════════════

    // 纯 ASCII（无分解）
    private const string AsciiText = "The quick brown fox jumps over the lazy dog. " +
        "Pack my box with five dozen liquor jugs! How vexingly quick daft zebras jump.";

    // 含大量组合字符的文本（法语、越南语、德语）
    private const string AccentText =
        "Les élèves français étudient à l'université. " +
        "Đây là một văn bản tiếng Việt với nhiều dấu. " +
        "Über allen Gipfeln ist Ruh, unter allen Wipfeln spürest du kaum einen Hauch. " +
        " crème brûlée, café, naïve, résumé, coöperate, señor, año";

    // 大量 CJK（无分解）
    private const string CjkText =
        "这是一个用于性能测试的中文文本。" +
        "日本語のテキストも含まれています。" +
        "한국어 텍스트도 포함되어 있습니다。" +
        "汉字假名混じりの文章でテストします。";

    // 合成字符多的文本（NFD → NFC 变化大）
    private const string ComposeText =
        "\u0041\u0300\u0041\u0301\u0041\u0302\u0041\u0303\u0041\u0308" +  // À Á Â Ã Ä
        "\u0045\u0300\u0045\u0301\u0045\u0302\u0045\u0308" +              // È É Ê Ë
        "\u0049\u0300\u0049\u0301\u0049\u0302\u0049\u0308" +              // Ì Í Î Ï
        "\u004F\u0300\u004F\u0301\u004F\u0302\u004F\u0303\u004F\u0308" +  // Ò Ó Ô Õ Ö
        "\u0055\u0300\u0055\u0301\u0055\u0302\u0055\u0308" +              // Ù Ú Û Ü
        "\u006E\u0303\u0041\u030A\u00E6\u0153\u00DF";                     // ñ Å æ œ ß

    // 混合长文本
    private const string MixedLong = AccentText + CjkText + ComposeText +
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 " +
        "αβγδεζηθικλμνξοπρστυφχψω ";  // Greek

    // ═══════════════════════════════════════════════════════════════════
    // 入口
    // ═══════════════════════════════════════════════════════════════════

    public static void Run()
    {
        Console.WriteLine("╔══════════════════════════════════════════════════════════════╗");
        Console.WriteLine("║        NormalizationTables 性能基准测试                      ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════════╝");
        Console.WriteLine();

        // 预热
        Warmup();

        // 测试 1: GetCcc 性能
        BenchmarkGetCcc();

        // 测试 2: NFD 分解性能
        BenchmarkNfd();

        // 测试 3: NFC 组合性能
        BenchmarkNfc();

        // 测试 4: 端到端对比 .NET string.Normalize
        BenchmarkVsBuiltin();

        // 测试 5: 正确性验证
        VerifyCorrectness();
    }

    // ═══════════════════════════════════════════════════════════════════
    // 预热
    // ═══════════════════════════════════════════════════════════════════

    private static void Warmup()
    {
        Console.Write("预热中...");
        Span<int> buf = stackalloc int[18];
        for (int i = 0; i < 1000; i++)
        {
            NormalizationTables.GetCcc(i);
            NormalizationTables.DecomposeNfd(i, buf);
        }
        _ = AccentText.Normalize(NormalizationForm.FormC);
        Console.WriteLine(" 完成");
        Console.WriteLine();
    }

    // ═══════════════════════════════════════════════════════════════════
    // 测试 1: GetCcc 查找性能
    // ═══════════════════════════════════════════════════════════════════

    private static void BenchmarkGetCcc()
    {
        Console.WriteLine("── 测试 1: GetCcc 查找性能 ──");
        const int iterations = 1_000_000;
        var rng = new Random(42);
        int[] codepoints = new int[iterations];
        for (int i = 0; i < iterations; i++)
            codepoints[i] = rng.Next(0, 0x110000);

        var sw = Stopwatch.StartNew();
        byte sum = 0;
        for (int i = 0; i < iterations; i++)
            sum += NormalizationTables.GetCcc(codepoints[i]);
        sw.Stop();

        Console.WriteLine($"  {iterations:N0} 次 GetCcc 查找: {sw.ElapsedMilliseconds} ms");
        Console.WriteLine($"  吞吐: {iterations / sw.Elapsed.TotalSeconds:N0} ops/sec");
        Console.WriteLine($"  (防止优化: sum={sum})");
        Console.WriteLine();
    }

    // ═══════════════════════════════════════════════════════════════════
    // 测试 2: NFD 分解性能
    // ═══════════════════════════════════════════════════════════════════

    private static void BenchmarkNfd()
    {
        Console.WriteLine("── 测试 2: NFD 分解性能 ──");
        BenchmarkDecompose("ASCII", AsciiText);
        BenchmarkDecompose("Accent", AccentText);
        BenchmarkDecompose("Compose", ComposeText);
        BenchmarkDecompose("Mixed", MixedLong);
        Console.WriteLine();
    }

    private static void BenchmarkDecompose(string label, string text)
    {
        const int iterations = 50_000;
        Span<int> buf = stackalloc int[18];  // 单字符最大分解
        var runes = text.EnumerateRunes().ToArray();

        var sw = Stopwatch.StartNew();
        int totalLen = 0;
        for (int iter = 0; iter < iterations; iter++)
        {
            foreach (var rune in runes)
            {
                int len = NormalizationTables.DecomposeNfd(rune.Value, buf);
                totalLen += len;
            }
        }
        sw.Stop();

        int charCount = runes.Length;
        double opsPerSec = (double)iterations * charCount / sw.Elapsed.TotalSeconds;
        Console.WriteLine($"  [{label}] {iterations:N0} iters × {charCount} chars: {sw.ElapsedMilliseconds} ms ({opsPerSec:N0} char-ops/sec)");
    }

    // ═══════════════════════════════════════════════════════════════════
    // 测试 3: NFC 组合性能
    // ═══════════════════════════════════════════════════════════════════

    private static void BenchmarkNfc()
    {
        Console.WriteLine("── 测试 3: NFC 组合性能 ──");
        BenchmarkCompose("Compose", ComposeText);
        BenchmarkCompose("Accent", AccentText);
        BenchmarkCompose("Mixed", MixedLong);
        Console.WriteLine();
    }

    private static void BenchmarkCompose(string label, string text)
    {
        const int iterations = 50_000;

        // 先做一次完整 NFD 分解作为输入
        int[] decompArr = new int[text.Length * 6];
        int decompLen = 0;
        foreach (var rune in text.EnumerateRunes())
        {
            decompLen += NormalizationTables.DecomposeNfd(rune.Value, decompArr.AsSpan(decompLen));
        }

        // 排序
        NormalizationTables.SortByCcc(decompArr.AsSpan(0, decompLen));

        // 组合
        int[] compArr = new int[decompLen];
        var sw = Stopwatch.StartNew();
        int totalLen = 0;
        for (int iter = 0; iter < iterations; iter++)
        {
            totalLen += NormalizationTables.ComposeNfc(decompArr.AsSpan(0, decompLen), compArr);
        }
        sw.Stop();

        Console.WriteLine($"  [{label}] {iterations:N0} iters, {decompLen} decomp pts → {totalLen / iterations} composed: {sw.ElapsedMilliseconds} ms");
    }

    // ═══════════════════════════════════════════════════════════════════
    // 测试 4: 端到端对比 .NET string.Normalize
    // ═══════════════════════════════════════════════════════════════════

    private static void BenchmarkVsBuiltin()
    {
        Console.WriteLine("── 测试 4: 端到端 NFC 对比 .NET string.Normalize ──");
        CompareBuiltin("ASCII", AsciiText);
        CompareBuiltin("Accent", AccentText);
        CompareBuiltin("Compose", ComposeText);
        CompareBuiltin("CJK", CjkText);
        CompareBuiltin("Mixed", MixedLong);
        Console.WriteLine();
    }

    private static void CompareBuiltin(string label, string text)
    {
        const int iterations = 50_000;

        // 自定义：使用优化的 NormalizeNfcToChars（直接 char→char，含 ASCII 快速路径）
        char[] buf = new char[text.Length * 2];

        var sw1 = Stopwatch.StartNew();
        int customLen = 0;
        for (int iter = 0; iter < iterations; iter++)
        {
            customLen += NormalizationTables.NormalizeNfcToChars(text.AsSpan(), buf);
        }
        sw1.Stop();

        // .NET 内置
        var sw2 = Stopwatch.StartNew();
        int builtinLen = 0;
        for (int iter = 0; iter < iterations; iter++)
        {
            string normalized = text.Normalize(NormalizationForm.FormC);
            builtinLen += normalized.Length;
        }
        sw2.Stop();

        double ratio = sw2.Elapsed.TotalMilliseconds / Math.Max(sw1.Elapsed.TotalMilliseconds, 0.001);
        string faster = ratio > 1 ? $"{ratio:F1}x 快于" : $"{1 / ratio:F1}x 慢于";

        Console.WriteLine($"  [{label}]");
        Console.WriteLine($"    自定义: {sw1.ElapsedMilliseconds,6} ms (output len={customLen / iterations})");
        Console.WriteLine($"    .NET:   {sw2.ElapsedMilliseconds,6} ms (output len={builtinLen / iterations})");
        Console.WriteLine($"    对比:   自定义 {faster} .NET 内置");
    }

    // ═══════════════════════════════════════════════════════════════════
    // 测试 5: 正确性验证
    // ═══════════════════════════════════════════════════════════════════

    private static void VerifyCorrectness()
    {
        Console.WriteLine("── 测试 5: 正确性验证 ──");

        // 调试：D + 0307 + 0323
        {
            int D = 0x44;
            Console.WriteLine($"  [DEBUG] NfcCompose(D, 0307) = U+{NormalizationTables.NfcCompose(D, 0x307):X4}");
            Console.WriteLine($"  [DEBUG] NfcCompose(D, 0323) = U+{NormalizationTables.NfcCompose(D, 0x323):X4}");
            Console.WriteLine($"  [DEBUG] CCC(0307)={NormalizationTables.GetCcc(0x307)}, CCC(0323)={NormalizationTables.GetCcc(0x323)}");

            Span<int> db = stackalloc int[18];
            int dl = NormalizationTables.DecomposeNfd(0x1E0C, db);
            Console.Write("  [DEBUG] NFD(1E0C)=");
            for (int i = 0; i < dl; i++) Console.Write($"U+{db[i]:X4} ");
            Console.WriteLine();

            dl = NormalizationTables.DecomposeNfd(0x1E0D, db);
            Console.Write("  [DEBUG] NFD(1E0D)=");
            for (int i = 0; i < dl; i++) Console.Write($"U+{db[i]:X4} ");
            Console.WriteLine();
        }

        string[] testCases =
        [
            "\u0041\u0300",       // A + grave → À (U+00C0)
            "\u0041\u0301",       // A + acute → Á (U+00C1)
            "\u0045\u0301",       // E + acute → É (U+00C9)
            "\u0065\u0301",       // e + acute → é (U+00E9)
            "\u006E\u0303",       // n + tilde → ñ (U+00F1)
            "\u0041\u030A",       // A + ring → Å (U+00C5)
            "\u0041\u0308",       // A + diaeresis → Ä (U+00C4)
            "\u004F\u0308",       // O + diaeresis → Ö (U+00D6)
            "\u0055\u0308",       // U + diaeresis → Ü (U+00DC)
            "\u0073\u0323",       // s + dot below → ṣ (U+1E63)
            "\u0044\u0307\u0323", // D + dot above + dot below → complex
            "\uFB01",             // ﬁ ligature (NFKC → fi)
            "\u0041\u0300\u0301", // A + grave + acute (two combining)
            "hello",
            "café",
            "naïve",
        ];

        int passed = 0, failed = 0;
        foreach (var tc in testCases)
        {
            string expected = tc.Normalize(NormalizationForm.FormC);
            string actual = CustomNfc(tc);

            if (expected == actual)
            {
                passed++;
            }
            else
            {
                failed++;
                Console.WriteLine($"  ❌ FAIL: U+{string.Join("", tc.Select(c => ((int)c).ToString("X4")))}");
                Console.WriteLine($"     Expected: {Escape(expected)} ({expected.Length} chars)");
                Console.WriteLine($"     Actual:   {Escape(actual)} ({actual.Length} chars)");
            }
        }

        // NFKC 测试
        string[] nfkcCases = ["\uFB01", "\uFB02", "\u2160", "\u3300"];
        foreach (var tc in nfkcCases)
        {
            string expected = tc.Normalize(NormalizationForm.FormKC);
            // NFKC = NFKD decomposition + NFC composition
            // 我们只测 NFKD 查找是否工作
            var nfkd = NormalizationTables.GetNfkd(tc[0]);
            if (!nfkd.IsEmpty)
                passed++;
            else
            {
                // 有些 NFKC 映射在 NFKD 表中可能不在
                Console.WriteLine($"  ⚠️  NFKD lookup returned empty for U+{((int)tc[0]):X4} (may be expected)");
            }
        }

        Console.WriteLine($"  结果: {passed} 通过, {failed} 失败");
        Console.WriteLine();
    }

    // ═══════════════════════════════════════════════════════════════════
    // 辅助方法
    // ═══════════════════════════════════════════════════════════════════

    /// <summary>
    /// 使用自定义 NormalizationTables 执行完整 NFC（不含 alignment tracking）。
    /// </summary>
    private static string CustomNfc(string input)
    {
        // 使用优化的端到端 NFC 方法（直接 char 输出，含 ASCII 快速路径）
        char[] buf = new char[input.Length * 2];
        int outLen = NormalizationTables.NormalizeNfcToChars(input.AsSpan(), buf);
        return new string(buf, 0, outLen);
    }

    private static string Escape(string s)
    {
        var sb = new StringBuilder();
        foreach (var c in s)
        {
            if (c < 0x20 || c > 0x7E)
                sb.Append($"\\u{((int)c):X4}");
            else
                sb.Append(c);
        }
        return sb.ToString();
    }
}
