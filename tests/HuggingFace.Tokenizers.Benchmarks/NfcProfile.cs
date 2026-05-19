using System.Diagnostics;
using System.Text;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Benchmarks;

/// <summary>
/// NFC 标准化各阶段 Profile — 定位瓶颈。
/// </summary>
public static class NfcProfile
{
    public static void Run()
    {
        Console.WriteLine("═══════════════════════════════════════════════");
        Console.WriteLine(" NFC 标准化各阶段 Profile");
        Console.WriteLine("═══════════════════════════════════════════════");
        Console.WriteLine();

        var chineseText = GenerateChineseText(50_000);
        var asciiText = GenerateAsciiText(50_000);
        Console.WriteLine($"  测试数据: ASCII={asciiText.Length} chars, 中文={chineseText.Length} chars");
        Console.WriteLine();

        // 1. 完整 NFC（包含 BuildNormalizationTransform）
        ProfileFullNfc("ASCII 50K", asciiText, 500);
        ProfileFullNfc("中文 50K", chineseText, 500);

        // 2. UnicodeConsistency.Nfc（纯规范化，不含 Transform）
        ProfileNfcOnly("ASCII 50K", asciiText, 500);
        ProfileNfcOnly("中文 50K", chineseText, 500);

        // 3. BuildNormalizationTransform 单独
        ProfileBuildTransform("ASCII 50K", asciiText, 500);
        ProfileBuildTransform("中文 50K", chineseText, 500);

        // 4. SortByCombiningClass 单独
        ProfileSortCcc("中文 50K", chineseText, 500);

        // 5. DecomposeFully 单独
        ProfileDecompose("中文 50K", chineseText, 500);
    }

    private static void ProfileFullNfc(string label, string text, int iterations)
    {
        Console.WriteLine($"  --- 完整 NFC (含 Transform): {label} ---");
        for (int w = 0; w < 10; w++) { var ns = new NormalizedString(text); ns.Nfc(); }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            var ns = new NormalizedString(text);
            ns.Nfc();
        }
        sw.Stop();
        Console.WriteLine($"    {iterations} ops: {sw.Elapsed.TotalMilliseconds:F1}ms, {sw.Elapsed.TotalMicroseconds / iterations:F1} µs/op");
        Console.WriteLine();
    }

    private static void ProfileNfcOnly(string label, string text, int iterations)
    {
        Console.WriteLine($"  --- UnicodeConsistency.Nfc (纯规范化): {label} ---");
        for (int w = 0; w < 10; w++) UnicodeConsistency.Nfc(text);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            UnicodeConsistency.Nfc(text);
        sw.Stop();
        Console.WriteLine($"    {iterations} ops: {sw.Elapsed.TotalMilliseconds:F1}ms, {sw.Elapsed.TotalMicroseconds / iterations:F1} µs/op");
        Console.WriteLine();
    }

    private static void ProfileBuildTransform(string label, string text, int iterations)
    {
        Console.WriteLine($"  --- BuildNormalizationTransform: {label} ---");
        var nfcResult = UnicodeConsistency.Nfc(text);
        for (int w = 0; w < 10; w++) UnicodeNormalizer.BuildNormalizationTransform(text, nfcResult, NormalizationForm.FormC);

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
            UnicodeNormalizer.BuildNormalizationTransform(text, nfcResult, NormalizationForm.FormC);
        sw.Stop();
        Console.WriteLine($"    {iterations} ops: {sw.Elapsed.TotalMilliseconds:F1}ms, {sw.Elapsed.TotalMicroseconds / iterations:F1} µs/op");
        Console.WriteLine();
    }

    private static void ProfileSortCcc(string label, string text, int iterations)
    {
        Console.WriteLine($"  --- SortByCombiningClass: {label} ---");
        // 预先分解
        var span = text.AsSpan();
        var decomposed = new int[span.Length * 6];
        int len = 0;
        foreach (var rune in span.EnumerateRunes())
        {
            int cp = rune.Value;
            var (d1, d2) = NormalizationTables.GetNfd(cp);
            if (d1 == cp) { decomposed[len++] = cp; continue; }
            decomposed[len++] = d1;
            if (d2 != 0) decomposed[len++] = d2;
        }

        for (int w = 0; w < 10; w++)
        {
            var copy = decomposed.AsSpan(0, len).ToArray();
            SortByCombiningClass(copy);
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            var copy = decomposed.AsSpan(0, len).ToArray();
            SortByCombiningClass(copy);
        }
        sw.Stop();
        Console.WriteLine($"    {iterations} ops: {sw.Elapsed.TotalMilliseconds:F1}ms, {sw.Elapsed.TotalMicroseconds / iterations:F1} µs/op (decomposed len={len})");
        Console.WriteLine();
    }

    private static void ProfileDecompose(string label, string text, int iterations)
    {
        Console.WriteLine($"  --- DecomposeFully (NFD): {label} ---");
        var span = text.AsSpan();
        for (int w = 0; w < 10; w++)
        {
            var buf = new int[span.Length * 6];
            DecomposeFully(span, buf);
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            var buf = new int[span.Length * 6];
            DecomposeFully(span, buf);
        }
        sw.Stop();
        Console.WriteLine($"    {iterations} ops: {sw.Elapsed.TotalMilliseconds:F1}ms, {sw.Elapsed.TotalMicroseconds / iterations:F1} µs/op");
        Console.WriteLine();
    }

    // 复制内部方法用于独立测试
    private static int DecomposeFully(ReadOnlySpan<char> input, Span<int> output)
    {
        int pos = 0;
        foreach (var rune in input.EnumerateRunes())
            pos += DecomposeNfd(rune.Value, output.Slice(pos));
        return pos;
    }

    private static int DecomposeNfd(int cp, Span<int> output)
    {
        var (d1, d2) = NormalizationTables.GetNfd(cp);
        if (d1 == cp) { output[0] = cp; return 1; }
        int pos = DecomposeNfd(d1, output);
        if (d2 != 0) pos += DecomposeNfd(d2, output.Slice(pos));
        return pos;
    }

    private static void SortByCombiningClass(Span<int> decomposed)
    {
        int len = decomposed.Length;
        for (int i = 0; i < len - 1; i++)
        {
            int cccI = NormalizationTables.GetCcc(decomposed[i]);
            if (cccI == 0) continue;
            for (int j = i + 1; j < len; j++)
            {
                int cccJ = NormalizationTables.GetCcc(decomposed[j]);
                if (cccJ == 0) break;
                if (cccI > cccJ)
                {
                    (decomposed[i], decomposed[j]) = (decomposed[j], decomposed[i]);
                    cccI = cccJ;
                }
            }
        }
    }

    private static string GenerateChineseText(int length)
    {
        var chars = "的一是不了人我在有他这为之大来以个中上们到说国和地也子时道出会三要于下得可你年生";
        var sb = new StringBuilder(length);
        var rng = new Random(42);
        while (sb.Length < length) sb.Append(chars[rng.Next(chars.Length)]);
        return sb.ToString(0, length);
    }

    private static string GenerateAsciiText(int length)
    {
        var words = new[] { "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog" };
        var sb = new StringBuilder(length);
        var rng = new Random(42);
        while (sb.Length < length) { if (sb.Length > 0) sb.Append(' '); sb.Append(words[rng.Next(words.Length)]); }
        return sb.ToString(0, length);
    }
}
