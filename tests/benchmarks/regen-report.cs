#!/usr/bin/env dotnet
// 重新生成报告（使用已有 result 数据）
// 用法: dotnet run --file tests/benchmarks/regen-report.cs

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

string outDir = Path.Combine(Environment.CurrentDirectory, "tests", "benchmarks", "result");
if (!File.Exists(Path.Combine(outDir, "meta.json")))
{
    Console.WriteLine("❌ 找不到 result/meta.json，请先运行全量基准测试");
    Environment.Exit(1);
}

Console.WriteLine("▶ 重新生成 REPORT.md ...");
GenerateReport(outDir);
Console.WriteLine("✅ 完成");

// ═══ 以下是 GenerateReport 方法（与 run.cs 完全一致） ═══

static Dictionary<string, JsonElement> LoadJ(string p)
{
    try
    {
        using var doc = JsonDocument.Parse(File.ReadAllText(p));
        return doc.RootElement.EnumerateObject().ToDictionary(prop => prop.Name, prop => prop.Value.Clone());
    }
    catch { return new(); }
}

static void GenerateReport(string outDir)
{
    var meta = LoadJ(Path.Combine(outDir, "meta.json"));
    var jit = LoadJ(Path.Combine(outDir, "perf-jit.json"));
    var aot = LoadJ(Path.Combine(outDir, "perf-aot.json"));
    var rust = LoadJ(Path.Combine(outDir, "perf-rust.json"));
    var cons = LoadJ(Path.Combine(outDir, "consistency-summary.json"));

    string now = DateTime.UtcNow.ToString("yyyy-MM-dd HH:mm:ss") + " UTC";
    string seed = meta.TryGetValue("seed", out var s) ? s.ToString() : "?";
    bool hasRust = rust.Count > 0;

    string MS(string key) => meta.TryGetValue(key, out var v) ? v.ValueKind switch
    {
        JsonValueKind.String => v.GetString() ?? "?",
        JsonValueKind.Number => v.ToString(),
        JsonValueKind.True => "true",
        JsonValueKind.False => "false",
        _ => v.ToString()
    } : "?";

    double GP(JsonElement e, string prop) =>
        e.ValueKind == JsonValueKind.Object && e.TryGetProperty(prop, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : 0;
    string Fmt(double v) => v > 0 ? $"{v:N0}" : "-";
    string Ratio(double a, double b) => a > 0 && b > 0 ? $"{a / b:F2}x" : "-";

    var sb = new StringBuilder();
    sb.AppendLine("# Tokenizers 基准测试报告");
    sb.AppendLine();

    // ── 测试环境 ──
    sb.AppendLine("## 测试环境");
    sb.AppendLine();
    sb.AppendLine("| 项目 | 值 |");
    sb.AppendLine("|------|-----|");
    sb.AppendLine($"| 生成时间 | {now} |");
    sb.AppendLine($"| 随机种子 | `{seed}` |");
    sb.AppendLine($"| 操作系统 | {MS("os")} |");
    sb.AppendLine($"| CPU | {MS("cpu")} |");
    sb.AppendLine($"| 内存 | {MS("memory")} |");
    sb.AppendLine();

    // ── 软件版本与编译设置 ──
    sb.AppendLine("## 软件版本与编译设置");
    sb.AppendLine();
    sb.AppendLine("| 项目 | 值 |");
    sb.AppendLine("|------|-----|");
    sb.AppendLine($"| .NET SDK | {MS("dotnet_version")} |");
    sb.AppendLine($"| .NET Runtime | {MS("dotnet_runtime")} |");
    sb.AppendLine($"| .NET 编译设置 | {MS("dotnet_settings")} |");
    sb.AppendLine($"| Rust rustc | {MS("rustc_version")} |");
    sb.AppendLine($"| Rust cargo | {MS("cargo_version")} |");
    sb.AppendLine($"| Rust 编译设置 | {MS("rust_settings")} |");
    sb.AppendLine($"| GCC | {MS("gcc_version")} |");
    sb.AppendLine();

    // ── 测试配置 ──
    sb.AppendLine("## 测试配置");
    sb.AppendLine();
    sb.AppendLine($"`{MS("benchmark_config")}`");
    sb.AppendLine();

    // ── 辅助 ──
    static string PctFmt(double value, double rustBaseline)
    {
        if (value <= 0) return "-";
        if (rustBaseline <= 0) return $"{value:N0}";
        int pct = (int)Math.Round(value / rustBaseline * 100);
        return $"{value:N0} ({pct}%)";
    }
    static string PctFmtF(double value, double rustBaseline)
    {
        if (value <= 0) return "-";
        if (rustBaseline <= 0) return $"{value:F1}";
        int pct = (int)Math.Round(value / rustBaseline * 100);
        return $"{value:F1} ({pct}%)";
    }

    // ═══ 1. 编码性能 ═══
    sb.AppendLine("## 1. 编码性能");
    sb.AppendLine();

    var encodeModels = new (string key, string label, string[] tests)[]
    {
        ("bpe", "BPE", new[] { "BpeEncodeSingle", "BpeEncodeBatch", "BpeEncodeFast" }),
        ("unigram", "Unigram", new[] { "UnigramEncodeSingle", "UnigramEncodeBatch" }),
        ("wordpiece", "WordPiece", new[] { "WordPieceEncodeSingle", "WordPieceEncodeBatch", "WordPieceEncodeFast" }),
        ("wordlevel", "WordLevel", new[] { "WordLevelEncodeSingle", "WordLevelEncodeBatch" }),
        ("qwen", "Qwen2.5", new[] { "Qwen25EncodeSingle", "Qwen25EncodeBatch" }),
    };

    foreach (var (_, modelName, tests) in encodeModels)
    {
        sb.AppendLine($"### {modelName}");
        sb.AppendLine();

        if (hasRust)
        {
            sb.AppendLine("| 指标 | Rust (100%) | AOT | JIT |");
            sb.AppendLine("|------|-------------|-----|-----|");

            foreach (var key in tests)
            {
                jit.TryGetValue(key, out var j); aot.TryGetValue(key, out var a); rust.TryGetValue(key, out var r);
                double rO = GP(r, "OpsPerSec"), aO = GP(a, "OpsPerSec"), jO = GP(j, "OpsPerSec");
                string niceName = key
                    .Replace("BpeEncodeSingle", "单条 Encode").Replace("BpeEncodeBatch", "批量 Encode (lines/s)")
                    .Replace("BpeEncodeFast", "EncodeFast")
                    .Replace("UnigramEncodeSingle", "单条 Encode").Replace("UnigramEncodeBatch", "批量 Encode (lines/s)")
                    .Replace("WordPieceEncodeSingle", "单条 Encode").Replace("WordPieceEncodeBatch", "批量 Encode (lines/s)")
                    .Replace("WordPieceEncodeFast", "EncodeFast")
                    .Replace("Qwen25EncodeSingle", "单条 Encode").Replace("Qwen25EncodeBatch", "批量 Encode (lines/s)");
                bool isEncodeFast = key.Contains("EncodeFast");
                sb.AppendLine($"| {niceName} | {(isEncodeFast ? "—" : $"{rO:N0}")} | {PctFmt(aO, rO)} | {PctFmt(jO, rO)} |");
            }
        }
        else
        {
            sb.AppendLine("| 指标 | JIT | AOT | AOT/JIT |");
            sb.AppendLine("|------|-----|-----|---------|");
            foreach (var key in tests)
            {
                jit.TryGetValue(key, out var j); aot.TryGetValue(key, out var a);
                double jO = GP(j, "OpsPerSec"), aO = GP(a, "OpsPerSec");
                string niceName = key
                    .Replace("BpeEncodeSingle", "单条 Encode").Replace("BpeEncodeBatch", "批量 Encode")
                    .Replace("BpeEncodeFast", "EncodeFast")
                    .Replace("UnigramEncodeSingle", "单条 Encode").Replace("UnigramEncodeBatch", "批量 Encode")
                    .Replace("WordPieceEncodeSingle", "单条 Encode").Replace("WordPieceEncodeBatch", "批量 Encode")
                    .Replace("WordPieceEncodeFast", "EncodeFast")
                    .Replace("Qwen25EncodeSingle", "单条 Encode").Replace("Qwen25EncodeBatch", "批量 Encode");
                sb.AppendLine($"| {niceName} ops/s | {Fmt(jO)} | {Fmt(aO)} | {Ratio(aO, jO)} |");
            }
        }
        sb.AppendLine();
    }

    // ═══ 1.5 并发/截断/序列化 ═══
    sb.AppendLine("### 并发 / 截断 / 序列化 / EncodeCharOffsets");
    sb.AppendLine();

    var extraBenchmarks = new (string key, string label)[]
    {
        ("BpeConcurrent4T", "BPE 并发 (4T)"),
        ("BpeTruncation512", "BPE 截断 (maxLength=512)"),
        ("BpeTruncation128", "BPE 截断 (maxLength=128)"),
        ("SerializationLoad", "序列化 Load (FromJson)"),
        ("SerializationSave", "序列化 Save (ToJson)"),
        ("SerializationDeserialize", "序列化 Deserialize (FromJson)"),
        ("BpeEncodeCharOffsets", "BPE EncodeCharOffsets (batch)"),
        ("UnigramEncodeCharOffsets", "Unigram EncodeCharOffsets (batch)"),
    };

    if (hasRust)
    {
        sb.AppendLine("| 指标 | Rust (100%) | AOT | JIT |");
        sb.AppendLine("|------|-------------|-----|-----|");
        foreach (var (key, label) in extraBenchmarks)
        {
            jit.TryGetValue(key, out var j); aot.TryGetValue(key, out var a); rust.TryGetValue(key, out var r);
            double rO = GP(r, "OpsPerSec"), aO = GP(a, "OpsPerSec"), jO = GP(j, "OpsPerSec");
            string rustStr = rO > 0 ? $"{rO:N0}" : "—";
            sb.AppendLine($"| {label} | {rustStr} | {(aO > 0 ? $"{aO:N0}" : "—")} | {(jO > 0 ? $"{jO:N0}" : "—")} |");
        }
    }
    else
    {
        sb.AppendLine("| 指标 | JIT | AOT | AOT/JIT |");
        sb.AppendLine("|------|-----|-----|---------|");
        foreach (var (key, label) in extraBenchmarks)
        {
            jit.TryGetValue(key, out var j); aot.TryGetValue(key, out var a);
            double jO = GP(j, "OpsPerSec"), aO = GP(a, "OpsPerSec");
            sb.AppendLine($"| {label} ops/s | {Fmt(jO)} | {Fmt(aO)} | {Ratio(aO, jO)} |");
        }
    }
    sb.AppendLine();

    // ═══ 2. 训练性能 ═══
    sb.AppendLine("## 2. 训练性能");
    sb.AppendLine();

    var trainModels = new (string label, string[] tests)[]
    {
        ("BPE", new[] { "BpeTrainSmall", "BpeTrainBig" }),
        ("Unigram", new[] { "UnigramTrainSmall", "UnigramTrainBig" }),
        ("WordPiece", new[] { "WordPieceTrainSmall", "WordPieceTrainBig" }),
    };

    if (hasRust)
    {
        sb.AppendLine("| 模型 | 数据集 | Rust (100%) | AOT | JIT |");
        sb.AppendLine("|------|--------|-------------|-----|-----|");
        foreach (var (label, tests) in trainModels)
        {
            foreach (var key in tests)
            {
                jit.TryGetValue(key, out var j); aot.TryGetValue(key, out var a); rust.TryGetValue(key, out var r);
                double rM = GP(r, "ThroughputMBps"), aM = GP(a, "ThroughputMBps"), jM = GP(j, "ThroughputMBps");
                string size = key.Contains("Small") ? "small (128KB)" : "big (1.8MB)";
                sb.AppendLine($"| {label} | {size} | {rM:F1} | {PctFmtF(aM, rM)} | {PctFmtF(jM, rM)} |");
            }
        }
    }
    else
    {
        sb.AppendLine("| 模型 | 数据集 | JIT MiB/s | AOT MiB/s | AOT/JIT |");
        sb.AppendLine("|------|--------|-----------|-----------|---------|");
        foreach (var (label, tests) in trainModels)
        {
            foreach (var key in tests)
            {
                jit.TryGetValue(key, out var j); aot.TryGetValue(key, out var a);
                double jM = GP(j, "ThroughputMBps"), aM = GP(a, "ThroughputMBps");
                string size = key.Contains("Small") ? "small (128KB)" : "big (1.8MB)";
                sb.AppendLine($"| {label} | {size} | {jM:F1} | {aM:F1} | {Ratio(aM, jM)} |");
            }
        }
    }
    sb.AppendLine();

    // ═══ 3. 一致性 ═══
    sb.AppendLine("## 3. 数据一致性");
    sb.AppendLine();

    int GM(string label) =>
        cons.TryGetValue(label, out var v) && v.TryGetProperty("match", out var m) ? m.GetInt32() : -1;
    int GT(string label) =>
        cons.TryGetValue(label, out var v) && v.TryGetProperty("total", out var t) ? t.GetInt32() : 0;

    var consData = new Dictionary<string, (int jitVsAot, int rustVsJit, int rustVsAot, int total)>();
    foreach (var kind in new[] { "bpe", "wordpiece", "wordlevel", "qwen" })
    {
        int total = GT($"JIT vs AOT ({kind})");
        consData[kind] = (
            GM($"JIT vs AOT ({kind})"),
            GM($"Rust vs JIT ({kind})"),
            GM($"Rust vs AOT ({kind})"),
            total
        );
    }
    {
        int total = GT("JIT vs AOT (unigram)");
        consData["unigram"] = (
            GM("JIT vs AOT (unigram)"),
            GM("Rust vs JIT (unigram, C#-trained)"),
            GM("Rust vs AOT (unigram, C#-trained)"),
            total
        );
    }

    if (hasRust)
    {
        sb.AppendLine("| 模型 | JIT vs AOT | Rust vs JIT | Rust vs AOT | 状态 |");
        sb.AppendLine("|------|------------|-------------|-------------|------|");
        foreach (var kind in new[] { "bpe", "unigram", "wordpiece", "wordlevel", "qwen" })
        {
            var (jvA, rvJ, rvA, total) = consData[kind];
            string niceName = kind switch { "bpe" => "BPE", "unigram" => "Unigram", "wordpiece" => "WordPiece", "wordlevel" => "WordLevel", "qwen" => "Qwen2.5", _ => kind };
            bool allOk = jvA == total && rvJ == total && rvA == total;
            string FmtC(int v) => v == total ? "✅" : v < 0 ? "⏭" : $"❌ {v}/{total}";
            sb.AppendLine($"| {niceName} | {FmtC(jvA)} | {FmtC(rvJ)} | {FmtC(rvA)} | {(allOk ? "✅ PASS" : "❌ FAIL")} |");
        }
        sb.AppendLine();
        sb.AppendLine("### Unigram 交叉一致性");
        sb.AppendLine();
        sb.AppendLine("| 对比 | C#-trained | Rust-trained | 状态 |");
        sb.AppendLine("|------|------------|--------------|------|");
        {
            int jitVsAot = GM("JIT vs AOT (unigram)");
            int rustJitCs = GM("Rust vs JIT (unigram, C#-trained)");
            int rustAotCs = GM("Rust vs AOT (unigram, C#-trained)");
            int jitRustRs = GM("JIT vs Rust (unigram, Rust-trained)");
            int aotRustRs = GM("AOT vs Rust (unigram, Rust-trained)");
            int unigramTotal = GT("JIT vs AOT (unigram)");
            string FC(int v) => v == unigramTotal ? "✅" : v < 0 ? "⏭" : $"❌ {v}/{unigramTotal}";
            bool allU = jitVsAot == unigramTotal && rustJitCs == unigramTotal && rustAotCs == unigramTotal && jitRustRs == unigramTotal && aotRustRs == unigramTotal;
            sb.AppendLine($"| JIT vs AOT | {FC(jitVsAot)} | — | |");
            sb.AppendLine($"| Rust vs JIT | {FC(rustJitCs)} | {FC(jitRustRs)} | |");
            sb.AppendLine($"| Rust vs AOT | {FC(rustAotCs)} | {FC(aotRustRs)} | {(allU ? "✅ PASS" : "❌ FAIL")} |");
        }
    }
    else
    {
        sb.AppendLine("| 模型 | JIT vs AOT | 状态 |");
        sb.AppendLine("|------|------------|------|");
        foreach (var kind in new[] { "bpe", "unigram", "wordpiece", "wordlevel", "qwen" })
        {
            var (jvA, _, _, total) = consData[kind];
            string niceName = kind switch { "bpe" => "BPE", "unigram" => "Unigram", "wordpiece" => "WordPiece", "wordlevel" => "WordLevel", "qwen" => "Qwen2.5", _ => kind };
            bool ok = jvA == total;
            string status = ok ? "✅ PASS" : "❌ FAIL";
            string result = ok ? "✅" : $"❌ {jvA}/{total}";
            sb.AppendLine($"| {niceName} | {result} | {status} |");
        }
    }
    sb.AppendLine();

    // ═══ 4. 结论 ═══
    sb.AppendLine("## 4. 结论");
    sb.AppendLine();

    int crossUnigramTotal = GT("JIT vs Rust (unigram, Rust-trained)");
    bool allConsOk = consData.All(kv => kv.Value.jitVsAot == kv.Value.total
        && (kv.Value.rustVsJit == kv.Value.total || kv.Value.rustVsJit < 0)
        && (kv.Value.rustVsAot == kv.Value.total || kv.Value.rustVsAot < 0))
        && GM("JIT vs Rust (unigram, Rust-trained)") == crossUnigramTotal
        && GM("AOT vs Rust (unigram, Rust-trained)") == crossUnigramTotal;
    sb.AppendLine(allConsOk
        ? "✅ **一致性**：所有模型在 JIT / AOT / Rust 三端编码结果完全一致。"
        : "❌ **一致性**：存在跨端编码差异，需排查。");
    sb.AppendLine();

    if (hasRust)
    {
        var ratios = new List<(string label, double ratio)>();
        foreach (var (key, _) in new[] { ("BpeEncodeSingle", ""), ("UnigramEncodeSingle", ""), ("WordPieceEncodeSingle", ""), ("WordLevelEncodeSingle", ""), ("Qwen25EncodeSingle", "") })
        {
            jit.TryGetValue(key, out var j); aot.TryGetValue(key, out var a); rust.TryGetValue(key, out var r);
            double rO = GP(r, "OpsPerSec"), aO = GP(a, "OpsPerSec");
            if (rO > 0 && aO > 0) ratios.Add((key.Replace("EncodeSingle", ""), aO / rO));
        }
        if (ratios.Count > 0)
        {
            double avgRatio = ratios.Average(r => r.ratio);
            string perfLevel = avgRatio switch
            {
                >= 1.0 => "超越 Rust",
                >= 0.8 => "接近 Rust",
                >= 0.5 => "约为 Rust 的 50%~80%",
                _ => "低于 Rust 的 50%"
            };
            sb.AppendLine($"⚡ **编码性能**：AOT 平均约为 Rust 的 **{avgRatio:P0}**（{perfLevel}）。");
            sb.AppendLine();
            sb.AppendLine("| 模型 | AOT/Rust |");
            sb.AppendLine("|------|----------|");
            foreach (var (label, ratio) in ratios)
                sb.AppendLine($"| {label} | {ratio:F2}x |");
        }
    }
    sb.AppendLine();

    sb.AppendLine("---");
    sb.AppendLine("_报告自动生成 by Tokenizers Benchmark Suite_");

    File.WriteAllText(Path.Combine(outDir, "REPORT.md"), sb.ToString());
    Console.WriteLine($"  REPORT.md saved ({sb.Length} bytes)");
}
