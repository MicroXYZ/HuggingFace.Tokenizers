#!/usr/bin/env dotnet
// ═══════════════════════════════════════════════════════════════
// Tokenizers 基准测试 — Rust / .NET JIT / .NET AOT
// 用法: dotnet run --file tests/benchmarks/run.cs --rust-dir <path>
//       dotnet run --file tests/benchmarks/run.cs --rust-dir <path> --consistency-only
//
// 流程：
//   Phase 1: 生成随机测试数据
//   Phase 2: Rust 训练 tokenizer 模型（仅训练，保存为 JSON）
//   Phase 3: 三端分别加载模型 + 测试（一致性 + 性能）
//   Phase 4: 生成 REPORT.md
// ═══════════════════════════════════════════════════════════════

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;

// ── 参数解析 ──
var cliArgs = Environment.GetCommandLineArgs().Skip(1).ToList();

if (cliArgs.Contains("--help") || cliArgs.Contains("-h"))
{
    Console.WriteLine("用法: dotnet run --file tests/benchmarks/run.cs [选项]");
    Console.WriteLine();
    Console.WriteLine("选项:");
    Console.WriteLine("  --rust-dir <path>      指定 Rust tokenizers 源码目录（必须）");
    Console.WriteLine("  --consistency-only     仅运行数据一致性校验，跳过性能基准测试");
    Console.WriteLine("  --help, -h             显示此帮助信息");
    Console.WriteLine();
    Console.WriteLine("示例:");
    Console.WriteLine("  dotnet run --file tests/benchmarks/run.cs --rust-dir ../tokenizers");
    Console.WriteLine("  dotnet run --file tests/benchmarks/run.cs --rust-dir ../tokenizers --consistency-only");
    Environment.Exit(0);
}

bool consistencyOnly = cliArgs.Contains("--consistency-only");
string? rustDirArg = null;
for (int i = 0; i < cliArgs.Count - 1; i++)
    if (cliArgs[i] == "--rust-dir") rustDirArg = cliArgs[i + 1];

// ── 环境资源检测 ──
var sysInfo = CollectSystemInfo();
int cpuCores = sysInfo.CpuCores;
long totalMemoryMB = sysInfo.TotalMemoryMB;
long availableMemoryMB = sysInfo.AvailableMemoryMB;
string resourceProfile = totalMemoryMB switch
{
    < 1024 => "minimal",
    < 4096 => "low",
    _ => "full"
};
int maxConcurrentThreads = cpuCores;
bool canDownloadLargeTokenizers = resourceProfile is not "minimal";
bool canRun400KVocab = totalMemoryMB >= 4096;
Console.WriteLine($"▶ 资源检测: {resourceProfile} (RAM={totalMemoryMB}MB, 可用={availableMemoryMB}MB, CPU={cpuCores}核)");
Console.WriteLine($"  大tokenizer下载: {(canDownloadLargeTokenizers ? "✅" : "⏭ 跳过")} | 400K词表: {(canRun400KVocab ? "✅" : "⏭ 跳过")}");
Console.WriteLine();

string csRoot = Environment.CurrentDirectory;
string scriptDir = Path.Combine(csRoot, "tests", "benchmarks");
string benchProj = Path.Combine("tests", "HuggingFace.Tokenizers.Benchmarks");
string outDir = Path.Combine(scriptDir, "result");

if (rustDirArg is null)
{
    Console.WriteLine("❌ 必须通过 --rust-dir <path> 指定 Rust tokenizers 源码目录。");
    Console.WriteLine("   用法: dotnet run --file tests/benchmarks/run.cs --rust-dir ../tokenizers");
    Environment.Exit(1);
}

// ── 校验 --rust-dir 路径 ──
rustDirArg = ValidateAndFixRustDir(rustDirArg);

// ── Banner ──
string modeLabel = consistencyOnly ? "Consistency Only" : "Full Benchmark";
Console.WriteLine("╔══════════════════════════════════════════════════════════╗");
Console.WriteLine("║         Tokenizers Benchmark Suite                      ║");
Console.WriteLine("╠══════════════════════════════════════════════════════════╣");
Console.WriteLine($"║  Mode:        {modeLabel}");
Console.WriteLine($"║  C# Project:  {csRoot}");
Console.WriteLine($"║  Rust Ref:    {rustDirArg}");
Console.WriteLine($"║  Output:      {outDir}");
Console.WriteLine($"║  资源:        {resourceProfile} (RAM={totalMemoryMB}MB, CPU={cpuCores}核)");
Console.WriteLine("╚══════════════════════════════════════════════════════════╝");
Console.WriteLine();

Directory.CreateDirectory(outDir);

// ── 路径常量 ──
string rustTok = Path.Combine(rustDirArg!, "tokenizers");
string rustBenchData = Path.Combine(rustTok, "benches", "data");
string benchDataDir = Path.Combine(csRoot, benchProj, "data");
string jitDataDir = Path.Combine(csRoot, benchProj, "bin", "Release", "net10.0", "data");
string aotPublishDir = Path.Combine(outDir, "aot-publish");
string aotDataDir = Path.Combine(aotPublishDir, "data");
string cargoToml = Path.Combine(rustTok, "Cargo.toml");
string[] requiredModels = ["tokenizer-rust-bpe.json", "tokenizer-rust-unigram.json", "tokenizer-rust-wordpiece.json", "tokenizer-rust-wordlevel.json"];

// ── 清理旧文件 ──
CleanOldResults(outDir, rustBenchData, jitDataDir, aotDataDir);

// ═══════════════════════════════════════════════════════════════
//  Phase 1: 生成随机测试数据
// ═══════════════════════════════════════════════════════════════
Console.WriteLine("▶ Phase 1: 生成随机测试数据");
int seed = (int)(DateTimeOffset.UtcNow.ToUnixTimeSeconds() & 0x7FFFFFFF);
string smallFile = Path.Combine(outDir, "random-small.txt");
string bigFile = Path.Combine(outDir, "random-big.txt");

GenerateTestData(seed, smallFile, bigFile);

// ── 收集测试环境信息 + 写入 meta.json ──
var meta = new Dictionary<string, object>
{
    ["timestamp"] = DateTime.UtcNow.ToString("o"),
    ["seed"] = seed,
    ["os"] = sysInfo.Os,
    ["cpu"] = sysInfo.Cpu,
    ["memory"] = sysInfo.Memory,
    ["dotnet_version"] = sysInfo.DotnetVersion,
    ["dotnet_runtime"] = Environment.Version.ToString(),
    ["rustc_version"] = sysInfo.RustcVersion,
    ["cargo_version"] = sysInfo.CargoVersion,
    ["gcc_version"] = sysInfo.GccVersion,
    ["dotnet_settings"] = "Release, PublishAot=true, OptimizationPreference=Speed, StripSymbols=true, InvariantGlobalization=false",
    ["rust_settings"] = "release (lto=fat, codegen-units=1, target-cpu=native)",
    ["benchmark_config"] = "BPE/Unigram/WordPiece vocab=30000/8000/30000, small=128KB, big=1.8MB, encode=1000 lines × 100 iter, batch=1000 lines × 20 iter",
    ["resource_profile"] = resourceProfile,
    ["total_memory_mb"] = totalMemoryMB,
    ["cpu_cores"] = cpuCores,
    ["max_concurrent_threads"] = maxConcurrentThreads,
    ["skip_400k_vocab"] = !canRun400KVocab,
    ["consistency_only"] = consistencyOnly,
    ["data_languages"] = "英文/中文/日文/韩文/拉丁扩展/CombiningMarks/阿拉伯文/希伯来文/天城文/泰文/全角/零宽字符/Emoji/特殊符号"
};
var metaLines = meta.Select(kv =>
{
    var val = kv.Value is string s ? $"\"{s}\""
        : kv.Value is bool b ? (b ? "true" : "false")
        : kv.Value.ToString()?.ToLowerInvariant() ?? "null";
    return $"  \"{kv.Key}\": {val}";
});
File.WriteAllText(Path.Combine(outDir, "meta.json"), "{\n" + string.Join(",\n", metaLines) + "\n}");
Console.WriteLine($"  Seed: {seed}");
Console.WriteLine($"  small.txt: 128KB");
Console.WriteLine($"  big.txt: 1894KB");

Directory.CreateDirectory(benchDataDir);
File.Copy(smallFile, Path.Combine(benchDataDir, "small.txt"), true);
File.Copy(bigFile, Path.Combine(benchDataDir, "big.txt"), true);
Console.WriteLine();

// ═══════════════════════════════════════════════════════════════
//  Phase 2: Rust 训练 tokenizer 模型（仅训练，保存为 JSON）
// ═══════════════════════════════════════════════════════════════
Console.WriteLine("═══════════════════════════════════════");
Console.WriteLine(" Phase 2: Rust 训练 tokenizer 模型");
Console.WriteLine("═══════════════════════════════════════");

if (!Directory.Exists(rustTok))
{
    Console.WriteLine($"❌ Rust tokenizers 目录不存在: {rustTok}");
    Environment.Exit(1);
}
if (!CommandExists("cargo"))
{
    Console.WriteLine("❌ cargo 不在 PATH 中。请先 source \"$HOME/.cargo/env\" 或确保 Rust 已安装。");
    Environment.Exit(1);
}

// 拷贝测试数据和 bench 源码到 Rust 目录
Directory.CreateDirectory(rustBenchData);
File.Copy(smallFile, Path.Combine(rustBenchData, "small.txt"), true);
File.Copy(bigFile, Path.Combine(rustBenchData, "big.txt"), true);
CopyBenchFilesToRust(scriptDir, rustTok);

// 添加 bench 条目到 Cargo.toml 并运行训练
var cargoBackup = EnsureCargoBenchEntries(cargoToml, ["cross_lang_consistency"], consistencyOnly);
RunProcess("cargo", ["bench", "--bench", "cross_lang_consistency", "--features", "http"],
    workDir: rustTok, logFile: Path.Combine(outDir, "log-rust-train.txt"), timeoutMinutes: 60);
RestoreCargoToml(cargoToml, cargoBackup);

// 验证模型文件
Console.WriteLine("  ▶ 验证 Rust 训练的模型文件...");
foreach (var model in requiredModels)
{
    var modelPath = Path.Combine(rustBenchData, model);
    Console.WriteLine(File.Exists(modelPath)
        ? $"    ✅ {model} ({new FileInfo(modelPath).Length / 1024}KB)"
        : $"    ⚠ {model} 未生成");
}
Console.WriteLine();

// ═══════════════════════════════════════════════════════════════
//  Phase 3: 三端分别加载模型 + 测试
// ═══════════════════════════════════════════════════════════════
Console.WriteLine("═══════════════════════════════════════");
Console.WriteLine($" Phase 3: 三端加载模型 + {(consistencyOnly ? "一致性测试" : "基准测试")}");
Console.WriteLine("═══════════════════════════════════════");

Environment.SetEnvironmentVariable("BENCH_RESOURCE_PROFILE", resourceProfile);
Environment.SetEnvironmentVariable("BENCH_MAX_CONCURRENT", maxConcurrentThreads.ToString());
Environment.SetEnvironmentVariable("BENCH_SKIP_LARGE_DOWNLOADS", canDownloadLargeTokenizers ? "0" : "1");
Environment.SetEnvironmentVariable("BENCH_SKIP_400K_VOCAB", canRun400KVocab ? "0" : "1");

// 拷贝模型到 JIT 数据目录
Directory.CreateDirectory(jitDataDir);
foreach (var model in requiredModels)
    CopyIfExists(Path.Combine(rustBenchData, model), Path.Combine(jitDataDir, model));
File.Copy(smallFile, Path.Combine(jitDataDir, "small.txt"), true);
File.Copy(bigFile, Path.Combine(jitDataDir, "big.txt"), true);

// ── Phase 3.1: .NET JIT ──
Console.WriteLine("\n  ── Phase 3.1: .NET JIT ──");
RunDotnet(["build", benchProj, "-c", "Release"]);
RunDotnet(
    ["run", "--project", benchProj, "-c", "Release", "--no-build", "--",
     consistencyOnly ? "--consistency" : "--all", "--json"],
    logFile: Path.Combine(outDir, "log-jit.txt"));

// ── Phase 3.2: .NET AOT ──
Console.WriteLine("\n  ── Phase 3.2: .NET AOT ──");
RunDotnet(["build", "src/HuggingFace.Tokenizers.SourceGenerator/HuggingFace.Tokenizers.SourceGenerator.csproj", "-c", "Release"]);
RunDotnet(["publish", benchProj, "-c", "Release", "-r", "linux-x64", "--self-contained",
    "-p:PublishAot=true", "-p:OptimizationPreference=Speed", "-p:StripSymbols=true",
    $"-p:IlcInstructionSet={DetectBestIlcInstructionSet()}",
    "-p:StackTraceSupport=false", "-p:EventSourceSupport=false",
    "-p:MetadataUpdaterSupport=false", "-p:UseSystemResourceKeys=true",
    "-p:IlcFoldIdenticalMethodBodies=true", "-o", aotPublishDir]);

string aotBin = Path.Combine(aotPublishDir, "HuggingFace.Tokenizers.Benchmarks");
if (File.Exists(aotBin))
{
    Directory.CreateDirectory(aotDataDir);
    File.Copy(smallFile, Path.Combine(aotDataDir, "small.txt"), true);
    File.Copy(bigFile, Path.Combine(aotDataDir, "big.txt"), true);
    foreach (var model in requiredModels)
        CopyIfExists(Path.Combine(rustBenchData, model), Path.Combine(aotDataDir, model));

    RunProcess("file", [aotBin]);
    RunProcess(aotBin, [consistencyOnly ? "--consistency" : "--all", "--json"],
        logFile: Path.Combine(outDir, "log-aot.txt"));
}
else
{
    Console.WriteLine("  ⚠ AOT publish failed");
}

// ── Phase 3.3: Rust 推理测试 ──
Console.WriteLine("\n  ── Phase 3.3: Rust 推理测试（从文件加载） ──");
cargoBackup = EnsureCargoBenchEntries(cargoToml, ["cross_lang_inference", "cross_lang_perf"], consistencyOnly);

RunProcess("cargo", ["bench", "--bench", "cross_lang_inference", "--features", "http"],
    workDir: rustTok, logFile: Path.Combine(outDir, "log-rust-inference.txt"), timeoutMinutes: 60);

if (!consistencyOnly)
{
    RunProcess("cargo", ["bench", "--bench", "cross_lang_perf", "--features", "http"],
        workDir: rustTok, logFile: Path.Combine(outDir, "log-rust-perf.txt"), timeoutMinutes: 60);
}

RestoreCargoToml(cargoToml, cargoBackup);

// ── 收集所有输出到 result 目录 ──
CollectResults(jitDataDir, aotDataDir, rustBenchData, outDir);
Console.WriteLine();

// ═══════════════════════════════════════════════════════════════
//  Phase 4: 一致性比对 + 生成 REPORT.md
// ═══════════════════════════════════════════════════════════════
Console.WriteLine("═══════════════════════════════════════");
Console.WriteLine(" Phase 4: 一致性比对 + 生成报告");
Console.WriteLine("═══════════════════════════════════════");

var modelKinds = new[] { "bpe", "unigram", "wordpiece", "wordlevel", "qwen" };
var consResults = new List<(string label, int match, int total, bool ok)>();
bool allOk = true;

// JIT vs AOT
foreach (var kind in modelKinds)
    CompareAndRecord($"JIT vs AOT ({kind})", $"consistency-jit-{kind}.json", $"consistency-aot-{kind}.json");

// Rust vs JIT / Rust vs AOT
foreach (var kind in modelKinds)
{
    CompareAndRecord($"Rust vs JIT ({kind})", $"consistency-rust-{kind}.json", $"consistency-jit-{kind}.json");
    CompareAndRecord($"Rust vs AOT ({kind})", $"consistency-rust-{kind}.json", $"consistency-aot-{kind}.json");
}

void CompareAndRecord(string label, string fileA, string fileB)
{
    string pa = Path.Combine(outDir, fileA);
    string pb = Path.Combine(outDir, fileB);
    if (File.Exists(pa) && File.Exists(pb))
    {
        var (match, total) = CompareConsistency(pa, pb);
        bool ok = match == total;
        if (!ok) allOk = false;
        consResults.Add((label, match, total, ok));
        Console.WriteLine($"  {(ok ? "✅" : "❌")} {label}: {match}/{total}");
    }
    else
    {
        Console.WriteLine($"  ⏭ {label}: files not found");
    }
}

var consJson = "{\n" + string.Join(",\n", consResults.Select(r =>
    $"  \"{r.label}\": {{\"match\": {r.match}, \"total\": {r.total}, \"ok\": {r.ok.ToString().ToLower()}}}")) + "\n}";
File.WriteAllText(Path.Combine(outDir, "consistency-summary.json"), consJson);
Console.WriteLine(allOk ? "\n  ✅ 所有一致性检查通过" : "\n  ❌ 存在不一致");
Console.WriteLine();

GenerateReport(outDir);
Console.WriteLine();

Console.WriteLine("✅ 全部完成。结果在: " + outDir);
Console.WriteLine("   ├── REPORT.md");
Console.WriteLine("   ├── perf-jit.json / perf-aot.json / perf-rust.json");
Console.WriteLine("   └── consistency-*.json");

// ═══════════════════════════════════════════════════════════════
//  辅助方法
// ═══════════════════════════════════════════════════════════════

static string ValidateAndFixRustDir(string rustDirArg)
{
    string rustTokValidate = Path.Combine(rustDirArg, "tokenizers");
    string rustCargoToml = Path.Combine(rustTokValidate, "Cargo.toml");
    if (!File.Exists(rustCargoToml))
    {
        string altCargoToml = Path.Combine(rustDirArg, "Cargo.toml");
        if (File.Exists(altCargoToml))
        {
            string parent = Path.GetDirectoryName(rustDirArg)!;
            if (File.Exists(Path.Combine(parent, "tokenizers", "Cargo.toml")))
            {
                Console.WriteLine($"⚠ --rust-dir 自动修正: {rustDirArg} → {parent}");
                return parent;
            }
        }
        Console.WriteLine($"❌ --rust-dir 路径无效: {rustDirArg}");
        Console.WriteLine($"   找不到 {rustCargoToml}");
        Environment.Exit(1);
    }
    return rustDirArg;
}

static void CleanOldResults(string outDir, string rustTokData, string jitDataDir, string aotCleanupDir)
{
    foreach (var oldFile in Directory.GetFiles(outDir, "consistency-*.json")) File.Delete(oldFile);
    foreach (var oldFile in Directory.GetFiles(outDir, "perf-*.json")) File.Delete(oldFile);
    var modelFiles = new[] {
        "unigram-dotnet-trained.json", "unigram-rust-trained.json",
        "tokenizer-rust-bpe.json", "tokenizer-rust-unigram.json",
        "tokenizer-rust-wordpiece.json", "tokenizer-rust-wordlevel.json"
    };
    foreach (var dir in new[] { outDir, rustTokData, jitDataDir, aotCleanupDir })
        foreach (var f in modelFiles)
        {
            var p = Path.Combine(dir, f);
            if (File.Exists(p)) File.Delete(p);
        }
}

static void CopyBenchFilesToRust(string scriptDir, string rustTok)
{
    // 拷贝 bench 源码（cross_lang_consistency.rs、cross_lang_inference.rs、cross_lang_perf.rs、cross_lang/）
    foreach (var rsFile in new[] { "cross_lang_consistency.rs", "cross_lang_inference.rs", "cross_lang_perf.rs" })
    {
        var src = Path.Combine(scriptDir, rsFile);
        if (File.Exists(src))
            File.Copy(src, Path.Combine(rustTok, "benches", rsFile), true);
    }
    var crossLangSrc = Path.Combine(scriptDir, "cross_lang");
    var crossLangDst = Path.Combine(rustTok, "benches", "cross_lang");
    if (Directory.Exists(crossLangSrc))
    {
        Directory.CreateDirectory(crossLangDst);
        foreach (var file in Directory.GetFiles(crossLangSrc))
            File.Copy(file, Path.Combine(crossLangDst, Path.GetFileName(file)), true);
    }
    // 清理旧的 bench 二进制
    var depsDir = Path.Combine(rustTok, "target", "release", "deps");
    if (Directory.Exists(depsDir))
        foreach (var f in Directory.GetFiles(depsDir, "cross_lang_consistency-*"))
            File.Delete(f);
}

/// <summary>
/// 确保 Cargo.toml 包含指定的 bench 条目，返回备份用于恢复。
/// </summary>
static string? EnsureCargoBenchEntries(string cargoToml, string[] benchNames, bool consistencyOnly)
{
    if (!File.Exists(cargoToml)) return null;
    var backup = File.ReadAllText(cargoToml);
    var toAdd = benchNames.Where(name => !backup.Contains(name)).ToList();
    // cross_lang_perf 在 consistency-only 模式下不添加
    if (consistencyOnly) toAdd = toAdd.Where(n => n != "cross_lang_perf").ToList();
    if (toAdd.Count > 0)
    {
        var sb = new StringBuilder();
        foreach (var entry in toAdd)
            sb.Append($"\n[[bench]]\nname = \"{entry}\"\nharness = false\n");
        File.AppendAllText(cargoToml, sb.ToString());
    }
    return backup;
}

static void RestoreCargoToml(string cargoToml, string? backup)
{
    if (backup is not null && File.Exists(cargoToml))
        File.WriteAllText(cargoToml, backup);
}

/// <summary>
/// 收集 JIT/AOT/Rust 的输出文件到 result 目录
/// </summary>
static void CollectResults(string jitDataDir, string aotDataDir, string rustBenchData, string outDir)
{
    // JIT 结果
    CopyIfExists(Path.Combine(jitDataDir, "benchmark-jit.json"), Path.Combine(outDir, "perf-jit.json"));
    foreach (var kind in new[] { "bpe", "unigram", "wordpiece", "wordlevel", "qwen2.5" })
        CopyIfExists(Path.Combine(jitDataDir, $"consistency-dotnet-{kind}.json"),
                     Path.Combine(outDir, $"consistency-jit-{kind.Replace("2.5", "")}.json"));

    // AOT 结果
    CopyIfExists(Path.Combine(aotDataDir, "benchmark-aot.json"), Path.Combine(outDir, "perf-aot.json"));
    foreach (var kind in new[] { "bpe", "unigram", "wordpiece", "wordlevel", "qwen2.5" })
        CopyIfExists(Path.Combine(aotDataDir, $"consistency-dotnet-{kind}.json"),
                     Path.Combine(outDir, $"consistency-aot-{kind.Replace("2.5", "")}.json"));

    // Rust 结果
    CopyIfExists(Path.Combine(rustBenchData, "perf-rust.json"), Path.Combine(outDir, "perf-rust.json"));
    foreach (var kind in new[] { "bpe", "unigram", "wordpiece", "wordlevel", "qwen2.5" })
        CopyIfExists(Path.Combine(rustBenchData, $"consistency-rust-{kind}.json"),
                     Path.Combine(outDir, $"consistency-rust-{kind.Replace("2.5", "")}.json"));
}

static void GenerateTestData(int seed, string smallPath, string bigPath)
{
    var words = new[]
    {
        "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
        "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would",
        "shall", "should", "may", "might", "must", "can", "could",
        "training", "inference", "neural", "network", "attention",
        "transformer", "encoding", "decoding", "tokenizer", "vocabulary",
        "embedding", "gradient", "optimizer", "learning", "rate", "batch",
        "epoch", "loss", "accuracy", "model", "data", "input", "output",
        "hidden", "layer", "activation", "softmax", "dropout", "normalization",
        "algorithm", "processing", "machine", "deep", "subword", "byte",
        "pair", "merge", "frequency", "sequence", "token", "piece",
        "unigram", "bpe", "wordpiece", "sentencepiece", "huggingface",
        "natural", "language", "understanding", "generation", "translation",
        "summarization", "classification", "sentiment", "analysis",
        "hello", "world", "test", "example", "sample", "random",
        "array", "string", "integer", "float", "boolean", "function",
        "class", "method", "variable", "constant", "loop", "condition",
        "import", "export", "module", "package", "library", "framework",
        "essential", "processing", "inference", "unigram", "suffix",
        "network", "encoding", "subword", "tokenizers", "models",
    };

    var chineseWords = new[]
    {
        "你好", "世界", "测试", "数据", "模型", "训练", "推理", "编码", "解码", "分词",
        "自然语言处理", "机器学习", "深度学习", "神经网络", "注意力机制", "变换器",
        "向量", "嵌入", "梯度", "优化器", "损失函数", "准确率", "批次", "迭代",
        "人工智能", "大语言模型", "预训练", "微调", "标记化", "子词", "词表",
        "今天天气真好", "我喜欢编程", "这是一段测试文本", "中文分词是个难题",
        "中华人民共和国", "科学技术是第一生产力", "机器翻译技术日趋成熟",
        "深度学习在自然语言处理领域取得了显著进展", "基于Transformer的模型已成为主流",
        "人工智能技术正在改变我们的生活方式", "数据驱动的方法论越来越受到重视",
    };

    var japaneseWords = new[]
    {
        "こんにちは", "世界", "テスト", "データ", "モデル", "自然言語処理",
        "機械学習", "深層学習", "トークナイザー", "こんにちは世界",
    };
    var koreanWords = new[]
    {
        "안녕하세요", "세계", "테스트", "데이터", "모델", "자연어 처리",
        "기계학습", "딥러닝", "토크나이저",
    };

    var latinAccentedWords = new[]
    {
        "café", "résumé", "naïve", "crème", "brûlée", "déjà", "voilà", "élève",
        "français", "çoğırdı", "où", "être", "avoir", "mémoire", "für",
        "Straße", "Über", "München", "Köln", "Größe", "schön", "für", "wörter",
        "Ärger", "Ökonomie", "Übung", "ß", "äöü",
        "señor", "año", "niño", "mañana", "corazón", "acción", "información",
        " São Paulo", "não", "coração", "três", "até", "você",
        "Türkiye", "İstanbul", "Çeşme", "Ğüzel", "şeker", "öğrenci",
        "Ångström", "Malmö", "Göteborg", "Øresund", "Björk", "Søren",
        "fjord", "smørrebrød", "hygge",
        "Praha", "Česko", "Dvořák", "Łódź", "Kraków", "Budapest",
        "Széchenyi", "űveg", "őszinte",
        "Việt Nam", "xin chào", "cảm ơn", "Hà Nội", "Đà Nẵng",
        "café résumé naïve", "Über Straße Größe", "señor año niño",
        "Ångström Göteborg fjord", "Việt Nam cảm ơn",
    };

    var combiningMarkWords = new[]
    {
        "e\u0301", "a\u0308", "n\u0303", "o\u0302", "u\u0308", "c\u0327", "a\u030a",
        "e\u0300e\u0301e\u0302e\u0308",
    };

    var arabicWords = new[]
    {
        "مرحبا", "السلام", "عليكم", "عالم", "اختبار", "بيانات",
        "ذكاء", "اصطناعي", "تعلم", "آلة", "شبكة", "عصبونية",
        "اللغة", "العربية", "معالجة", "طبيعية",
    };

    var hebrewWords = new[]
    {
        "שלום", "עולם", "בדיקה", "נתונים", "בינה", "מלאכותית",
        "למידה", "מכונה", "עברית", "שפת",
    };

    var devanagariWords = new[]
    {
        "नमस्ते", "दुनिया", "परीक्षा", "डेटा", "मॉडल",
        "कृत्रिम", "बुद्धिमत्ता", "मशीन", "लर्निंग", "हिन्दी",
        "भाषा", "प्रसंस्करण", "स्वाभाविक",
    };

    var thaiWords = new[]
    {
        "สวัสดี", "โลก", "ทดสอบ", "ข้อมูล", "โมเดล",
        "ปัญญา", "ประดิษฐ์", "เรียนรู้", "ภาษาไทย",
    };

    var fullwidthWords = new[]
    {
        "ＡＢＣＤ", "Ｈｅｌｌｏ", "Ｗｏｒｌｄ", "０１２３",
        "Ｔｅｓｔ", "ｄａｔａ", "ｍｏｄｅｌ",
    };

    var longConcatWords = new[]
    {
        "supercalifragilisticexpialidocious",
        "Pneumonoultramicroscopicsilicovolcanoconiosis",
        "antidisestablishmentarianism",
        "中华人民共和国国务院总理",
        "自然语言处理是人工智能的重要方向",
        "この文章は日本語のテストです",
    };

    var zeroWidthWords = new[]
    {
        "hello\u200Bworld", "hello\u200Cworld", "hello\u200Dworld",
        "hello\uFEFFworld", "test\u00ADsoft-hyphen",
    };

    var emojis = new[]
    {
        "😀", "😂", "🤣", "😍", "🤔", "👍", "👎", "🔥", "💯", "✅",
        "🎉", "🚀", "💡", "⚡", "🤖", "🧠", "📊", "💻", "🔧", "🎯",
        "👨‍💻", "👩‍🔬", "🇨🇳", "🇺🇸", "🇯🇵", "🏳️‍🌈", "👨‍👩‍👧‍👦",
        "🏻", "🏼", "🏽", "🏾", "🏿",
    };

    var specialSymbols = new[]
    {
        "①②③④⑤", "→←↑↓", "★☆♠♣♥♦", "±×÷√∞∑∏∫",
        "$€£¥₹₽", "©®™§¶", "≈≠≤≥≡≪≫", "【】「」『』〈〉《》",
        "…—–·•", "‰‱°℃℉", "∀∃∅∈∉⊂⊃∪∩",
        "\t", "  ", "   ",
    };

    var punctuations = new[] { "", "", "", "", ".", ",", "!", "?", ";", ":", "...", "。", "，", "！", "？", "、" };

    var rng = new Random(seed);

    string PickWord()
    {
        double r = rng.NextDouble();
        if (r < 0.35) return words[rng.Next(words.Length)];
        if (r < 0.50) return chineseWords[rng.Next(chineseWords.Length)];
        if (r < 0.57) return rng.NextDouble() < 0.5 ? japaneseWords[rng.Next(japaneseWords.Length)] : koreanWords[rng.Next(koreanWords.Length)];
        if (r < 0.69) return latinAccentedWords[rng.Next(latinAccentedWords.Length)];
        if (r < 0.71) return combiningMarkWords[rng.Next(combiningMarkWords.Length)];
        if (r < 0.74) return arabicWords[rng.Next(arabicWords.Length)];
        if (r < 0.76) return hebrewWords[rng.Next(hebrewWords.Length)];
        if (r < 0.78) return devanagariWords[rng.Next(devanagariWords.Length)];
        if (r < 0.80) return thaiWords[rng.Next(thaiWords.Length)];
        if (r < 0.82) return fullwidthWords[rng.Next(fullwidthWords.Length)];
        if (r < 0.84) return longConcatWords[rng.Next(longConcatWords.Length)];
        if (r < 0.85) return zeroWidthWords[rng.Next(zeroWidthWords.Length)];
        if (r < 0.92) return emojis[rng.Next(emojis.Length)];
        return specialSymbols[rng.Next(specialSymbols.Length)];
    }

    string GenLine(int minW, int maxW)
    {
        int n = rng.Next(minW, maxW + 1);
        var sb = new StringBuilder(n * 16);
        for (int i = 0; i < n; i++)
        {
            if (i > 0) sb.Append(' ');
            sb.Append(PickWord());
            if (rng.NextDouble() < 0.3)
                sb.Append(punctuations[rng.Next(punctuations.Length)]);
        }
        return sb.ToString();
    }

    using (var w = new StreamWriter(smallPath))
    {
        long size = 0;
        while (size < 128 * 1024) { var l = GenLine(5, 20); w.WriteLine(l); size += l.Length + 1; }
    }

    using (var w = new StreamWriter(bigPath))
    {
        long size = 0;
        while (size < 1894 * 1024) { var l = GenLine(5, 30); w.WriteLine(l); size += l.Length + 1; }
    }
}

static (int match, int total) CompareConsistency(string pathA, string pathB)
{
    static List<uint> GetIds(JsonElement e)
    {
        if (e.TryGetProperty("ids", out var v) || e.TryGetProperty("Ids", out v))
            if (v.ValueKind == JsonValueKind.Array)
                return v.EnumerateArray().Select(x => x.GetUInt32()).ToList();
        return [];
    }

    static string GetDecoded(JsonElement e)
    {
        if (e.TryGetProperty("decoded", out var v) || e.TryGetProperty("Decoded", out v))
            return v.GetString() ?? "";
        return "";
    }

    static List<string> GetTokens(JsonElement e)
    {
        if (e.TryGetProperty("tokens", out var v) || e.TryGetProperty("Tokens", out v))
            if (v.ValueKind == JsonValueKind.Array)
                return v.EnumerateArray().Select(x => x.GetString() ?? "").ToList();
        return [];
    }

    static string GetInput(JsonElement e)
    {
        if (e.TryGetProperty("input", out var v) || e.TryGetProperty("Input", out v))
            return v.GetString() ?? "";
        return "";
    }

    using var docA = JsonDocument.Parse(File.ReadAllText(pathA));
    using var docB = JsonDocument.Parse(File.ReadAllText(pathB));
    var da = docA.RootElement.EnumerateArray().ToArray();
    var db = docB.RootElement.EnumerateArray().ToArray();

    if (da.Length != db.Length)
    {
        Console.WriteLine($"  ❌ 条目数不一致: {Path.GetFileName(pathA)}={da.Length}, {Path.GetFileName(pathB)}={db.Length}");
        return (0, Math.Max(da.Length, db.Length));
    }

    int n = da.Length;
    int match = 0;
    for (int i = 0; i < n; i++)
    {
        var aIds = GetIds(da[i]);
        var bIds = GetIds(db[i]);
        var aDec = GetDecoded(da[i]);
        var bDec = GetDecoded(db[i]);
        var aTok = GetTokens(da[i]);
        var bTok = GetTokens(db[i]);
        var aInput = GetInput(da[i]);
        var bInput = GetInput(db[i]);

        if (aInput != bInput)
        {
            Console.WriteLine($"  ⚠ 第 {i} 条 input 不对齐: \"{aInput}\" vs \"{bInput}\"");
            continue;
        }

        if (aIds.Count == bIds.Count && aIds.SequenceEqual(bIds) && aDec == bDec && aTok.SequenceEqual(bTok))
            match++;
    }
    return (match, n);
}

static void GenerateReport(string outDir)
{
    var modelKinds = new[] { "bpe", "unigram", "wordpiece", "wordlevel", "qwen" };
    static Dictionary<string, JsonElement> LoadJ(string p)
    {
        try
        {
            using var doc = JsonDocument.Parse(File.ReadAllText(p));
            return doc.RootElement.EnumerateObject().ToDictionary(prop => prop.Name, prop => prop.Value.Clone());
        }
        catch { return new(); }
    }

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

    // niceName 映射表
    var niceNames = new Dictionary<string, string>
    {
        ["BpeEncodeSingle"] = "单条 Encode", ["BpeEncodeBatch"] = "批量 Encode (lines/s)", ["BpeEncodeFast"] = "EncodeFast",
        ["UnigramEncodeSingle"] = "单条 Encode", ["UnigramEncodeBatch"] = "批量 Encode (lines/s)",
        ["WordPieceEncodeSingle"] = "单条 Encode", ["WordPieceEncodeBatch"] = "批量 Encode (lines/s)", ["WordPieceEncodeFast"] = "EncodeFast",
        ["WordLevelEncodeSingle"] = "单条 Encode", ["WordLevelEncodeBatch"] = "批量 Encode (lines/s)",
        ["Qwen25EncodeSingle"] = "单条 Encode", ["Qwen25EncodeBatch"] = "批量 Encode (lines/s)",
    };

    var sb = new StringBuilder();
    sb.AppendLine("# Tokenizers 基准测试报告");
    sb.AppendLine();

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

    sb.AppendLine("## 测试配置");
    sb.AppendLine();
    sb.AppendLine($"`{MS("benchmark_config")}`");
    sb.AppendLine();

    sb.AppendLine("## 测试流程");
    sb.AppendLine();
    sb.AppendLine("```");
    sb.AppendLine("Phase 1: 生成随机测试数据");
    sb.AppendLine("Phase 2: Rust 训练 tokenizer 模型（BPE/Unigram/WordPiece），保存为 JSON");
    sb.AppendLine("Phase 3: 三端分别加载模型 + 测试");
    sb.AppendLine("  3.1: .NET JIT（加载 Rust 训练的模型）");
    sb.AppendLine("  3.2: .NET AOT（加载 Rust 训练的模型）");
    sb.AppendLine("  3.3: Rust（从文件重新加载模型）");
    sb.AppendLine("Phase 4: 一致性比对 + 生成报告");
    sb.AppendLine("```");
    sb.AppendLine();

    static string PctFmt(double value, double rustBaseline)
    {
        if (value <= 0) return "-";
        if (rustBaseline <= 0) return $"{value:N0}";
        int pct = (int)Math.Round(value / rustBaseline * 100);
        return $"{value:N0} ({pct}%)";
    }

    // ── 编码性能 ──
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
                string name = niceNames.GetValueOrDefault(key, key);
                sb.AppendLine($"| {name} | {(key.Contains("EncodeFast") ? "—" : $"{rO:N0}")} | {PctFmt(aO, rO)} | {PctFmt(jO, rO)} |");
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
                string name = niceNames.GetValueOrDefault(key, key);
                sb.AppendLine($"| {name} ops/s | {Fmt(jO)} | {Fmt(aO)} | {Ratio(aO, jO)} |");
            }
        }
        sb.AppendLine();
    }

    // ── 并发 / 截断 / 序列化 / EncodeCharOffsets ──
    var comparableBenchmarks = new (string key, string label)[]
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

    sb.AppendLine("### 并发 / 截断 / 序列化 / EncodeCharOffsets");
    sb.AppendLine();

    if (hasRust)
    {
        sb.AppendLine("| 指标 | Rust (100%) | AOT | JIT |");
        sb.AppendLine("|------|-------------|-----|-----|");
        foreach (var (key, label) in comparableBenchmarks)
        {
            jit.TryGetValue(key, out var j); aot.TryGetValue(key, out var a); rust.TryGetValue(key, out var r);
            double rO = GP(r, "OpsPerSec"), aO = GP(a, "OpsPerSec"), jO = GP(j, "OpsPerSec");
            sb.AppendLine($"| {label} | {(rO > 0 ? $"{rO:N0}" : "—")} | {PctFmt(aO, rO)} | {PctFmt(jO, rO)} |");
        }
    }
    else
    {
        sb.AppendLine("| 指标 | JIT | AOT | AOT/JIT |");
        sb.AppendLine("|------|-----|-----|---------|");
        foreach (var (key, label) in comparableBenchmarks)
        {
            jit.TryGetValue(key, out var j); aot.TryGetValue(key, out var a);
            double jO = GP(j, "OpsPerSec"), aO = GP(a, "OpsPerSec");
            sb.AppendLine($"| {label} ops/s | {Fmt(jO)} | {Fmt(aO)} | {Ratio(aO, jO)} |");
        }
    }
    sb.AppendLine();

    // ── 数据一致性 ──
    sb.AppendLine("## 2. 数据一致性");
    sb.AppendLine();
    sb.AppendLine("> 所有平台使用 Rust 统一训练的模型，验证编码/解码结果一致。");
    sb.AppendLine();

    int GM(string label) =>
        cons.TryGetValue(label, out var v) && v.TryGetProperty("match", out var m) ? m.GetInt32() : -1;
    int GT(string label) =>
        cons.TryGetValue(label, out var v) && v.TryGetProperty("total", out var t) ? t.GetInt32() : 0;

    var consData = new Dictionary<string, (int jitVsAot, int rustVsJit, int rustVsAot, int total)>();
    foreach (var kind in modelKinds)
    {
        int total = GT($"JIT vs AOT ({kind})");
        consData[kind] = (
            GM($"JIT vs AOT ({kind})"),
            GM($"Rust vs JIT ({kind})"),
            GM($"Rust vs AOT ({kind})"),
            total
        );
    }

    if (hasRust)
    {
        sb.AppendLine("| 模型 | JIT vs AOT | Rust vs JIT | Rust vs AOT | 状态 |");
        sb.AppendLine("|------|------------|-------------|-------------|------|");
        foreach (var kind in modelKinds)
        {
            var (jvA, rvJ, rvA, total) = consData[kind];
            string niceName = kind switch { "bpe" => "BPE", "unigram" => "Unigram", "wordpiece" => "WordPiece", "wordlevel" => "WordLevel", "qwen" => "Qwen2.5", _ => kind };
            bool ok = jvA == total && rvJ == total && rvA == total;
            string FmtC(int v) => v == total ? "✅" : v < 0 ? "⏭" : $"❌ {v}/{total}";
            sb.AppendLine($"| {niceName} | {FmtC(jvA)} | {FmtC(rvJ)} | {FmtC(rvA)} | {(ok ? "✅ PASS" : "❌ FAIL")} |");
        }
    }
    else
    {
        sb.AppendLine("| 模型 | JIT vs AOT | 状态 |");
        sb.AppendLine("|------|------------|------|");
        foreach (var kind in modelKinds)
        {
            var (jvA, _, _, total) = consData[kind];
            string niceName = kind switch { "bpe" => "BPE", "unigram" => "Unigram", "wordpiece" => "WordPiece", "wordlevel" => "WordLevel", "qwen" => "Qwen2.5", _ => kind };
            bool ok = jvA == total;
            sb.AppendLine($"| {niceName} | {(ok ? "✅" : $"❌ {jvA}/{total}")} | {(ok ? "✅ PASS" : "❌ FAIL")} |");
        }
    }
    sb.AppendLine();

    // ── 结论 ──
    sb.AppendLine("## 3. 结论");
    sb.AppendLine();

    bool allConsOk = consData.All(kv => kv.Value.jitVsAot == kv.Value.total
        && (kv.Value.rustVsJit == kv.Value.total || kv.Value.rustVsJit < 0)
        && (kv.Value.rustVsAot == kv.Value.total || kv.Value.rustVsAot < 0));
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

// ── 环境信息采集（一次性） ──

static SystemInfo CollectSystemInfo()
{
    string os = "unknown", cpu = "unknown", memory = "unknown";
    int cores = Environment.ProcessorCount;
    long totalMB = 4096, availMB = 2048;

    try
    {
        if (File.Exists("/etc/os-release"))
            foreach (var line in File.ReadLines("/etc/os-release"))
                if (line.StartsWith("PRETTY_NAME="))
                { os = line["PRETTY_NAME=".Length..].Trim('"'); break; }

        if (File.Exists("/proc/cpuinfo"))
        {
            string? modelName = null;
            int c = 0;
            foreach (var line in File.ReadLines("/proc/cpuinfo"))
            {
                if (line.StartsWith("model name") && modelName is null) modelName = line.Split(':').Last().Trim();
                if (line.StartsWith("processor")) c++;
            }
            if (modelName is not null) { cpu = $"{modelName} ({c} cores)"; cores = c; }
        }

        if (File.Exists("/proc/meminfo"))
            foreach (var line in File.ReadLines("/proc/meminfo"))
            {
                if (line.StartsWith("MemTotal:"))
                {
                    memory = line.Split(':', 2)[1].Trim();
                    var numStr = new string(line.SkipWhile(c => !char.IsDigit(c)).TakeWhile(c => char.IsDigit(c) || c == '.').ToArray());
                    if (long.TryParse(numStr, out long kb)) totalMB = kb / 1024;
                }
                if (line.StartsWith("MemAvailable:"))
                {
                    var numStr = new string(line.SkipWhile(c => !char.IsDigit(c)).TakeWhile(c => char.IsDigit(c) || c == '.').ToArray());
                    if (long.TryParse(numStr, out long kb)) availMB = kb / 1024;
                }
            }
    }
    catch { }

    return new SystemInfo(
        Os: os != "unknown" ? os : Environment.OSVersion.ToString(),
        Cpu: cpu != "unknown" ? cpu : $"{cores} cores",
        Memory: memory,
        DotnetVersion: RunCmd("dotnet", "--version"),
        RustcVersion: RunCmd("rustc", "--version"),
        CargoVersion: RunCmd("cargo", "--version"),
        GccVersion: RunCmd("gcc", "--version").Split('\n').FirstOrDefault() ?? "unknown",
        CpuCores: cores,
        TotalMemoryMB: totalMB,
        AvailableMemoryMB: availMB
    );
}

static bool CommandExists(string cmd)
{
    try
    {
        using var p = Process.Start(new ProcessStartInfo("which", cmd) { RedirectStandardOutput = true, UseShellExecute = false })!;
        p.WaitForExit(3000);
        return p.ExitCode == 0;
    }
    catch { return false; }
}

static void CopyIfExists(string src, string dst)
{
    if (File.Exists(src)) File.Copy(src, dst, true);
}

static void RunProcess(string file, string[] args, string? workDir = null, string? logFile = null, bool append = false, long timeoutMinutes = 30)
{
    var psi = new ProcessStartInfo(file, args)
    {
        UseShellExecute = false,
        WorkingDirectory = workDir ?? Environment.CurrentDirectory,
        RedirectStandardOutput = logFile is not null,
        RedirectStandardError = logFile is not null
    };

    using var p = Process.Start(psi)!;
    if (logFile is not null)
    {
        var output = p.StandardOutput.ReadToEnd();
        var err = p.StandardError.ReadToEnd();
        if (!p.WaitForExit((int)(timeoutMinutes * 60_000)))
        {
            p.Kill(true);
            Console.WriteLine($"  ❌ 进程超时 ({timeoutMinutes}min): {file} {string.Join(" ", args)}");
            Environment.Exit(1);
        }
        if (append) File.AppendAllText(logFile, output + err);
        else File.WriteAllText(logFile, output + err);
    }
    else
    {
        if (!p.WaitForExit((int)(timeoutMinutes * 60_000)))
        {
            p.Kill(true);
            Console.WriteLine($"  ❌ 进程超时 ({timeoutMinutes}min): {file} {string.Join(" ", args)}");
            Environment.Exit(1);
        }
    }

    if (p.ExitCode != 0)
    {
        Console.WriteLine($"  ❌ 进程退出码 {p.ExitCode}: {file} {string.Join(" ", args)}");
        Environment.Exit(p.ExitCode);
    }
}

static void RunDotnet(string[] args, string? logFile = null, long timeoutMinutes = 30)
{
    var psi = new ProcessStartInfo("dotnet", args)
    {
        UseShellExecute = false,
        RedirectStandardOutput = logFile is not null,
        RedirectStandardError = logFile is not null
    };

    using var p = Process.Start(psi)!;
    if (logFile is not null)
    {
        var output = p.StandardOutput.ReadToEnd();
        var err = p.StandardError.ReadToEnd();
        if (!p.WaitForExit((int)(timeoutMinutes * 60_000)))
        {
            p.Kill(true);
            Console.WriteLine($"  ❌ dotnet 超时 ({timeoutMinutes}min): dotnet {string.Join(" ", args)}");
            Environment.Exit(1);
        }
        var lines = (output + err).Split('\n');
        foreach (var line in lines.TakeLast(5)) Console.WriteLine(line);
        File.WriteAllText(logFile, output + err);
    }
    else
    {
        if (!p.WaitForExit((int)(timeoutMinutes * 60_000)))
        {
            p.Kill(true);
            Console.WriteLine($"  ❌ dotnet 超时 ({timeoutMinutes}min): dotnet {string.Join(" ", args)}");
            Environment.Exit(1);
        }
    }

    if (p.ExitCode != 0)
    {
        Console.WriteLine($"  ❌ dotnet 退出码 {p.ExitCode}: dotnet {string.Join(" ", args)}");
        Environment.Exit(p.ExitCode);
    }
}

static string RunCmd(string file, string args, int timeoutMs = 5000)
{
    try
    {
        using var p = Process.Start(new ProcessStartInfo(file, args)
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false
        })!;
        var output = p.StandardOutput.ReadToEnd();
        p.WaitForExit(timeoutMs);
        return output.Trim();
    }
    catch { return ""; }
}

static string DetectBestIlcInstructionSet()
{
    try
    {
        if (File.Exists("/proc/cpuinfo"))
        {
            string? flags = null;
            foreach (var line in File.ReadLines("/proc/cpuinfo"))
                if (line.StartsWith("flags")) { flags = line; break; }
            if (flags is not null)
            {
                if (flags.Contains("avx512f") && flags.Contains("avx512bw")) return "avx512";
                if (flags.Contains("avx2")) return "avx2";
                if (flags.Contains("avx")) return "avx";
                if (flags.Contains("sse4.2")) return "sse4.2";
            }
        }
    }
    catch { }
    if (System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture ==
        System.Runtime.InteropServices.Architecture.Arm64)
        return "genericarm64";
    return "x86-64";
}

record SystemInfo(string Os, string Cpu, string Memory, string DotnetVersion,
                  string RustcVersion, string CargoVersion, string GccVersion,
                  int CpuCores, long TotalMemoryMB, long AvailableMemoryMB);

