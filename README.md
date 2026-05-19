# HuggingFace.Tokenizers

![.NET 10](https://img.shields.io/badge/.NET-10-purple)
![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue)

[HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)（Rust 版本）的纯 C# 实现，面向 **.NET 10**。

提供完整的分词管道 — Normalizer → PreTokenizer → Model → PostProcessor → Decoder — 零外部 NuGet 依赖，可直接集成到任何 .NET 10 项目中。输入输出与 Rust 版本保持一致。

**适用场景：**

- 在 .NET 生态中本地运行 HuggingFace 预训练模型的分词器，无需 Python/Rust 互操作
- 需要 AOT 兼容的 Native AOT 部署场景
- LLM 推理服务中的流式解码（逐 token 输出）

本项目虽然是用来验证AI的，但是所有核心模块已完成开发。训练器功能已移除，本项目仅提供推理能力。

部分性能已经进行了优化具体参考[基准测试](./tests/benchmarks/README.md)

> **注意：** 本项目仅实现推理（Encode/Decode），不包含训练器（Trainer）功能。  

> **⚠️ AI 辅助开发** — 本项目使用 AI 辅助开发。代码审查、错误修复、性能优化和功能实现均通过 AI 辅助工作流生成并验证。所有测试均已通过验证。

## 快速开始

### 环境要求

- .NET 10 SDK（[下载](https://dotnet.microsoft.com/download/dotnet/10.0)）

### 安装

直接引用项目：

```xml
<ProjectReference Include="..\src\HuggingFace.Tokenizers\HuggingFace.Tokenizers.csproj" />
```

### 构建与运行

```bash
# 克隆仓库
git clone https://github.com/JottingCN/HuggingFace.Tokenizers.git
cd HuggingFace.Tokenizers

# 构建
dotnet build

# 运行测试（.NET 10 使用 MTP，需通过 dotnet run 运行）
dotnet run --project tests/HuggingFace.Tokenizers.Tests
```

### 最小示例

```csharp
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Models.BPE;
using HuggingFace.Tokenizers.Normalizers;
using HuggingFace.Tokenizers.PreTokenizers;
using HuggingFace.Tokenizers.Decoders;

// 构建分词器
var vocab = new Dictionary<string, uint>
{
    ["<unk>"] = 0, ["hello"] = 1, ["world"] = 2
};
var model = new BpeModel.BpeBuilder()
    .SetVocab(vocab)
    .SetMerges([])
    .SetUnkToken("<unk>")
    .Build();

var tokenizer = new TokenizerBuilder()
    .WithModel(model)
    .WithNormalizer(new NfcNormalizer())
    .WithPreTokenizer(new WhitespacePreTokenizer())
    .WithDecoder(new WordPieceDecoder())
    .Build();

// 编码
var encoding = tokenizer.Encode("Hello World!");
Console.WriteLine(string.Join(", ", encoding.GetTokens()));
// → ["hello", "world", "!"]

// 解码
var text = tokenizer.Decode(encoding.GetIds());
Console.WriteLine(text);
// → "hello world !"
```

### 从 HuggingFace Hub 加载预训练分词器

```csharp
using HuggingFace.Tokenizers.Serialization;

var tokenizer = await TokenizerLoader.FromPretrainedAsync("Qwen/Qwen2-7B-Instruct");
var encoding = tokenizer.Encode("Hello, how are you?");
Console.WriteLine(string.Join(", ", encoding.GetTokens()));
```

## 使用说明

### 核心接口

所有核心接口位于 `HuggingFace.Tokenizers.Abstractions`：

| 接口 | 职责 |
|------|------|
| `INormalizer` | 文本标准化（Unicode 规范化、小写、去音标等） |
| `IPreTokenizer` | 将文本拆分为预分词（空格、标点、字节级等） |
| `IModel` | 应用核心分词算法（BPE、WordPiece、Unigram、WordLevel） |
| `IPostProcessor` | 添加特殊 token 并格式化输出（[CLS]、[SEP]、模板等） |
| `IDecoder` | 将 token ID 解码回可读文本 |

### 模型（4 种）

| 模型 | 适用场景 | 说明 |
|------|----------|------|
| `BpeModel` | GPT-2、RoBERTa、LLaMA | 基于迭代合并的子词分词 |
| `WordPieceModel` | BERT、DistilBERT | 贪心最长匹配子词分词 |
| `WordLevelModel` | 简单词表 | 直接的 word-to-ID 映射 |
| `UnigramModel` | T5、ALBERT、XLNet | 概率子词分词（SentencePiece） |

### 标准化器（15 个）

`NfcNormalizer` · `NfdNormalizer` · `NfkcNormalizer` · `NfkdNormalizer` · `BertNormalizer` · `ByteLevelNormalizer` · `LowercaseNormalizer` · `StripNormalizer` · `StripAccentsNormalizer` · `PrependNormalizer` · `ReplaceNormalizer` · `SequenceNormalizer` · `PrecompiledNormalizer` · `NmtNormalizer` · `NormalizerWrapper`

### 预分词器（13 个）

`BertPreTokenizer` · `ByteLevelPreTokenizer` · `WhitespacePreTokenizer` · `WhitespaceSplitPreTokenizer` · `MetaspacePreTokenizer` · `DigitsPreTokenizer` · `PunctuationPreTokenizer` · `DelimiterSplitPreTokenizer` · `SequencePreTokenizer` · `SplitPreTokenizer` · `UnicodeScriptsPreTokenizer` · `FixedLengthPreTokenizer` · `PreTokenizerWrapper`

### 解码器（11 个）

`BpeDecoder` · `ByteLevelDecoder` · `ByteFallbackDecoder` · `CtcDecoder` · `FuseDecoder` · `MetaspaceDecoder` · `ReplaceDecoder` · `SequenceDecoder` · `StripDecoder` · `WordPieceDecoder` · `DecoderWrapper`

### 后处理器（6 个）

`BertProcessing` · `RobertaProcessing` · `TemplateProcessing` · `ByteLevelPostProcessor` · `SequenceProcessor` · `PostProcessorWrapper`

### 编码与解码

```csharp
// 单条编码（字节偏移，默认，与 Rust/Python/JS 一致）
var encoding = tokenizer.Encode("Hello World!");

// 字符级偏移（适用于 .NET 字符串操作）
var charEncoding = tokenizer.EncodeCharOffsets("Hello World!");

// 配对编码
var pairEncoding = tokenizer.EncodePair("Question?", "Answer.");

// 快速编码（仅返回 ID，跳过偏移追踪，更快）
uint[] ids = tokenizer.EncodeFast("Hello World!");

// 批量编码（自动并行）
var batch = tokenizer.EncodeBatch(["Hello", "World", "!"]);

// 批量快速编码
var fastBatch = tokenizer.EncodeBatchFast(["Hello", "World", "!"]);

// 解码
string text = tokenizer.Decode(encoding.GetIds());

// 批量解码
var texts = tokenizer.DecodeBatch([ids1, ids2, ids3]);
```

### 偏移映射

```csharp
var encoding = tokenizer.Encode("Hello World!");

// token 索引 → 字符偏移
(int Start, int End)? offsets = encoding.TokenToChars(0);

// 字符位置 → token 索引
int? tokenIdx = encoding.CharToToken(charPos: 5);

// token 索引 → word ID
uint? wordId = encoding.TokenToWord(0);

// word ID → token 范围
(int Start, int Count)? range = encoding.WordToTokens(0);

// word ID → 字符偏移
(int Start, int End)? wordChars = encoding.WordToChars(0);
```

### 零拷贝 Span API

```csharp
var encoding = tokenizer.Encode("Hello World!");

// 零拷贝只读视图（不分配数组）
ReadOnlySpan<uint> ids = encoding.IdsSpan;
ReadOnlySpan<string> tokens = encoding.TokensSpan;
ReadOnlySpan<(int Start, int End)> offsets = encoding.OffsetsSpan;
ReadOnlySpan<uint> typeIds = encoding.TypeIdsSpan;
ReadOnlySpan<uint> attention = encoding.AttentionMaskSpan;

// 修改偏移（自动 COW）
encoding.SetOffset(0, (1, 5));
```

### 填充与截断

```csharp
using HuggingFace.Tokenizers.Abstractions;

// 配置填充
tokenizer.Padding = new PaddingParams
{
    Strategy = PaddingStrategy.BatchLongest,  // 或 Fixed
    Direction = PaddingDirection.Right,
    PadToken = "[PAD]",
    PadTypeId = 0
};

// 配置截断
tokenizer.Truncation = new TruncationParams
{
    MaxLength = 512,
    Strategy = TruncationStrategy.LongestFirst,  // 或 OnlyFirst / OnlySecond
    Direction = TruncationDirection.Right
};
```

### 流式解码

`DecodeStream` 支持逐 token 增量解码，适用于 LLM 流式输出场景：

```csharp
var stream = tokenizer.CreateDecodeStream(skipSpecialTokens: true);

foreach (var tokenId in llmOutputTokenIds)
{
    var chunk = stream.Step(tokenId);
    if (chunk is not null)
        Console.Write(chunk);  // 当有效字符串块准备好时增量输出
}
```

### 序列化

兼容 HuggingFace `tokenizer.json` 格式：

```csharp
using HuggingFace.Tokenizers.Serialization;

// 从文件加载
var tokenizer = TokenizerLoader.FromFile("tokenizer.json");

// 从 JSON 字符串加载
var tokenizer2 = TokenizerLoader.FromJson(jsonString);

// 从字节数组加载
var tokenizer3 = TokenizerLoader.FromBytes(bytes);

// 从 HuggingFace Hub 下载并加载
var tokenizer4 = await TokenizerLoader.FromPretrainedAsync("bert-base-uncased");

// 序列化为 JSON
var json = tokenizer.ToJson(pretty: true);

// 保存到文件
tokenizer.Save("tokenizer.json");
```

### 并行度控制

```csharp
// 通过代码设置
Tokenizer.MaxDegreeOfParallelism = 4;

// 通过环境变量
// TOKENIZERS_PARALLELISM=1    → 顺序处理
// TOKENIZERS_PARALLELISM=4    → 4 线程
// 默认值：处理器核心数
```

### 特殊 Token 管理

```csharp
// 添加特殊 token
tokenizer.AddToken(new AddedToken("[CLS]", isSpecial: true));
tokenizer.AddToken(new AddedToken("[SEP]", isSpecial: true));

// 添加普通 token
tokenizer.AddToken(new AddedToken("custom_word"));

// 查询
uint? id = tokenizer.TokenToId("[CLS]");
string? token = tokenizer.IdToToken(0);

// 完整词表（模型 + 已添加）
var vocab = tokenizer.GetVocabWithAddedTokens();
int size = tokenizer.GetVocabSizeWithAddedTokens();
```

## 开发说明

### 项目结构

```
HuggingFace.Tokenizers/
├── src/
│   ├── HuggingFace.Tokenizers/                 # 完整实现（含 Abstractions 接口定义）
│   └── HuggingFace.Tokenizers.SourceGenerator/ # IIncrementalGenerator AOT 工厂
├── tests/
│   ├── HuggingFace.Tokenizers.Tests/           # MSTest v4 + MTP 测试（719 个）
│   └── HuggingFace.Tokenizers.Benchmarks/      # 性能基准测试

```

**`HuggingFace.Tokenizers`** 包含所有实现：Abstractions（核心接口 + 共享类型）、Models、Normalizers、PreTokenizers、Decoders、Processors、Internal（工具类）和 Serialization。核心接口和共享类型（`Tokenizer`、`Encoding`、`NormalizedString`、`PreTokenizedString`、`Token`、`AddedToken`、`DecodeStream`、`Pattern` 等）位于 `Abstractions/` 子目录。

**`HuggingFace.Tokenizers.SourceGenerator`** 提供 `IIncrementalGenerator`，为 AOT 场景生成编译期工厂（组件类型名解析、属性写入）。

### 技术栈

| 项目 | 选型 |
|------|------|
| 目标框架 | .NET 10 |
| 测试框架 | MSTest v4 + MTP（Microsoft Testing Platform） |
| 包管理 | NuGet |
| 外部依赖 | 零（仅 .NET 10 BCL） |
| AOT 兼容 | `IsAotCompatible=true`，无反射，SourceGenerator 编译期工厂 |

## 测试说明

```bash
# 运行所有测试（719 个）
dotnet run --project tests/HuggingFace.Tokenizers.Tests

# 运行基准测试
dotnet run --file tests/benchmarks/run.cs --rust-dir ../tokenizers
```

测试覆盖：

- 各组件单元测试（Normalizer、PreTokenizer、Model、Decoder、Processor）
- 端到端管道测试（Encode/Decode 全流程）
- 序列化往返测试（tokenizer.json 加载 → 保存 → 加载一致性）
- 偏移对齐测试（字节偏移 ↔ 字符偏移转换）
- 边界条件测试（空输入、特殊字符、补充平面 Unicode）
- 与 Rust 版本的输入输出对齐验证

### AI 辅助开发

本项目使用 AI 辅助开发。代码审查、测试补全、性能优化、AOT 兼容性改造等工作由 AI 工具协助完成，所有代码均经过人工审核与验证。

## 特殊说明

### 性能优化

### 与 Rust 原版的差异

- **正则表达式**：.NET `System.Text.RegularExpressions.Regex` 对 lookahead/lookbehind 支持有限，使用高级正则特性的分词器可能产生不同结果。


## 部署与配置

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `TOKENIZERS_PARALLELISM` | 批量编码最大并行度 | 处理器核心数 |

### AOT 部署

所有库均标记为 `IsAotCompatible=true`，可直接用于 Native AOT 部署：

- 无反射 — 不使用 `System.Text.Json` 反射序列化、`Activator.CreateInstance`、`Type.GetType`
- 源代码生成器 — `HuggingFace.Tokenizers.SourceGenerator` 通过 `IIncrementalGenerator` 提供编译期工厂
- 所有泛型类型在编译期解析

### HuggingFace Hub 缓存

从 Hub 下载的分词器缓存在 `~/.cache/huggingface/tokenizers/{modelId}/tokenizer.json`。

## 许可证

[Apache-2.0](LICENSE.txt) — 与 [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers) 相同。

---

