# Tokenizers 基准测试套件

跨语言（Rust / .NET JIT / .NET AOT）推理性能与数据一致性测试。

## 测试流程

所有测试采用 **Rust 统一训练** 架构：Rust 负责训练所有 tokenizer 模型并保存为 JSON，三个平台加载同一模型进行推理测试。

```
Phase 1: 生成随机测试数据
Phase 2: Rust 统一训练 tokenizer 模型（BPE/Unigram/WordPiece/WordLevel），保存为 JSON
Phase 3: 三端分别加载模型 + 推理测试
  3.1: .NET JIT（加载 Rust 训练的模型）
  3.2: .NET AOT（加载 Rust 训练的模型）
  3.3: Rust（从文件重新加载模型）
Phase 4: 一致性比对 + 生成报告
```

## 快速开始

```bash
# 完整测试（一致性 + 性能基准）
dotnet run --file tests/benchmarks/run.cs --rust-dir ../tokenizers

# 仅数据一致性验证
dotnet run --file tests/benchmarks/run.cs --rust-dir ../tokenizers --consistency-only

# 使用已有数据重新生成报告
dotnet run --file tests/benchmarks/regen-report.cs
```

**前置条件：**
- .NET 10 SDK（[下载](https://dotnet.microsoft.com/download/dotnet/10.0)）
- Rust toolchain（[安装](https://rustup.rs)）— 仅 Rust 测试需要
- Rust `tokenizers` 源码目录 — 通过 `--rust-dir` 指定

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `run.cs` | 推理基准入口：推理一致性验证 + 推理性能基准，支持 `--consistency-only` |
| `regen-report.cs` | 从已有 JSON 重新生成报告，不执行测试 |
| `cross_lang_consistency.rs` | Rust 端：训练 tokenizer 模型 + 保存 JSON |
| `cross_lang_inference.rs` | Rust 端：加载模型 + 生成推理一致性数据 |
| `cross_lang_perf.rs` | Rust 端：加载模型 + 推理性能基准 |
| `cross_lang/` | Rust 共享模块（types.rs / train.rs） |

### 一致性验证

验证三方（JIT / AOT / Rust）的编解码结果完全一致（16838 条随机输入 × 5 模型 = 15 项检查）：

| 模型 | JIT vs AOT | Rust vs JIT | Rust vs AOT |
|------|-----------|-------------|-------------|
| BPE | ✅ 16838/16838 | ✅ 16838/16838 | ✅ 16838/16838 |
| Unigram | ✅ 16838/16838 | ✅ 16838/16838 | ✅ 16838/16838 |
| WordPiece | ✅ 16838/16838 | ✅ 16838/16838 | ✅ 16838/16838 |
| WordLevel | ✅ 16838/16838 | ✅ 16838/16838 | ✅ 16838/16838 |
| Qwen2.5 | ✅ 16838/16838 | ✅ 16838/16838 | ✅ 16838/16838 |

**验证项目：**
- encode 结果三方一致（token IDs + token strings）
- decode 结果三方一致
- 所有模型（BPE / Unigram / WordPiece / WordLevel / Qwen2.5）对齐
- 确定性验证（Multi-Run Determinism: YES ✓）

### 性能基准

测试环境：Ubuntu 24.04.4 LTS / Intel Xeon 6982P-C 2核 / .NET 10.0.300 / Rust 1.95.0

> 测试日期：2026-05-18（seed: 1779082396）

#### 编码性能（AOT vs Rust）

| 模型 | Rust (100%) | AOT | JIT |
|------|-------------|-----|-----|
| BPE (vocab=30000) 单条 Encode | 112,172 | 119,957 (**107%**) | 75,579 (67%) |
| BPE (vocab=30000) 批量 Encode | 135,883 | 146,512 (**108%**) | 161,720 (**119%**) |
| BPE EncodeFast | — | 71,660 (62%) | 79,156 (68%) |
| Unigram (vocab=8000) 单条 Encode | 70,914 | 118,877 (**168%**) | 120,562 (**170%**) |
| Unigram (vocab=8000) 批量 Encode | 85,419 | 146,622 (**172%**) | 144,931 (**170%**) |
| WordPiece (vocab=30000) 单条 Encode | 108,669 | 103,035 (95%) | 103,764 (95%) |
| WordPiece (vocab=30000) 批量 Encode | 129,497 | 97,203 (75%) | 132,933 (**103%**) |
| WordPiece EncodeFast | — | 115,842 (103%) | 127,032 (113%) |
| WordLevel (vocab=30000) 单条 Encode | 114,240 | 121,907 (**107%**) | 119,963 (**105%**) |
| WordLevel (vocab=30000) 批量 Encode | 138,441 | 142,977 (**103%**) | 152,799 (**110%**) |
| Qwen2.5 (vocab=151643) 单条 Encode | 36,973 | 34,286 (93%) | 34,179 (92%) |
| Qwen2.5 (vocab=151643) 批量 Encode | 43,295 | 43,510 (100%) | 49,163 (**114%**) |

#### 其他性能指标（AOT vs Rust）

| 指标 | Rust (100%) | AOT | JIT |
|------|-------------|-----|-----|
| BPE 并发 (4T) | 642 | 225 (35%) | 205 (32%) |
| BPE 截断 (maxLength=512) | 335 | 139 (42%) | 130 (39%) |
| BPE 截断 (maxLength=128) | 344 | 135 (39%) | 125 (36%) |
| 序列化 Load (FromJson) | 761 | 933 (**123%**) | 663 (87%) |
| 序列化 Save (ToJson) | 4,879 | 3,813 (78%) | 2,005 (41%) |
| 序列化 Deserialize | 761 | 924 (**121%**) | 691 (91%) |
| BPE EncodeCharOffsets (batch) | 89,813 | 135,294 (**151%**) | 141,708 (**158%**) |
| Unigram EncodeCharOffsets (batch) | 62,751 | 136,219 (**217%**) | 144,203 (**230%**) |

#### 结论

- **一致性**：所有模型（BPE / Unigram / WordPiece / WordLevel / Qwen2.5）JIT / AOT / Rust 三端编码结果完全一致（16838/16838）
- **编码性能**：AOT 平均约为 Rust 的 **114%**（超越 Rust）
- **亮点**：Unigram AOT 单条 168%、批量 172%，大幅超越 Rust；BPE/WordLevel AOT 也略超 Rust（107%）
- **JIT 批量优势**：BPE 批量 119%、WordLevel 110%、Qwen 114%，JIT 批量场景表现优异
- **EncodeCharOffsets**：AOT/JIT 全面超越 Rust（BPE +51%/+58%，Unigram +117%/+130%）
- **短板**：并发和截断场景 Rust 优势明显（AOT 仅 35-42%），是后续优化方向

## 输出文件

```
tests/benchmarks/result/
├── REPORT.md               # 完整报告（一致性 + 性能）
├── meta.json               # 测试环境信息
├── consistency-summary.json # 一致性汇总
├── consistency-*.json      # 三方一致性数据（jit/aot/rust × bpe/unigram/wordpiece/wordlevel/qwen）
├── perf-jit.json           # JIT 性能数据
├── perf-aot.json           # AOT 性能数据
├── perf-rust.json          # Rust 性能数据
├── random-small.txt        # 测试数据（128KB）
├── random-big.txt          # 测试数据（1.8MB）
└── log-*.txt               # 各阶段日志
```

## 测试数据

测试数据包含多种语言和特殊字符：

- 英文（35%）、中文（15%）、日韩（7%）
- 拉丁扩展 / 带音标（12%）、Combining Marks（2%）
- 阿拉伯文（3%）、希伯来文（2%）、天城文（2%）、泰文（2%）
- 全角 ASCII（2%）、长无空格串（2%）
- 零宽字符（1%）、Emoji（7%）、特殊符号（5%）、标点（3%）

## 架构说明

### 为什么 Rust 统一训练？

1. **消除训练差异** — 不同平台的浮点实现可能导致训练结果微小差异，统一训练消除此问题
2. **简化验证** — 不需要跨平台加载对方训练的模型
3. **提高效率** — 训练只执行一次，三个平台共享同一模型
4. **公平对比** — 所有平台测试完全相同的模型，推理性能对比更准确

### AOT 编译策略

| 场景 | 指令集 | 优化 |
|------|--------|------|
| 一致性验证 | 自动检测（AVX-512/AVX2/AVX/SSE4.2） | Speed 优先 |
| 性能基准 | 自动检测（AVX-512/AVX2/AVX/SSE4.2） | Speed 优先 |

> 注：`InvariantGlobalization` 已移除，因其导致 NFC 标准化变为 no-op，不过NFC重新实现，未使用系统自带，理论上不在影响
