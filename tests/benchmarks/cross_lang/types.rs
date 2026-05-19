//! 数据结构 — 与 .NET 端 ConsistencyEntry / BenchResult / PerfResults 对齐

use serde::Serialize;

/// 编码+解码一致性条目
#[derive(Serialize)]
pub struct ConsistencyEntry {
    pub input: String,
    pub ids: Vec<u32>,
    pub tokens: Vec<String>,
    pub decoded: String,
}

/// 单项性能结果
#[derive(Serialize, Default)]
pub struct BenchResult {
    #[serde(rename = "TotalOps")]
    pub total_ops: u64,
    #[serde(rename = "ElapsedMs")]
    pub elapsed_ms: f64,
    #[serde(rename = "AvgMs")]
    pub avg_ms: f64,
    #[serde(rename = "OpsPerSec")]
    pub ops_per_sec: f64,
    #[serde(rename = "UsPerOp")]
    pub us_per_op: f64,
    #[serde(rename = "ThroughputMBps")]
    pub throughput_mbps: f64,
}

/// 完整性能结果集
#[derive(Serialize)]
pub struct PerfResults {
    #[serde(rename = "Mode")]
    pub mode: String,
    #[serde(rename = "Platform")]
    pub platform: String,
    #[serde(rename = "Timestamp")]
    pub timestamp: String,
    #[serde(rename = "RustcVersion")]
    pub rustc_version: String,
    #[serde(rename = "CargoVersion")]
    pub cargo_version: String,
    #[serde(rename = "CompileSettings")]
    pub compile_settings: String,
    #[serde(rename = "BpeEncodeSingle")]
    pub bpe_encode_single: Option<BenchResult>,
    #[serde(rename = "BpeEncodeBatch")]
    pub bpe_encode_batch: Option<BenchResult>,
    #[serde(rename = "BpeEncodeFast")]
    pub bpe_encode_fast: Option<BenchResult>,
    #[serde(rename = "UnigramEncodeSingle")]
    pub unigram_encode_single: Option<BenchResult>,
    #[serde(rename = "UnigramEncodeBatch")]
    pub unigram_encode_batch: Option<BenchResult>,
    #[serde(rename = "WordPieceEncodeSingle")]
    pub wordpiece_encode_single: Option<BenchResult>,
    #[serde(rename = "WordPieceEncodeBatch")]
    pub wordpiece_encode_batch: Option<BenchResult>,
    #[serde(rename = "Qwen25EncodeSingle")]
    pub qwen25_encode_single: Option<BenchResult>,
    #[serde(rename = "Qwen25EncodeBatch")]
    pub qwen25_encode_batch: Option<BenchResult>,
    #[serde(rename = "WordPieceEncodeFast")]
    pub wordpiece_encode_fast: Option<BenchResult>,
    #[serde(rename = "WordLevelEncodeSingle")]
    pub wordlevel_encode_single: Option<BenchResult>,
    #[serde(rename = "WordLevelEncodeBatch")]
    pub wordlevel_encode_batch: Option<BenchResult>,
    #[serde(rename = "BpeEncodeCharOffsets")]
    pub bpe_encode_char_offsets: Option<BenchResult>,
    #[serde(rename = "UnigramEncodeCharOffsets")]
    pub unigram_encode_char_offsets: Option<BenchResult>,
    #[serde(rename = "WordPieceEncodeCharOffsets")]
    pub wordpiece_encode_char_offsets: Option<BenchResult>,
    #[serde(rename = "BpeConcurrent1T")]
    pub bpe_concurrent_1t: Option<BenchResult>,
    #[serde(rename = "BpeConcurrent2T")]
    pub bpe_concurrent_2t: Option<BenchResult>,
    #[serde(rename = "BpeConcurrent4T")]
    pub bpe_concurrent_4t: Option<BenchResult>,
    #[serde(rename = "BpeConcurrent8T")]
    pub bpe_concurrent_8t: Option<BenchResult>,
    #[serde(rename = "BpeTruncation512")]
    pub bpe_truncation_512: Option<BenchResult>,
    #[serde(rename = "BpeTruncation128")]
    pub bpe_truncation_128: Option<BenchResult>,
    #[serde(rename = "SerializationLoad")]
    pub serialization_load: Option<BenchResult>,
    #[serde(rename = "SerializationSave")]
    pub serialization_save: Option<BenchResult>,
    #[serde(rename = "SerializationDeserialize")]
    pub serialization_deserialize: Option<BenchResult>,
}
