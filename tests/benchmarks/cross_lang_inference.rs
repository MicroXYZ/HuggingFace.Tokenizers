//! 跨语言推理一致性测试 — 加载已训练的 tokenizer 模型，生成一致性数据供三端比对。
//!
//! 运行：
//!   cargo bench --bench cross_lang_inference
//!
//! 输入（由 cross_lang_consistency 训练生成）：
//!   benches/data/tokenizer-rust-bpe.json
//!   benches/data/tokenizer-rust-unigram.json
//!   benches/data/tokenizer-rust-wordpiece.json
//!   benches/data/qwen2.5-tokenizer.json（Phase 3 下载）
//!
//! 输出：
//!   benches/data/consistency-rust-bpe.json
//!   benches/data/consistency-rust-unigram.json
//!   benches/data/consistency-rust-wordpiece.json
//!   benches/data/consistency-rust-qwen2.5.json

mod cross_lang;

use criterion::{Criterion, Throughput};
use cross_lang::types::ConsistencyEntry;
use std::fs;
use std::hint::black_box;
use tokenizers::{EncodeInput, Tokenizer};

// ── 一致性工具函数 ──

fn encode_decode_consistency(tokenizer: &Tokenizer, lines: &[&str]) -> Vec<ConsistencyEntry> {
    lines
        .iter()
        .map(|line| {
            let encoding = tokenizer
                .encode(EncodeInput::from(*line), false)
                .expect("encode");
            let ids = encoding.get_ids().to_vec();
            let tokens = encoding.get_tokens().to_vec();
            let decoded = tokenizer.decode(&ids, false).expect("decode");
            ConsistencyEntry {
                input: line.to_string(),
                ids,
                tokens,
                decoded,
            }
        })
        .collect()
}

fn save_consistency(tag: &str, data_dir: &str, entries: &[ConsistencyEntry]) {
    let json = serde_json::to_string_pretty(entries).expect("serialize");
    let path = format!("{}/consistency-rust-{}.json", data_dir, tag);
    fs::write(&path, &json).expect("write consistency");
    println!("  ✅ {}: {} entries → {}", tag, entries.len(), path);
}

/// 从文件加载已训练的 tokenizer
fn load_tokenizer(path: &str) -> Option<Tokenizer> {
    if fs::metadata(path).is_ok() {
        Some(Tokenizer::from_file(path).expect(&format!("load tokenizer: {}", path)))
    } else {
        println!("  ⚠ 未找到: {}", path);
        None
    }
}

// ── Criterion 入口 ──

fn bench_inference(c: &mut Criterion) {
    let data_dir = "benches/data";
    let small_path = format!("{}/small.txt", data_dir);

    assert!(
        fs::metadata(&small_path).is_ok(),
        "small.txt not found in {}",
        data_dir
    );

    let small_data = fs::read_to_string(&small_path).expect("read small.txt");
    let big_path = format!("{}/big.txt", data_dir);
    let big_data = fs::read_to_string(&big_path).expect("read big.txt");
    let all_lines: Vec<&str> = small_data.lines().chain(big_data.lines()).collect();
    println!("▶ Rust: 一致性数据: small={} 行 + big={} 行 = {} 行",
        small_data.lines().count(), big_data.lines().count(), all_lines.len());

    // ── 加载已训练的模型（由 cross_lang_consistency 训练生成） ──
    println!("▶ Rust: 加载已训练的 tokenizer 模型...");

    let bpe_tokenizer = load_tokenizer(&format!("{}/tokenizer-rust-bpe.json", data_dir));
    let unigram_tokenizer = load_tokenizer(&format!("{}/tokenizer-rust-unigram.json", data_dir));
    let wp_tokenizer = load_tokenizer(&format!("{}/tokenizer-rust-wordpiece.json", data_dir));
    let wordlevel_tokenizer = load_tokenizer(&format!("{}/tokenizer-rust-wordlevel.json", data_dir));

    // ── 生成一致性数据 ──
    println!("▶ Rust: 生成一致性数据 ({} 行)...", all_lines.len());

    if let Some(ref tok) = bpe_tokenizer {
        save_consistency("bpe", data_dir, &encode_decode_consistency(tok, &all_lines));
    }
    if let Some(ref tok) = unigram_tokenizer {
        save_consistency(
            "unigram",
            data_dir,
            &encode_decode_consistency(tok, &all_lines),
        );
    }
    if let Some(ref tok) = wp_tokenizer {
        save_consistency(
            "wordpiece",
            data_dir,
            &encode_decode_consistency(tok, &all_lines),
        );
    }
    if let Some(ref tok) = wordlevel_tokenizer {
        save_consistency(
            "wordlevel",
            data_dir,
            &encode_decode_consistency(tok, &all_lines),
        );
    }

    // ── Qwen2.5（from_pretrained 自动下载） ──
    println!("  ⏳ 加载 Qwen/Qwen2.5-7B-Instruct...");
    match Tokenizer::from_pretrained("Qwen/Qwen2.5-7B-Instruct", None) {
        Ok(qwen_tok) => {
            save_consistency("qwen2.5", data_dir, &encode_decode_consistency(&qwen_tok, &all_lines));
        }
        Err(e) => println!("  ⏭ Qwen2.5 跳过: {}", e),
    }

    // ── Criterion benchmark（推理一致性验证：编码 1000 行） ──
    if let Some(ref tok) = bpe_tokenizer {
        let mut group = c.benchmark_group("cross-lang-inference");
        group.throughput(Throughput::Elements(all_lines.len() as u64));
        group.bench_function("bpe_encode_1000_lines", |b| {
            b.iter(|| {
                for line in black_box(&all_lines) {
                    let _ = tok.encode(EncodeInput::from(*line), false);
                }
            })
        });
        group.finish();
    }
}

criterion::criterion_group!(benches, bench_inference);
criterion::criterion_main!(benches);
