//! 跨语言训练 — 仅训练 + 保存 tokenizer 模型 JSON，不生成一致性/性能数据。
//!
//! 运行：
//!   cargo bench --bench cross_lang_consistency
//!
//! 输出：
//!   benches/data/tokenizer-rust-bpe.json
//!   benches/data/tokenizer-rust-unigram.json
//!   benches/data/tokenizer-rust-wordpiece.json

mod cross_lang;

use criterion::Criterion;
use cross_lang::train::{train_bpe, train_unigram, train_wordpiece, train_wordlevel};
use std::fs;
use tokenizers::models::ModelWrapper;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

// ── Criterion 入口（仅训练，不执行推理测试） ──

fn bench_train_only(c: &mut Criterion) {
    let data_dir = "benches/data";
    let small_path = format!("{}/small.txt", data_dir);

    assert!(
        fs::metadata(&small_path).is_ok(),
        "small.txt not found in {}",
        data_dir
    );

    // ── 训练 ──
    println!("▶ Rust: 训练 tokenizer...");
    let bpe_tokenizer = train_bpe(&small_path, 30_000, 2);
    let unigram_tokenizer = train_unigram(&small_path, 8_000);
    let wp_tokenizer = train_wordpiece(&small_path, 30_000, 2);
    let wordlevel_tokenizer = train_wordlevel(&small_path, 30_000);

    // ── 保存训练好的 tokenizer 供三端加载 ──
    println!("▶ Rust: 保存训练好的 tokenizer...");
    bpe_tokenizer
        .save(&format!("{}/tokenizer-rust-bpe.json", data_dir), false)
        .expect("save bpe tokenizer");
    unigram_tokenizer
        .save(
            &format!("{}/tokenizer-rust-unigram.json", data_dir),
            false,
        )
        .expect("save unigram tokenizer");
    wp_tokenizer
        .save(
            &format!("{}/tokenizer-rust-wordpiece.json", data_dir),
            false,
        )
        .expect("save wordpiece tokenizer");
    wordlevel_tokenizer
        .save(
            &format!("{}/tokenizer-rust-wordlevel.json", data_dir),
            false,
        )
        .expect("save wordlevel tokenizer");
    println!("  ✅ tokenizer-rust-bpe.json / unigram / wordpiece / wordlevel 已保存");

    // ── Criterion 占位 benchmark（训练已完成，此 benchmark 仅用于触发 Criterion 框架） ──
    let small_data = fs::read_to_string(&small_path).expect("read small.txt");
    let lines_100: Vec<&str> = small_data.lines().take(100).collect();

    let mut group = c.benchmark_group("cross-lang-train");
    group.bench_function("bpe_encode_100_lines_warmup", |b| {
        b.iter(|| {
            for line in &lines_100 {
                let _ = bpe_tokenizer.encode(
                    tokenizers::EncodeInput::from(*line),
                    false,
                );
            }
        })
    });
    group.finish();
}

criterion::criterion_group!(benches, bench_train_only);
criterion::criterion_main!(benches);
