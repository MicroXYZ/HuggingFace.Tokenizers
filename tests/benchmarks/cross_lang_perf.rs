//! 跨语言性能基准 — 加载已训练的 tokenizer 模型，执行性能基准测试。
//!
//! 运行：
//!   cargo bench --bench cross_lang_perf
//!
//! 输入（由 cross_lang_consistency 训练生成）：
//!   benches/data/tokenizer-rust-bpe.json
//!   benches/data/tokenizer-rust-unigram.json
//!   benches/data/tokenizer-rust-wordpiece.json
//!   benches/data/qwen2.5-tokenizer.json（Phase 3 下载）
//!
//! 输出：
//!   benches/data/perf-rust.json

mod cross_lang;

use criterion::Criterion;
use cross_lang::types::{BenchResult, PerfResults};
use std::fs;
use std::hint::black_box;
use std::str::FromStr;
use std::time::Instant;
use tokenizers::{EncodeInput, Tokenizer};

// ── 性能基准函数 ──

fn bench_encode(
    label: &str,
    tokenizer: &Tokenizer,
    lines: &[&str],
    iterations: u32,
) -> BenchResult {
    for line in lines {
        let _ = tokenizer.encode(EncodeInput::from(*line), false);
    }
    let total_ops = iterations as u64 * lines.len() as u64;
    let start = Instant::now();
    for _ in 0..iterations {
        for line in lines {
            let _ = tokenizer.encode(EncodeInput::from(*line), false);
        }
    }
    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();
    let us_per_op = elapsed.as_micros() as f64 / total_ops as f64;
    println!("  {}: {:.0} ops/sec, {:.2} µs/op", label, ops_per_sec, us_per_op);
    BenchResult {
        total_ops,
        elapsed_ms,
        avg_ms: elapsed_ms / total_ops as f64,
        ops_per_sec,
        us_per_op,
        throughput_mbps: 0.0,
    }
}

fn bench_encode_batch(
    label: &str,
    tokenizer: &Tokenizer,
    lines: &[&str],
    iterations: u32,
) -> BenchResult {
    let input: Vec<EncodeInput> = lines.iter().map(|l| EncodeInput::from(*l)).collect();
    let _ = tokenizer.encode_batch(input.clone(), false);
    let total_lines = iterations as u64 * lines.len() as u64;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = tokenizer.encode_batch(input.clone(), false);
    }
    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let ops_per_sec = total_lines as f64 / elapsed.as_secs_f64();
    let us_per_op = elapsed.as_micros() as f64 / total_lines as f64;
    println!(
        "  {}: {:.0} lines/sec, {:.2} µs/line ({} lines/batch)",
        label,
        ops_per_sec,
        us_per_op,
        lines.len()
    );
    BenchResult {
        total_ops: total_lines,
        elapsed_ms,
        avg_ms: elapsed_ms / total_lines as f64,
        ops_per_sec,
        us_per_op,
        throughput_mbps: 0.0,
    }
}

fn bench_encode_fast(
    label: &str,
    tokenizer: &Tokenizer,
    lines: &[&str],
    iterations: u32,
) -> BenchResult {
    for line in lines {
        let _ = tokenizer.encode_fast(*line, false);
    }
    let total_ops = iterations as u64 * lines.len() as u64;
    let start = Instant::now();
    for _ in 0..iterations {
        for line in lines {
            let _ = tokenizer.encode_fast(*line, false);
        }
    }
    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();
    let us_per_op = elapsed.as_micros() as f64 / total_ops as f64;
    println!("  {}: {:.0} ops/sec, {:.2} µs/op", label, ops_per_sec, us_per_op);
    BenchResult {
        total_ops,
        elapsed_ms,
        avg_ms: elapsed_ms / total_ops as f64,
        ops_per_sec,
        us_per_op,
        throughput_mbps: 0.0,
    }
}

fn bench_encode_char_offsets(
    label: &str,
    tokenizer: &Tokenizer,
    lines: &[&str],
    iterations: u32,
) -> BenchResult {
    let _ = tokenizer.encode_batch_char_offsets(lines.to_vec(), false);
    let total_ops = iterations as u64 * lines.len() as u64;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = tokenizer.encode_batch_char_offsets(lines.to_vec(), false);
    }
    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();
    let us_per_op = elapsed.as_micros() as f64 / total_ops as f64;
    println!(
        "  {}: {:.0} lines/sec, {:.2} µs/line",
        label, ops_per_sec, us_per_op
    );
    BenchResult {
        total_ops,
        elapsed_ms,
        avg_ms: elapsed_ms / total_ops as f64,
        ops_per_sec,
        us_per_op,
        throughput_mbps: 0.0,
    }
}

fn bench_concurrent(
    label: &str,
    tokenizer: &Tokenizer,
    lines: &[&str],
    num_threads: usize,
    iterations: u32,
) -> BenchResult {
    use std::sync::Arc;
    let tok = Arc::new(tokenizer.clone());
    let lines_per_thread = lines.len() / num_threads;
    let inputs: Vec<String> = (0..num_threads)
        .map(|i| lines[i * lines_per_thread..(i + 1) * lines_per_thread].join("\n"))
        .collect();
    let total_bytes: usize = inputs.iter().map(|s| s.len()).sum();

    // warmup
    std::thread::scope(|s| {
        let handles: Vec<_> = inputs
            .iter()
            .map(|input| {
                let t = &tok;
                s.spawn(move || {
                    let _ = t.encode(black_box(input.as_str()), false);
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    });

    let total_ops = iterations as u64 * num_threads as u64;
    let start = Instant::now();
    for _ in 0..iterations {
        std::thread::scope(|s| {
            let handles: Vec<_> = inputs
                .iter()
                .map(|input| {
                    let t = &tok;
                    s.spawn(move || {
                        black_box(t.encode(black_box(input.as_str()), false).unwrap());
                    })
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
        });
    }
    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();
    let throughput_mbps =
        (total_bytes as f64 / (1024.0 * 1024.0)) / (elapsed.as_secs_f64() / iterations as f64);
    println!(
        "  {}: {:.0} ops/sec, {:.1} MiB/s ({} threads)",
        label, ops_per_sec, throughput_mbps, num_threads
    );
    BenchResult {
        total_ops,
        elapsed_ms,
        avg_ms: elapsed_ms / total_ops as f64,
        ops_per_sec,
        us_per_op: elapsed.as_micros() as f64 / total_ops as f64,
        throughput_mbps,
    }
}

fn bench_truncation(
    label: &str,
    tokenizer: &Tokenizer,
    input: &str,
    max_length: usize,
    direction: &str,
    iterations: u32,
) -> BenchResult {
    use tokenizers::tokenizer::{TruncationDirection, TruncationParams, TruncationStrategy};
    let mut tok = tokenizer.clone();
    let dir = match direction {
        "left" => TruncationDirection::Left,
        _ => TruncationDirection::Right,
    };
    let _ = tok.with_truncation(Some(TruncationParams {
        max_length,
        strategy: TruncationStrategy::LongestFirst,
        stride: 0,
        direction: dir,
    }));

    let _ = tok.encode(EncodeInput::from(input.to_string()), false);
    let total_bytes = input.len();
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = tok.encode(EncodeInput::from(input.to_string()), false);
    }
    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();
    let throughput_mbps =
        (total_bytes as f64 / (1024.0 * 1024.0)) / (elapsed.as_secs_f64() / iterations as f64);
    println!(
        "  {}: {:.0} ops/sec, {:.1} MiB/s (max_len={})",
        label, ops_per_sec, throughput_mbps, max_length
    );
    BenchResult {
        total_ops: iterations as u64,
        elapsed_ms,
        avg_ms: elapsed_ms / iterations as f64,
        ops_per_sec,
        us_per_op: elapsed.as_micros() as f64 / iterations as f64,
        throughput_mbps,
    }
}

fn bench_serialization(
    label: &str,
    tokenizer: &Tokenizer,
    iterations: u32,
) -> (BenchResult, BenchResult, BenchResult) {
    let json_str = tokenizer.to_string(false).expect("serialize tokenizer");

    // Save
    let _ = tokenizer.to_string(false);
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = tokenizer.to_string(false);
    }
    let save_elapsed = start.elapsed();
    let save_ms = save_elapsed.as_secs_f64() * 1000.0;
    let save_avg = save_ms / iterations as f64;
    let save_ops = iterations as f64 / save_elapsed.as_secs_f64();
    println!(
        "  {} Save: {:.0} ops/sec, {:.2} ms/op",
        label, save_ops, save_avg
    );
    let save_result = BenchResult {
        total_ops: iterations as u64,
        elapsed_ms: save_ms,
        avg_ms: save_avg,
        ops_per_sec: save_ops,
        us_per_op: save_elapsed.as_micros() as f64 / iterations as f64,
        throughput_mbps: 0.0,
    };

    // Load / Deserialize
    let _ = Tokenizer::from_str(&json_str).expect("deserialize");
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = Tokenizer::from_str(&json_str).expect("deserialize");
    }
    let load_elapsed = start.elapsed();
    let load_ms = load_elapsed.as_secs_f64() * 1000.0;
    let load_avg = load_ms / iterations as f64;
    let load_ops = iterations as f64 / load_elapsed.as_secs_f64();
    println!(
        "  {} Load: {:.0} ops/sec, {:.2} ms/op",
        label, load_ops, load_avg
    );
    let load_result = BenchResult {
        total_ops: iterations as u64,
        elapsed_ms: load_ms,
        avg_ms: load_avg,
        ops_per_sec: load_ops,
        us_per_op: load_elapsed.as_micros() as f64 / iterations as f64,
        throughput_mbps: 0.0,
    };

    let deser_result = BenchResult {
        total_ops: iterations as u64,
        elapsed_ms: load_ms,
        avg_ms: load_avg,
        ops_per_sec: load_ops,
        us_per_op: load_elapsed.as_micros() as f64 / iterations as f64,
        throughput_mbps: 0.0,
    };

    (load_result, save_result, deser_result)
}

// ── 从文件加载已训练的 tokenizer ──

fn load_tokenizer(path: &str) -> Option<Tokenizer> {
    if fs::metadata(path).is_ok() {
        Some(Tokenizer::from_file(path).expect(&format!("load tokenizer: {}", path)))
    } else {
        println!("  ⚠ 未找到: {}", path);
        None
    }
}

// ── Criterion 入口 ──

fn bench_perf(c: &mut Criterion) {
    let data_dir = "benches/data";
    let small_path = format!("{}/small.txt", data_dir);
    let big_path = format!("{}/big.txt", data_dir);

    assert!(fs::metadata(&small_path).is_ok(), "small.txt not found in {}", data_dir);
    assert!(fs::metadata(&big_path).is_ok(), "big.txt not found in {}", data_dir);

    let small_data = fs::read_to_string(&small_path).expect("read small.txt");

    // ── 加载已训练的模型（由 cross_lang_consistency 训练生成） ──
    println!("▶ Rust: 加载已训练的 tokenizer 模型...");
    let bpe_tokenizer = load_tokenizer(&format!("{}/tokenizer-rust-bpe.json", data_dir));
    let unigram_tokenizer = load_tokenizer(&format!("{}/tokenizer-rust-unigram.json", data_dir));
    let wp_tokenizer = load_tokenizer(&format!("{}/tokenizer-rust-wordpiece.json", data_dir));
    let wordlevel_tokenizer = load_tokenizer(&format!("{}/tokenizer-rust-wordlevel.json", data_dir));

    // ── Criterion benchmark ──
    if let Some(ref tok) = bpe_tokenizer {
        let mut group = c.benchmark_group("cross-lang-perf");
        let lines: Vec<&str> = small_data.lines().take(100).collect();
        group.bench_function("bpe_encode_single", |b| {
            b.iter(|| {
                for line in black_box(&lines) {
                    let _ = tok.encode(EncodeInput::from(*line), false);
                }
            })
        });
        group.finish();
    }

    // ═══════════════════════════════════════════════════════════════
    //  性能基准测试（非 criterion，手动计时）
    // ═══════════════════════════════════════════════════════════════

    println!("\n▶ Rust: 性能基准测试...");

    let encode_lines: Vec<&str> = small_data.lines().take(1000).collect();

    // ── BPE 编码性能（从文件加载） ──
    let (bpe_encode_single, bpe_encode_batch, bpe_encode_fast) = if let Some(ref tok) = bpe_tokenizer {
        let single = bench_encode("BPE Encode", tok, &encode_lines, 100);
        let batch = bench_encode_batch("BPE Batch", tok, &encode_lines, 100);
        let fast = bench_encode_fast("BPE EncodeFast", tok, &encode_lines, 100);
        (Some(single), Some(batch), Some(fast))
    } else {
        (None, None, None)
    };

    // ── Unigram 编码性能（从文件加载） ──
    let (unigram_encode_single, unigram_encode_batch) = if let Some(ref tok) = unigram_tokenizer {
        let single = bench_encode("Unigram Encode", tok, &encode_lines, 100);
        let batch = bench_encode_batch("Unigram Batch", tok, &encode_lines, 100);
        (Some(single), Some(batch))
    } else {
        (None, None)
    };

    // ── WordPiece 编码性能（从文件加载） ──
    let (wp_encode, wp_encode_batch, wp_encode_fast) = if let Some(ref tok) = wp_tokenizer {
        let encode = bench_encode("WordPiece Encode", tok, &encode_lines, 100);
        let batch = bench_encode_batch("WordPiece Batch", tok, &encode_lines, 100);
        let fast = bench_encode_fast("WordPiece EncodeFast", tok, &encode_lines, 100);
        (Some(encode), Some(batch), Some(fast))
    } else {
        (None, None, None)
    };

    // ── WordLevel 编码性能（从文件加载） ──
    let (wordlevel_encode, wordlevel_encode_batch) = if let Some(ref tok) = wordlevel_tokenizer {
        let single = bench_encode("WordLevel Encode", tok, &encode_lines, 100);
        let batch = bench_encode_batch("WordLevel Batch", tok, &encode_lines, 100);
        (Some(single), Some(batch))
    } else {
        (None, None)
    };

    // ── Qwen2.5（from_pretrained 自动下载） ──
    println!("  ⏳ 加载 Qwen/Qwen2.5-7B-Instruct...");
    let (qwen_encode, qwen_batch) = match Tokenizer::from_pretrained("Qwen/Qwen2.5-7B-Instruct", None) {
        Ok(tok) => {
            println!("  ✅ Qwen2.5 loaded");
            let single = bench_encode("Qwen2.5 Encode", &tok, &encode_lines, 100);
            let batch = bench_encode_batch("Qwen2.5 Batch", &tok, &encode_lines, 100);
            (Some(single), Some(batch))
        }
        Err(e) => {
            println!("  ⏭ Qwen2.5 跳过: {}", e);
            (None, None)
        }
    };

    // ── EncodeCharOffsets（从文件加载） ──
    let bpe_encode_char_offsets = bpe_tokenizer.as_ref().map(|tok| bench_encode_char_offsets("BPE CharOffsets", tok, &encode_lines, 100));
    let unigram_encode_char_offsets = unigram_tokenizer.as_ref().map(|tok| bench_encode_char_offsets("Unigram CharOffsets", tok, &encode_lines, 100));
    let wordpiece_encode_char_offsets = wp_tokenizer.as_ref().map(|tok| bench_encode_char_offsets("WordPiece CharOffsets", tok, &encode_lines, 100));

    // ── Concurrent（从文件加载） ──
    let (bpe_concurrent_1t, bpe_concurrent_2t, bpe_concurrent_4t, bpe_concurrent_8t) =
        if let Some(ref tok) = bpe_tokenizer {
            let t1 = bench_concurrent("BPE Concurrent 1T", tok, &encode_lines, 1, 20);
            let t2 = bench_concurrent("BPE Concurrent 2T", tok, &encode_lines, 2, 20);
            let t4 = bench_concurrent("BPE Concurrent 4T", tok, &encode_lines, 4, 20);
            let t8 = bench_concurrent("BPE Concurrent 8T", tok, &encode_lines, 8, 20);
            (Some(t1), Some(t2), Some(t4), Some(t8))
        } else {
            (None, None, None, None)
        };

    // ── Truncation（从文件加载） ──
    let big_data = fs::read_to_string(&big_path).expect("read big.txt");
    let trunc_input_50k: String = big_data.chars().take(50_000).collect();
    let (bpe_truncation_512, bpe_truncation_128) = if let Some(ref tok) = bpe_tokenizer {
        let t512 = bench_truncation("BPE Truncation 512", tok, &trunc_input_50k, 512, "right", 100);
        let t128 = bench_truncation("BPE Truncation 128", tok, &trunc_input_50k, 128, "right", 100);
        (Some(t512), Some(t128))
    } else {
        (None, None)
    };

    // ── Serialization（从文件加载） ──
    let (ser_load, ser_save, ser_deser) = if let Some(ref tok) = bpe_tokenizer {
        let (l, s, d) = bench_serialization("Serialization", tok, 100);
        (Some(l), Some(s), Some(d))
    } else {
        (None, None, None)
    };

    // ── 输出 JSON ──
    let rustc_version = std::process::Command::new("rustc").arg("--version").output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());
    let cargo_version = std::process::Command::new("cargo").arg("--version").output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    let perf = PerfResults {
        mode: "Rust".to_string(),
        platform: format!("{}-bit", std::mem::size_of::<usize>() * 8),
        timestamp: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs().to_string(),
        rustc_version,
        cargo_version,
        compile_settings: "release (lto=fat, codegen-units=1, target-cpu=native)".to_string(),
        bpe_encode_single,
        bpe_encode_batch,
        bpe_encode_fast,
        unigram_encode_single,
        unigram_encode_batch,
        wordpiece_encode_single: wp_encode,
        wordpiece_encode_batch: wp_encode_batch,
        qwen25_encode_single: qwen_encode,
        qwen25_encode_batch: qwen_batch,
        wordpiece_encode_fast: wp_encode_fast,
        wordlevel_encode_single: wordlevel_encode,
        wordlevel_encode_batch: wordlevel_encode_batch,
        bpe_encode_char_offsets,
        unigram_encode_char_offsets,
        wordpiece_encode_char_offsets,
        bpe_concurrent_1t,
        bpe_concurrent_2t,
        bpe_concurrent_4t,
        bpe_concurrent_8t,
        bpe_truncation_512,
        bpe_truncation_128,
        serialization_load: ser_load,
        serialization_save: ser_save,
        serialization_deserialize: ser_deser,
    };

    let perf_json = serde_json::to_string_pretty(&perf).expect("serialize perf");
    let perf_out = format!("{}/perf-rust.json", data_dir);
    fs::write(&perf_out, &perf_json).expect("write perf-rust");
    println!("\n  ✅ 性能数据 → {}", perf_out);
}

criterion::criterion_group!(benches, bench_perf);
criterion::criterion_main!(benches);
