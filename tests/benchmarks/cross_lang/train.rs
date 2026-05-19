//! 训练函数 — BPE / Unigram / WordPiece / WordLevel

use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::models::unigram::{Unigram, UnigramTrainer};
use tokenizers::models::wordpiece::{WordPiece, WordPieceTrainerBuilder};
use tokenizers::models::wordlevel::{WordLevel, WordLevelTrainer};
use tokenizers::models::TrainerWrapper;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::Tokenizer;

pub fn train_bpe(corpus_path: &str, vocab_size: usize, min_freq: u32) -> Tokenizer {
    let mut trainer: TrainerWrapper = BpeTrainerBuilder::default()
        .show_progress(false)
        .vocab_size(vocab_size)
        .min_frequency(min_freq as u64)
        .build()
        .into();
    let mut tok = Tokenizer::new(BPE::default()).into_inner();
    tok.with_pre_tokenizer(Some(Whitespace {}));
    tok.train_from_files(&mut trainer, vec![corpus_path.to_string()])
        .expect("train bpe");
    tok.into()
}

pub fn train_unigram(corpus_path: &str, vocab_size: usize) -> Tokenizer {
    let trainer: UnigramTrainer = UnigramTrainer::builder()
        .show_progress(false)
        .vocab_size(vocab_size as u32)
        .build()
        .expect("build unigram trainer");
    let mut trainer_wrapper: TrainerWrapper = trainer.into();
    let mut tok = Tokenizer::new(Unigram::default()).into_inner();
    tok.with_pre_tokenizer(Some(Whitespace {}));
    tok.train_from_files(&mut trainer_wrapper, vec![corpus_path.to_string()])
        .expect("train unigram");
    tok.into()
}

pub fn train_wordpiece(corpus_path: &str, vocab_size: usize, min_freq: u32) -> Tokenizer {
    // 通过 special_tokens 让 trainer 将 [UNK] 加入模型词表
    let mut trainer: TrainerWrapper = WordPieceTrainerBuilder::default()
        .show_progress(false)
        .vocab_size(vocab_size)
        .min_frequency(min_freq as u64)
        .special_tokens(vec![tokenizers::AddedToken::from("[UNK]", true)])
        .build()
        .into();
    let mut tok = Tokenizer::new(WordPiece::default()).into_inner();
    tok.with_pre_tokenizer(Some(Whitespace {}));
    tok.train_from_files(&mut trainer, vec![corpus_path.to_string()])
        .expect("train wordpiece");
    tok.into()
}

pub fn train_wordlevel(corpus_path: &str, vocab_size: usize) -> Tokenizer {
    // 通过 special_tokens 让 trainer 将 <unk> 加入模型词表
    let mut trainer: TrainerWrapper = WordLevelTrainer::builder()
        .show_progress(false)
        .vocab_size(vocab_size)
        .special_tokens(vec![tokenizers::AddedToken::from("<unk>", true)])
        .build()
        .unwrap()
        .into();
    let mut tok = Tokenizer::new(WordLevel::default()).into_inner();
    tok.with_pre_tokenizer(Some(Whitespace {}));
    tok.train_from_files(&mut trainer, vec![corpus_path.to_string()])
        .expect("train wordlevel");
    tok.into()
}
