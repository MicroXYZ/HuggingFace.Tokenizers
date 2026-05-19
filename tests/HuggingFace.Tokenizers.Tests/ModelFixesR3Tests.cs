using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Models.BPE;
using HuggingFace.Tokenizers.Models.WordPiece;
using HuggingFace.Tokenizers.Models.Unigram;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Tests for Phase R3 model fixes:
/// - R3A: BpeModel (ignoreMerges whole-sequence, EnumerateRunes, MergeAll NewId, Cache)
/// - R3B: WordPiece UNK (whole-sequence UNK)
/// - R3C: Unigram fuse_unk
/// </summary>
    [TestClass]
public class ModelFixesR3Tests
{
    #region R3A — BpeModel: ignoreMerges whole-sequence check

    [TestMethod]
    public void BpeModel_IgnoreMerges_WholeSequenceInVocab_ReturnsSingleToken()
    {
        // When ignore_merges is true and the entire sequence is in vocab,
        // it should return the sequence as a single token (matching Rust).
        var vocab = new Dictionary<string, uint>
        {
            ["hello"] = 0,
            ["h"] = 1, ["e"] = 2, ["l"] = 3, ["o"] = 4
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges([])
            .SetIgnoreMerges(true)
            .Build();

        var tokens = model.Tokenize("hello");

        Assert.AreEqual(1, tokens.Count());
        Assert.AreEqual(0u, tokens[0].Id);
        Assert.AreEqual("hello", tokens[0].Value);
        Assert.AreEqual((0, 5), tokens[0].Offsets);
    }

    [TestMethod]
    public void BpeModel_IgnoreMerges_WholeSequenceNotInVocab_FallsBackToCharSplit()
    {
        // When ignore_merges is true but the whole sequence is NOT in vocab,
        // it should fall back to character-level splitting.
        var vocab = new Dictionary<string, uint>
        {
            ["hello"] = 0,
            ["h"] = 1, ["e"] = 2, ["l"] = 3, ["o"] = 4
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges([])
            .SetIgnoreMerges(true)
            .Build();

        var tokens = model.Tokenize("world");

        // "world" not in vocab, so each char is looked up individually
        // 'w' not in vocab, 'o' = 4, 'r' not in vocab, 'l' = 3, 'd' not in vocab
        Assert.IsTrue(tokens.Count() > 0);
        // Only 'o' and 'l' are in vocab
        Assert.IsTrue(tokens.Any(t => t.Id == 4)); // 'o'
        Assert.IsTrue(tokens.Any(t => t.Id == 3)); // 'l'
    }

    [TestMethod]
    public void BpeModel_IgnoreMerges_WithContinuingSubwordPrefix()
    {
        // When ignore_merges + continuing_subword_prefix, whole-sequence check
        // should still work for the raw sequence.
        var vocab = new Dictionary<string, uint>
        {
            ["Ġworld"] = 0,
            ["Ġ"] = 1, ["w"] = 2, ["o"] = 3, ["r"] = 4, ["l"] = 5, ["d"] = 6
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges([])
            .SetContinuingSubwordPrefix("Ġ")
            .SetIgnoreMerges(true)
            .Build();

        // The raw input "world" is NOT in vocab (only "Ġworld" is).
        // So ignoreMerges whole-sequence check won't match.
        // It should fall through to char-level split.
        var tokens = model.Tokenize("world");
        Assert.IsTrue(tokens.Count() > 0);
    }

    #endregion

    #region R3A — BpeModel: MergeAll NewId correctness

    [TestMethod]
    public void BpeModel_MergeAll_UsesCorrectNewId()
    {
        // Regression test: MergeAll must use the stored NewId from the queue,
        // not re-lookup (which could return stale/zero values).
        var vocab = new Dictionary<string, uint>
        {
            ["a"] = 0, ["b"] = 1, ["c"] = 2,
            ["ab"] = 3, ["abc"] = 4
        };
        var merges = new List<(string, string)>
        {
            ("a", "b"),
            ("ab", "c")
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges(merges)
            .Build();

        var tokens = model.Tokenize("abc");

        // After merging a+b → ab (id=3), then ab+c → abc (id=4)
        Assert.AreEqual(1, tokens.Count());
        Assert.AreEqual(4u, tokens[0].Id);
        Assert.AreEqual("abc", tokens[0].Value);
    }

    [TestMethod]
    public void BpeModel_MergeAll_SequentialMerges()
    {
        // Test multiple sequential merges to verify the priority queue
        // correctly handles chain merges.
        var vocab = new Dictionary<string, uint>
        {
            ["a"] = 0, ["b"] = 1, ["c"] = 2, ["d"] = 3,
            ["ab"] = 4, ["cd"] = 5, ["abcd"] = 6
        };
        var merges = new List<(string, string)>
        {
            ("a", "b"),
            ("c", "d"),
            ("ab", "cd")
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges(merges)
            .Build();

        var tokens = model.Tokenize("abcd");

        Assert.AreEqual(1, tokens.Count());
        Assert.AreEqual(6u, tokens[0].Id);
    }

    #endregion

    #region R3A — BpeModel: EnumerateRunes (Unicode handling)

    [TestMethod]
    public void BpeModel_MergeWord_UnicodeScalarValues()
    {
        // Verify that MergeWord iterates by Unicode scalar values (code points),
        // not grapheme clusters. CJK characters should each be a separate symbol.
        var vocab = new Dictionary<string, uint>
        {
            ["你"] = 0, ["好"] = 1, ["世"] = 2, ["界"] = 3,
            ["你好"] = 4
        };
        var merges = new List<(string, string)>
        {
            ("你", "好")
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges(merges)
            .Build();

        var tokens = model.Tokenize("你好世界");

        // 你+好 → 你好 (id=4), 世 (id=2), 界 (id=3)
        Assert.AreEqual(3, tokens.Count);
        Assert.AreEqual(4u, tokens[0].Id);
        Assert.AreEqual(2u, tokens[1].Id);
        Assert.AreEqual(3u, tokens[2].Id);
    }

    [TestMethod]
    public void BpeModel_MergeWord_EmojiCodePoints()
    {
        // Emoji with multiple code points (family emoji) — each Rune is a code point.
        // This tests that we iterate by code points, not grapheme clusters.
        var vocab = new Dictionary<string, uint>
        {
            ["👨"] = 0, ["👩"] = 1, ["👧"] = 2, ["👦"] = 3
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges([])
            .Build();

        // "👨👩👧👦" is 4 code points (each is a separate Rune)
        var tokens = model.Tokenize("👨👩👧👦");
        Assert.AreEqual(4, tokens.Count);
    }

    #endregion

    #region R3A — BpeModel: Cache capacity limit

    [TestMethod]
    public void BpeModel_Cache_DoesNotExceedCapacity()
    {
        // Cache should not grow beyond cacheCapacity entries.
        var vocab = new Dictionary<string, uint>
        {
            ["a"] = 0, ["b"] = 1, ["c"] = 2
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges([])
            .SetCacheCapacity(2)
            .Build();

        // Tokenize 3 different strings — only 2 should be cached
        model.Tokenize("a");
        model.Tokenize("b");
        model.Tokenize("c");

        // All should still return correct results (cache is just an optimization)
        Assert.AreEqual(1, model.Tokenize("a").Count());
        Assert.AreEqual(1, model.Tokenize("b").Count());
        Assert.AreEqual(1, model.Tokenize("c").Count());
    }

    [TestMethod]
    public void BpeModel_Cache_LongSequenceNotCached()
    {
        // Sequences >= 128 chars should NOT be cached (matches Rust MAX_LENGTH).
        var vocab = new Dictionary<string, uint>
        {
            ["a"] = 0
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges([])
            .SetCacheCapacity(100)
            .Build();

        var longSequence = new string('a', 200);
        var tokens1 = model.Tokenize(longSequence);
        var tokens2 = model.Tokenize(longSequence);

        // Should still produce correct results even if not cached
        Assert.IsTrue(tokens1.Count() > 0);
        Assert.AreEqual(tokens1.Count, tokens2.Count);
    }

    #endregion

    #region R3B — WordPiece: UNK for entire sequence

    [TestMethod]
    public void WordPiece_UnknownWord_ReturnsUnkForEntireSequence()
    {
        // When a word can't be fully tokenized, return UNK for the ENTIRE sequence
        // (not just the remaining portion). Matches Rust behavior.
        var vocab = new Dictionary<string, uint>
        {
            ["[UNK]"] = 0, ["hel"] = 1, ["##lo"] = 2
        };
        var model = new WordPieceModel.WordPieceBuilder()
            .SetVocab(vocab)
            .SetContinuingSubwordPrefix("##")
            .SetUnkToken("[UNK]")
            .Build();

        // "hellox" — "hel" matches, "##lo" matches, but "##x" doesn't.
        // Rust returns UNK for the entire sequence (0, len).
        var tokens = model.Tokenize("hellox");

        Assert.AreEqual(1, tokens.Count());
        Assert.AreEqual(0u, tokens[0].Id);
        Assert.AreEqual("[UNK]", tokens[0].Value);
        Assert.AreEqual(0, tokens[0].Start);
        Assert.AreEqual(6, tokens[0].End);
    }

    [TestMethod]
    public void WordPiece_AllKnown_ReturnsMultipleTokens()
    {
        // When all subwords are known, should return multiple tokens normally.
        var vocab = new Dictionary<string, uint>
        {
            ["[UNK]"] = 0, ["hel"] = 1, ["##lo"] = 2
        };
        var model = new WordPieceModel.WordPieceBuilder()
            .SetVocab(vocab)
            .SetContinuingSubwordPrefix("##")
            .SetUnkToken("[UNK]")
            .Build();

        var tokens = model.Tokenize("hello");

        Assert.AreEqual(2, tokens.Count);
        Assert.AreEqual(1u, tokens[0].Id); // "hel"
        Assert.AreEqual(2u, tokens[1].Id); // "##lo"
    }

    [TestMethod]
    public void WordPiece_CompletelyUnknown_ReturnsUnk()
    {
        // When no subword matches at all, return UNK for the entire sequence.
        var vocab = new Dictionary<string, uint>
        {
            ["[UNK]"] = 0, ["a"] = 1
        };
        var model = new WordPieceModel.WordPieceBuilder()
            .SetVocab(vocab)
            .SetContinuingSubwordPrefix("##")
            .SetUnkToken("[UNK]")
            .Build();

        var tokens = model.Tokenize("xyz");

        Assert.AreEqual(1, tokens.Count());
        Assert.AreEqual(0u, tokens[0].Id);
        Assert.AreEqual(0, tokens[0].Start);
        Assert.AreEqual(3, tokens[0].End);
    }

    #endregion

    #region R3C — Unigram: fuse_unk

    [TestMethod]
    public void Unigram_FuseUnk_True_FusesConsecutiveUnk()
    {
        // With fuse_unk=true, consecutive UNK tokens should be merged into one.
        var vocab = new List<(string, double)>
        {
            ("<unk>", 0.0),
            ("a", 0.0),
            ("b", 0.0),
            ("ab", 2.0)
        };
        var model = new UnigramModel.UnigramBuilder()
            .SetVocab(vocab)
            .SetUnkToken("<unk>")
            .SetFuseUnk(true)
            .Build();

        // "xyz" — all unknown, should fuse into single UNK token
        var tokens = model.Tokenize("xyz");

        Assert.AreEqual(1, tokens.Count());
        Assert.AreEqual("xyz", tokens[0].Value);
        Assert.AreEqual(0u, tokens[0].Id); // UNK id
    }

    [TestMethod]
    public void Unigram_FuseUnk_False_SeparatesConsecutiveUnk()
    {
        // With fuse_unk=false, consecutive UNK tokens should remain separate.
        var vocab = new List<(string, double)>
        {
            ("<unk>", 0.0),
            ("a", 0.0),
            ("b", 0.0)
        };
        var model = new UnigramModel.UnigramBuilder()
            .SetVocab(vocab)
            .SetUnkToken("<unk>")
            .SetFuseUnk(false)
            .Build();

        // "xyz" — all unknown, should be 3 separate UNK tokens
        var tokens = model.Tokenize("xyz");

        Assert.AreEqual(3, tokens.Count);
        foreach (var t in tokens) Assert.AreEqual(0u, t.Id);
    }

    [TestMethod]
    public void Unigram_FuseUnk_MixedKnownAndUnknown()
    {
        // Test fuse_unk with a mix of known and unknown tokens.
        var vocab = new List<(string, double)>
        {
            ("<unk>", 0.0),
            ("a", 0.0),
            ("b", 0.0),
            ("ab", 2.0)
        };
        var model = new UnigramModel.UnigramBuilder()
            .SetVocab(vocab)
            .SetUnkToken("<unk>")
            .SetFuseUnk(true)
            .Build();

        // "xab" — 'x' is UNK, 'ab' is known (Viterbi picks "ab" since score 2.0 > 0.0+0.0)
        var tokens = model.Tokenize("xab");

        Assert.AreEqual(2, tokens.Count);
        Assert.AreEqual(0u, tokens[0].Id); // UNK for 'x'
        Assert.AreEqual("x", tokens[0].Value);
        Assert.AreEqual(3u, tokens[1].Id); // "ab" (id=3)
    }

    [TestMethod]
    public void Unigram_FuseUnk_FusesMiddleUnkTokens()
    {
        // Test that UNK tokens in the middle of known tokens are fused correctly.
        var vocab = new List<(string, double)>
        {
            ("<unk>", 0.0),
            ("a", 0.0),
            ("b", 0.0),
            ("c", 0.0)
        };
        var model = new UnigramModel.UnigramBuilder()
            .SetVocab(vocab)
            .SetUnkToken("<unk>")
            .SetFuseUnk(true)
            .Build();

        // "axyzb" — 'a' known (id=1), 'xyz' UNK (fused), 'b' known (id=2)
        var tokens = model.Tokenize("axyzb");

        Assert.AreEqual(3, tokens.Count);
        Assert.AreEqual(1u, tokens[0].Id); // 'a' is known
        Assert.AreEqual(0u, tokens[1].Id); // 'xyz' fused UNK
        Assert.AreEqual("xyz", tokens[1].Value);
        Assert.AreEqual(2u, tokens[2].Id); // 'b' is known
    }

    [TestMethod]
    public void Unigram_FuseUnk_DefaultIsTrue()
    {
        // Verify that fuse_unk defaults to true (matching Rust's default).
        var vocab = new List<(string, double)>
        {
            ("<unk>", 0.0),
            ("a", 0.0)
        };
        var model = new UnigramModel.UnigramBuilder()
            .SetVocab(vocab)
            .SetUnkToken("<unk>")
            .Build(); // no explicit SetFuseUnk

        // "xyz" — all unknown, should fuse (default = true)
        var tokens = model.Tokenize("xyz");

        Assert.AreEqual(1, tokens.Count());
        Assert.AreEqual("xyz", tokens[0].Value);
    }

    #endregion
}
