using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Models.BPE;
using HuggingFace.Tokenizers.Normalizers;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Tests for AddedVocabulary functionality.
/// </summary>
    [TestClass]
public class AddedVocabularyTests
{
    private static (Tokenizer Tokenizer, BpeModel Model) CreateTokenizer()
    {
        var vocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0, ["hello"] = 1, ["world"] = 2, [" "] = 3
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges([])
            .SetUnkToken("<unk>")
            .Build();

        var tokenizer = new Tokenizer(model);
        return (tokenizer, model);
    }

    [TestMethod]
    public void AddSpecialTokens_AssignsIdsStartingFromVocabSize()
    {
        var (tokenizer, model) = CreateTokenizer();

        tokenizer.AddToken(new AddedToken("[CLS]", isSpecial: true));
        tokenizer.AddToken(new AddedToken("[SEP]", isSpecial: true));

        // Vocab size = 4, so added tokens get ids 4 and 5
        Assert.AreEqual(4u, tokenizer.TokenToId("[CLS]"));
        Assert.AreEqual(5u, tokenizer.TokenToId("[SEP]"));
    }

    [TestMethod]
    public void AddTokens_RegularTokens_AssignsIdsFromVocabSize()
    {
        var (tokenizer, model) = CreateTokenizer();

        tokenizer.AddToken(new AddedToken("custom_word"));
        tokenizer.AddToken(new AddedToken("another_word"));

        // Vocab size = 4, regular tokens get ids 4 and 5
        Assert.AreEqual(4u, tokenizer.TokenToId("custom_word"));
        Assert.AreEqual(5u, tokenizer.TokenToId("another_word"));
    }

    [TestMethod]
    public void AddTokens_SpecialAndRegular_IdsDoNotCollide()
    {
        var (tokenizer, model) = CreateTokenizer();

        tokenizer.AddToken(new AddedToken("[CLS]", isSpecial: true));
        tokenizer.AddToken(new AddedToken("custom"));

        // [CLS] gets id 4, "custom" gets id 5
        Assert.AreEqual(4u, tokenizer.TokenToId("[CLS]"));
        Assert.AreEqual(5u, tokenizer.TokenToId("custom"));
    }

    [TestMethod]
    public void TokenToId_AddedTokenOverridesModel()
    {
        var (tokenizer, model) = CreateTokenizer();

        // "hello" exists in model vocab with id 1
        Assert.AreEqual(1u, tokenizer.TokenToId("hello"));

        // Add "hello" as an added token — should get a new id
        tokenizer.AddToken(new AddedToken("hello", isSpecial: true));

        // Added tokens are checked first, so "hello" now returns the added id
        var id = tokenizer.TokenToId("hello");
        Assert.IsNotNull(id);
        Assert.AreEqual(4u, id); // vocab size = 4, so first added token gets id 4
    }

    [TestMethod]
    public void IdToToken_AddedToken_ReturnsContent()
    {
        var (tokenizer, model) = CreateTokenizer();

        tokenizer.AddToken(new AddedToken("[PAD]", isSpecial: true));

        var token = tokenizer.IdToToken(4u);
        Assert.AreEqual("[PAD]", token);
    }

    [TestMethod]
    public void IsSpecialToken_ReturnsTrueForSpecialTokens()
    {
        var (tokenizer, model) = CreateTokenizer();

        tokenizer.AddToken(new AddedToken("[CLS]", isSpecial: true));
        tokenizer.AddToken(new AddedToken("regular"));

        Assert.IsTrue(tokenizer.AddedVocabulary.IsSpecialToken("[CLS]"));
        Assert.IsFalse(tokenizer.AddedVocabulary.IsSpecialToken("regular"));
    }

    [TestMethod]
    public void IsSpecialToken_ReturnsFalseForUnknownTokens()
    {
        var (tokenizer, model) = CreateTokenizer();

        Assert.IsFalse(tokenizer.AddedVocabulary.IsSpecialToken("[UNKNOWN]"));
    }

    [TestMethod]
    public void AddDuplicateToken_DoesNotAssignNewId()
    {
        var (tokenizer, model) = CreateTokenizer();

        tokenizer.AddToken(new AddedToken("[CLS]", isSpecial: true));
        var firstId = tokenizer.TokenToId("[CLS]");

        // Adding the same token again should not change the id
        tokenizer.AddToken(new AddedToken("[CLS]", isSpecial: true));
        var secondId = tokenizer.TokenToId("[CLS]");

        Assert.AreEqual(firstId, secondId);
    }

    [TestMethod]
    public void LStrip_RStrip_TokenWithSpaces_MatchesCorrectly()
    {
        var (tokenizer, model) = CreateTokenizer();

        // Must enable EncodeSpecialTokens BEFORE adding special tokens so the regex includes them
        tokenizer.EncodeSpecialTokens = true;
        tokenizer.AddToken(new AddedToken("[SEP]", isSpecial: true, lStrip: true, rStrip: true));

        // The regex should match [SEP] with optional surrounding whitespace
        var encoding = tokenizer.Encode("hello [SEP] world", addSpecialTokens: true);

        Assert.IsTrue(encoding.Length > 0);
        // [SEP] should appear in the tokens
        var tokens = encoding.GetTokens();
        CollectionAssert.Contains(tokens, "[SEP]");
    }

    [TestMethod]
    public void SpecialTokens_EncodeSpecialTokensFalse_NotMatched()
    {
        var (tokenizer, model) = CreateTokenizer();

        tokenizer.AddToken(new AddedToken("[CLS]", isSpecial: true));

        // With addSpecialTokens=false, [CLS] in text should not be matched as special
        var encoding = tokenizer.Encode("[CLS]", addSpecialTokens: false);

        // [CLS] should NOT be in the tokens since addSpecialTokens=false
        var tokens = encoding.GetTokens();
        CollectionAssert.DoesNotContain(tokens.ToList(), "[CLS]");
    }

    [TestMethod]
    public void SpecialTokens_EncodeSpecialTokensTrue_Matched()
    {
        var (tokenizer, model) = CreateTokenizer();

        // Must enable EncodeSpecialTokens BEFORE adding special tokens so the regex includes them
        tokenizer.EncodeSpecialTokens = true;
        tokenizer.AddToken(new AddedToken("[CLS]", isSpecial: true));

        var encoding = tokenizer.Encode("[CLS]", addSpecialTokens: true);

        var tokens = encoding.GetTokens();
        CollectionAssert.Contains(tokens, "[CLS]");
    }

    [TestMethod]
    public void GetAddedTokens_ReturnsAllAddedTokens()
    {
        var (tokenizer, model) = CreateTokenizer();

        tokenizer.AddToken(new AddedToken("[CLS]", isSpecial: true));
        tokenizer.AddToken(new AddedToken("[SEP]", isSpecial: true));
        tokenizer.AddToken(new AddedToken("custom"));

        var added = tokenizer.AddedVocabulary.GetAddedTokens();
        Assert.AreEqual(3, added.Count);
        Assert.IsTrue(added.ContainsKey("[CLS]"));
        Assert.IsTrue(added.ContainsKey("[SEP]"));
        Assert.IsTrue(added.ContainsKey("custom"));
    }

    [TestMethod]
    public void AddTokens_BatchAssignsSequentialIds()
    {
        var (tokenizer, model) = CreateTokenizer();

        tokenizer.AddTokens([
            new AddedToken("[CLS]", isSpecial: true),
            new AddedToken("[SEP]", isSpecial: true),
            new AddedToken("[PAD]", isSpecial: true),
        ]);

        Assert.AreEqual(4u, tokenizer.TokenToId("[CLS]"));
        Assert.AreEqual(5u, tokenizer.TokenToId("[SEP]"));
        Assert.AreEqual(6u, tokenizer.TokenToId("[PAD]"));
    }

    // ===================================================================
    // Two-phase split tests (Phase R2A)
    // ===================================================================

    /// <summary>
    /// Helper: extract split results as (text, hasToken) pairs for easy assertion.
    /// </summary>
    private static List<(string Text, bool HasToken, uint? Id)> GetSplitResults(
        AddedVocabulary vocab, INormalizer? normalizer, string text)
    {
        var pretokenized = vocab.ExtractAndNormalize(normalizer, text);
        return pretokenized.GetSplits()
            .Select(s => (
                s.Normalized.Get(),
                s.Tokens is not null,
                s.Tokens is { Count: > 0 } ? (uint?)s.Tokens[0].Id : null))
            .ToList();
    }

    [TestMethod]
    public void Phase1_NonNormalizedSpecialToken_MatchedOnRawText()
    {
        // Special tokens default to normalized=false via AddedToken.From
        // They should be matched against the raw (un-normalized) text in Phase 1.
        var vocab = new AddedVocabulary();
        var model = new BpeModel.BpeBuilder()
            .SetVocab(new Dictionary<string, uint> { ["<unk>"] = 0 })
            .SetMerges([])
            .SetUnkToken("<unk>")
            .Build();
        var normalizer = new LowercaseNormalizer();

        // Must enable EncodeSpecialTokens so special tokens are included in the regex
        vocab.EncodeSpecialTokens = true;
        // AddedToken.From("[CLS]", isSpecial: true) → normalized = false
        vocab.AddSpecialTokens(
            [AddedToken.From("[CLS]", isSpecial: true)],
            model, normalizer);

        // "[CLS]" should be matched in raw text BEFORE normalization
        // even though the normalizer lowercases everything.
        var results = GetSplitResults(vocab, normalizer, "[CLS] Hello World");

        Assert.AreEqual(2, results.Count);
        // [CLS] matched as non-normalized token (Phase 1)
        Assert.AreEqual("[CLS]", results[0].Text);
        Assert.IsTrue(results[0].HasToken);
        // " Hello World" normalized → " hello world"
        Assert.AreEqual(" hello world", results[1].Text);
        Assert.IsFalse(results[1].HasToken);
    }

    [TestMethod]
    public void Phase2_NormalizedToken_MatchedOnNormalizedText()
    {
        // AddedToken.From("Hello", isSpecial: false) → normalized = true
        // Should be matched AFTER the text is normalized.
        var vocab = new AddedVocabulary();
        var model = new BpeModel.BpeBuilder()
            .SetVocab(new Dictionary<string, uint> { ["<unk>"] = 0 })
            .SetMerges([])
            .SetUnkToken("<unk>")
            .Build();
        var normalizer = new LowercaseNormalizer();

        // normalized=true (default for From with isSpecial=false)
        vocab.AddTokens(
            [AddedToken.From("Hello", isSpecial: false)],
            model, normalizer);

        // "HELLO" should be normalized to "hello" and then matched
        // because the token "Hello" normalizes to "hello"
        var results = GetSplitResults(vocab, normalizer, "say HELLO world");

        Assert.AreEqual(3, results.Count);
        Assert.AreEqual("say ", results[0].Text);
        Assert.IsFalse(results[0].HasToken);
        Assert.AreEqual("hello", results[1].Text);
        Assert.IsTrue(results[1].HasToken);
        Assert.AreEqual(" world", results[2].Text);
        Assert.IsFalse(results[2].HasToken);
    }

    [TestMethod]
    public void TwoPhase_MixedNormalizedAndNonNormalized()
    {
        // Mix of normalized=false special tokens and normalized=true regular tokens.
        // Phase 1: match non-normalized tokens on raw text
        // Phase 2: normalize non-token parts, match normalized tokens
        var vocab = new AddedVocabulary();
        var model = new BpeModel.BpeBuilder()
            .SetVocab(new Dictionary<string, uint> { ["<unk>"] = 0 })
            .SetMerges([])
            .SetUnkToken("<unk>")
            .Build();
        var normalizer = new LowercaseNormalizer();

        vocab.EncodeSpecialTokens = true;
        // [CLS] is special → normalized=false (from Phase 1)
        vocab.AddSpecialTokens(
            [AddedToken.From("[CLS]", isSpecial: true)],
            model, normalizer);
        // "Hello" is regular → normalized=true (from Phase 2)
        vocab.AddTokens(
            [AddedToken.From("Hello", isSpecial: false)],
            model, normalizer);

        // "[CLS] HELLO world"
        // Phase 1: [CLS] matched on raw text → ["[CLS]", " HELLO world"]
        // Phase 2: " HELLO world" → normalize → " hello world" → match "hello"
        //   → [" ", "hello", " world"]
        var results = GetSplitResults(vocab, normalizer, "[CLS] HELLO world");

        Assert.AreEqual(4, results.Count);
        // [CLS] from Phase 1 (non-normalized, matched on raw text)
        Assert.AreEqual("[CLS]", results[0].Text);
        Assert.IsTrue(results[0].HasToken);
        // " " (space before HELLO, normalized)
        Assert.AreEqual(" ", results[1].Text);
        Assert.IsFalse(results[1].HasToken);
        // "hello" from Phase 2 (normalized token, matched on normalized text)
        Assert.AreEqual("hello", results[2].Text);
        Assert.IsTrue(results[2].HasToken);
        // " world" (remaining text, normalized)
        Assert.AreEqual(" world", results[3].Text);
        Assert.IsFalse(results[3].HasToken);
    }

    [TestMethod]
    public void TwoPhase_LStripRStrip_MergeWithBehavior()
    {
        // Test LStrip/RStrip behavior in the two-phase split.
        // LStrip/RStrip should consume surrounding whitespace.
        var vocab = new AddedVocabulary();
        var model = new BpeModel.BpeBuilder()
            .SetVocab(new Dictionary<string, uint> { ["<unk>"] = 0 })
            .SetMerges([])
            .SetUnkToken("<unk>")
            .Build();
        var normalizer = new LowercaseNormalizer();

        vocab.EncodeSpecialTokens = true;
        // Non-normalized special token with lstrip/rstrip
        vocab.AddSpecialTokens(
            [new AddedToken("[SEP]", isSpecial: true, lStrip: true, rStrip: true, normalized: false)],
            model, normalizer);
        // Normalized regular token with lstrip/rstrip
        vocab.AddTokens(
            [new AddedToken("hello", isSpecial: false, lStrip: true, rStrip: true, normalized: true)],
            model, normalizer);

        // "say [SEP] HELLO there"
        // Phase 1: [SEP] matched with lstrip/rstrip on raw text
        //   → ["say", " [SEP] ", "HELLO there"]
        //   Wait, lstrip on [SEP] means it consumes leading whitespace.
        //   The regex pattern: (\s*\[SEP\]\s*)
        //   "say [SEP] HELLO there" → match " [SEP] " at positions 3..10
        //   → splits: ["say" (0..3), " [SEP] " (3..10), "HELLO there" (10..21)]
        // Phase 2: "say" → normalize → "say" (no normalized token match)
        //          "HELLO there" → normalize → "hello there" → match "hello " with lstrip/rstrip
        //   The regex for "hello" with lstrip/rstrip: (\s*hello\s*)
        //   "hello there" → match "hello " at positions 0..6
        //   → splits: ["hello " (0..6), "there" (6..11)]

        var results = GetSplitResults(vocab, normalizer, "say [SEP] HELLO there");

        // Expected:
        // "say" - non-token from Phase 1, not further split in Phase 2
        // " [SEP] " - token from Phase 1 (lstrip/rstrip consumed spaces)
        // "hello " - token from Phase 2 (lstrip/rstrip consumed trailing space)
        // "there" - non-token from Phase 2
        Assert.IsTrue(results.Count >= 3);

        // [SEP] should be found
        var sepResult = results.FirstOrDefault(r => r.Text.Contains("[SEP]") && r.HasToken);
        Assert.IsTrue(sepResult.HasToken);
        StringAssert.Contains(sepResult.Text, "[SEP]");

        // "hello" should be found (normalized from "HELLO")
        var helloResult = results.FirstOrDefault(r => r.Text.Contains("hello") && r.HasToken);
        Assert.IsTrue(helloResult.HasToken);
        StringAssert.Contains(helloResult.Text, "hello");
    }

    [TestMethod]
    public void TwoPhase_AllNormalizedTokens()
    {
        // When all tokens are normalized=true, Phase 1 has nothing to match.
        // Phase 2 normalizes everything and matches.
        var vocab = new AddedVocabulary();
        var model = new BpeModel.BpeBuilder()
            .SetVocab(new Dictionary<string, uint> { ["<unk>"] = 0 })
            .SetMerges([])
            .SetUnkToken("<unk>")
            .Build();
        var normalizer = new LowercaseNormalizer();

        vocab.AddTokens(
            [
                AddedToken.From("hello", isSpecial: false),
                AddedToken.From("world", isSpecial: false),
            ],
            model, normalizer);

        var results = GetSplitResults(vocab, normalizer, "HELLO WORLD");

        Assert.AreEqual(3, results.Count);
        Assert.AreEqual("hello", results[0].Text);
        Assert.IsTrue(results[0].HasToken);
        Assert.AreEqual(" ", results[1].Text);
        Assert.IsFalse(results[1].HasToken);
        Assert.AreEqual("world", results[2].Text);
        Assert.IsTrue(results[2].HasToken);
    }

    [TestMethod]
    public void TwoPhase_AllNonNormalizedTokens()
    {
        // When all tokens are normalized=false, Phase 1 does all the work.
        // Phase 2 has nothing to match.
        var vocab = new AddedVocabulary();
        var model = new BpeModel.BpeBuilder()
            .SetVocab(new Dictionary<string, uint> { ["<unk>"] = 0 })
            .SetMerges([])
            .SetUnkToken("<unk>")
            .Build();
        var normalizer = new LowercaseNormalizer();

        vocab.EncodeSpecialTokens = true;
        vocab.AddSpecialTokens(
            [
                new AddedToken("[CLS]", isSpecial: true, normalized: false),
                new AddedToken("[SEP]", isSpecial: true, normalized: false),
            ],
            model, normalizer);

        var results = GetSplitResults(vocab, normalizer, "[CLS] hello [SEP]");

        Assert.AreEqual(3, results.Count);
        Assert.AreEqual("[CLS]", results[0].Text);
        Assert.IsTrue(results[0].HasToken);
        Assert.AreEqual(" hello ", results[1].Text);
        Assert.IsFalse(results[1].HasToken);
        // The " hello " text gets normalized to " hello " (already lowercase)
        Assert.AreEqual("[SEP]", results[2].Text);
        Assert.IsTrue(results[2].HasToken);
    }

    [TestMethod]
    public void AddedToken_From_DefaultNormalized()
    {
        // AddedToken.From("x", isSpecial: true) → normalized = false
        // AddedToken.From("x", isSpecial: false) → normalized = true
        var specialToken = AddedToken.From("[CLS]", isSpecial: true);
        Assert.IsFalse(specialToken.Normalized);
        Assert.IsTrue(specialToken.IsSpecial);

        var regularToken = AddedToken.From("hello", isSpecial: false);
        Assert.IsTrue(regularToken.Normalized);
        Assert.IsFalse(regularToken.IsSpecial);
    }

    [TestMethod]
    public void TwoPhase_NonNormalizedTokenNotAffectedByNormalizer()
    {
        // A non-normalized token with special characters should be matched
        // on raw text, not affected by normalization.
        var vocab = new AddedVocabulary();
        var model = new BpeModel.BpeBuilder()
            .SetVocab(new Dictionary<string, uint> { ["<unk>"] = 0 })
            .SetMerges([])
            .SetUnkToken("<unk>")
            .Build();
        var normalizer = new NfcNormalizer();

        vocab.EncodeSpecialTokens = true;
        // Non-normalized special token with Unicode content
        vocab.AddSpecialTokens(
            [new AddedToken("café", isSpecial: true, normalized: false)],
            model, normalizer);

        // "café" with composed é should be matched as-is
        var results = GetSplitResults(vocab, normalizer, "I love café today");

        Assert.AreEqual(3, results.Count);
        Assert.AreEqual("I love ", results[0].Text);
        Assert.IsFalse(results[0].HasToken);
        Assert.AreEqual("café", results[1].Text);
        Assert.IsTrue(results[1].HasToken);
        Assert.AreEqual(" today", results[2].Text);
        Assert.IsFalse(results[2].HasToken);
    }

    [TestMethod]
    public void TwoPhase_NormalizedTokenMatchedAfterNfcNormalization()
    {
        // A normalized token should be matched after NFC normalization.
        var vocab = new AddedVocabulary();
        var model = new BpeModel.BpeBuilder()
            .SetVocab(new Dictionary<string, uint> { ["<unk>"] = 0 })
            .SetMerges([])
            .SetUnkToken("<unk>")
            .Build();
        var normalizer = new NfcNormalizer();

        // Normalized regular token — NFC normalizer won't change "hello"
        vocab.AddTokens(
            [AddedToken.From("hello", isSpecial: false)],
            model, normalizer);

        var results = GetSplitResults(vocab, normalizer, "say hello there");

        Assert.AreEqual(3, results.Count);
        Assert.AreEqual("say ", results[0].Text);
        Assert.IsFalse(results[0].HasToken);
        Assert.AreEqual("hello", results[1].Text);
        Assert.IsTrue(results[1].HasToken);
        Assert.AreEqual(" there", results[2].Text);
        Assert.IsFalse(results[2].HasToken);
    }

    [TestMethod]
    public void TwoPhase_EmptyInput()
    {
        var vocab = new AddedVocabulary();
        var model = new BpeModel.BpeBuilder()
            .SetVocab(new Dictionary<string, uint> { ["<unk>"] = 0 })
            .SetMerges([])
            .SetUnkToken("<unk>")
            .Build();
        var normalizer = new LowercaseNormalizer();

        vocab.AddTokens([AddedToken.From("hello", isSpecial: false)], model, normalizer);

        var results = GetSplitResults(vocab, normalizer, "");

        Assert.AreEqual(1, results.Count());
        Assert.AreEqual("", results[0].Text);
        Assert.IsFalse(results[0].HasToken);
    }

    [TestMethod]
    public void TwoPhase_SingleWordConstraint()
    {
        // SingleWord=true should prevent matching inside words.
        var vocab = new AddedVocabulary();
        var model = new BpeModel.BpeBuilder()
            .SetVocab(new Dictionary<string, uint> { ["<unk>"] = 0 })
            .SetMerges([])
            .SetUnkToken("<unk>")
            .Build();
        var normalizer = new LowercaseNormalizer();

        // Non-normalized token with single_word
        vocab.AddTokens(
            [new AddedToken("cat", isSpecial: false, singleWord: true, normalized: false)],
            model, normalizer);

        // "cat" should match standalone but not inside "catalog"
        var results = GetSplitResults(vocab, normalizer, "the cat in the catalog");

        // "cat" should be matched, "catalog" should NOT be split
        var tokenResults = results.Where(r => r.HasToken).ToList();
        Assert.AreEqual(1, tokenResults.Count());
        Assert.AreEqual("cat", tokenResults[0].Text);
    }
}
