using HuggingFace.Tokenizers.Internal;
using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Models.BPE;
using HuggingFace.Tokenizers.Models.WordPiece;
using HuggingFace.Tokenizers.Serialization;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Tests for tokenizer.json serialization round-trip (TokenizerSerializer).
/// Validates that added_tokens, merges, continuing_subword_prefix, and end_of_word_suffix
/// survive serialize → deserialize.
/// </summary>
    [TestClass]
public class TokenizerSerializerTests
{
    // ────────────────────────────────────────────────────────────────────────────
    //  Round-trip: added_tokens
    // ────────────────────────────────────────────────────────────────────────────

    [TestMethod]
    public void Serialize_BpeModel_AddedTokensPresentInJson()
    {
        var tokenizer = TokenizerLoader.FromJson(SimpleBpeJson);

        var json = TokenizerSerializer.Serialize(tokenizer);

        using var doc = JsonDocument.Parse(json);
        Assert.IsTrue(doc.RootElement.TryGetProperty("added_tokens", out var addedTokens));
        Assert.AreEqual(JsonValueKind.Array, addedTokens.ValueKind);

        // Should have at least the <unk> token
        var tokens = addedTokens.EnumerateArray().ToList();
        Assert.IsTrue(tokens.Any(t => t.GetProperty("content").GetString() == "<unk>" &&
            t.GetProperty("special").GetBoolean() == true));
    }

    [TestMethod]
    public void Serialize_WordPiece_AddedTokensRoundTrip()
    {
        // Use a simple WordPiece tokenizer without BertProcessing post-processor
        // (BertProcessing sep/cls serialization is a pre-existing gap)
        var tokenizer = TokenizerLoader.FromJson(SimpleWordPieceNoPostProcessorJson);

        var json = TokenizerSerializer.Serialize(tokenizer);
        var restored = TokenizerLoader.FromJson(json);

        // Encode with both — should produce same result
        var origEncoding = tokenizer.Encode("hello");
        var restoredEncoding = restored.Encode("hello");

        CollectionAssert.AreEqual(origEncoding.GetIds(), restoredEncoding.GetIds());
    }

    [TestMethod]
    public void Serialize_WithAddedTokens_RoundTripPreservesContent()
    {
        // Build a tokenizer and add special tokens manually
        var vocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0, ["hello"] = 1, ["world"] = 2
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .Build();

        var tokenizer = new Tokenizer(model);
        tokenizer.AddToken(new AddedToken("[CLS]", isSpecial: true));
        tokenizer.AddToken(new AddedToken("[SEP]", isSpecial: true));
        tokenizer.AddToken(new AddedToken("custom", isSpecial: false));

        // Serialize
        var json = TokenizerSerializer.Serialize(tokenizer);

        // Deserialize
        var restored = TokenizerLoader.FromJson(json);

        // Verify added tokens survived
        Assert.AreEqual(3u, restored.TokenToId("[CLS]"));
        Assert.AreEqual(4u, restored.TokenToId("[SEP]"));
        Assert.AreEqual(5u, restored.TokenToId("custom"));

        // Verify original tokenizer
        Assert.AreEqual(3u, tokenizer.TokenToId("[CLS]"));
        Assert.AreEqual(4u, tokenizer.TokenToId("[SEP]"));
        Assert.AreEqual(5u, tokenizer.TokenToId("custom"));
    }

    [TestMethod]
    public void Serialize_AddedTokens_SpecialTokensHaveCorrectFlags()
    {
        var vocab = new Dictionary<string, uint> { ["<unk>"] = 0, ["a"] = 1 };
        var model = new BpeModel.BpeBuilder().SetVocab(vocab).Build();

        var tokenizer = new Tokenizer(model);
        tokenizer.AddToken(new AddedToken("[PAD]", isSpecial: true, lStrip: true, rStrip: true));
        tokenizer.AddToken(new AddedToken("foo", isSpecial: false, singleWord: true, normalized: true));

        var json = TokenizerSerializer.Serialize(tokenizer);

        using var doc = JsonDocument.Parse(json);
        var addedTokens = doc.RootElement.GetProperty("added_tokens").EnumerateArray().ToList();

        var pad = addedTokens.First(t => t.GetProperty("content").GetString() == "[PAD]");
        Assert.IsTrue(pad.GetProperty("special").GetBoolean());
        Assert.IsTrue(pad.GetProperty("lstrip").GetBoolean());
        Assert.IsTrue(pad.GetProperty("rstrip").GetBoolean());
        Assert.IsFalse(pad.GetProperty("single_word").GetBoolean());

        var foo = addedTokens.First(t => t.GetProperty("content").GetString() == "foo");
        Assert.IsFalse(foo.GetProperty("special").GetBoolean());
        Assert.IsTrue(foo.GetProperty("single_word").GetBoolean());
        Assert.IsTrue(foo.GetProperty("normalized").GetBoolean());
    }

    // ────────────────────────────────────────────────────────────────────────────
    //  Round-trip: model merges
    // ────────────────────────────────────────────────────────────────────────────

    [TestMethod]
    public void Serialize_BpeModel_MergesPresentInJson()
    {
        var tokenizer = TokenizerLoader.FromJson(SimpleBpeJson);

        var json = TokenizerSerializer.Serialize(tokenizer);

        using var doc = JsonDocument.Parse(json);
        var model = doc.RootElement.GetProperty("model");
        Assert.IsTrue(model.TryGetProperty("merges", out var merges));
        Assert.AreEqual(JsonValueKind.Array, merges.ValueKind);

        var mergeList = merges.EnumerateArray().Select(m => m.GetString()).ToList();
        Assert.IsTrue(mergeList.Count() > 0);
        // All merges should be non-null strings
        foreach (var m in mergeList) Assert.IsNotNull(m);
    }

    [TestMethod]
    public void Serialize_BpeModel_MergesRoundTrip()
    {
        var tokenizer = TokenizerLoader.FromJson(SimpleBpeJson);

        // Serialize
        var json = TokenizerSerializer.Serialize(tokenizer);
        var restored = TokenizerLoader.FromJson(json);

        // Both should encode identically
        var origIds = tokenizer.Encode("hello").GetIds();
        var restoredIds = restored.Encode("hello").GetIds();
        CollectionAssert.AreEqual(origIds, restoredIds);
    }

    // ────────────────────────────────────────────────────────────────────────────
    //  Round-trip: continuing_subword_prefix
    // ────────────────────────────────────────────────────────────────────────────

    [TestMethod]
    public void Serialize_WordPiece_ContinuingSubwordPrefixInJson()
    {
        var tokenizer = TokenizerLoader.FromJson(SimpleWordPieceJson);

        var json = TokenizerSerializer.Serialize(tokenizer);

        using var doc = JsonDocument.Parse(json);
        var model = doc.RootElement.GetProperty("model");
        Assert.IsTrue(model.TryGetProperty("continuing_subword_prefix", out var prefix));
        Assert.AreEqual("##", prefix.GetString());
    }

    [TestMethod]
    public void Serialize_WordPiece_ContinuingSubwordPrefixRoundTrip()
    {
        // Use a simpler WordPiece config without BertProcessing post-processor
        var tokenizer = TokenizerLoader.FromJson(SimpleWordPieceNoPostProcessorJson);

        var json = TokenizerSerializer.Serialize(tokenizer);
        var restored = TokenizerLoader.FromJson(json);

        // WordPiece uses "##" prefix for subwords
        var origIds = tokenizer.Encode("hello worlds").GetIds();
        var restoredIds = restored.Encode("hello worlds").GetIds();
        CollectionAssert.AreEqual(origIds, restoredIds);
    }

    // ────────────────────────────────────────────────────────────────────────────
    //  Round-trip: end_of_word_suffix
    // ────────────────────────────────────────────────────────────────────────────

    [TestMethod]
    public void Serialize_BpeWithSuffix_EndOfWordSuffixInJson()
    {
        var vocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0, ["h"] = 1, ["e"] = 2, ["l"] = 3, ["o"] = 4,
            ["h ello</w>"] = 5
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetEndOfWordSuffix("</w>")
            .Build();

        var tokenizer = new Tokenizer(model);

        var json = TokenizerSerializer.Serialize(tokenizer);

        using var doc = JsonDocument.Parse(json);
        var modelJson = doc.RootElement.GetProperty("model");
        Assert.IsTrue(modelJson.TryGetProperty("end_of_word_suffix", out var suffix));
        Assert.AreEqual("</w>", suffix.GetString());
    }

    [TestMethod]
    public void Serialize_BpeWithSuffix_SuffixSurvivesRoundTrip()
    {
        var vocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0, ["h"] = 1, ["e"] = 2, ["l"] = 3, ["o"] = 4,
            ["h ello</w>"] = 5
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetEndOfWordSuffix("</w>")
            .Build();

        var tokenizer = new Tokenizer(model);

        var json = TokenizerSerializer.Serialize(tokenizer);
        var restored = TokenizerLoader.FromJson(json);

        // Encode with both — should produce same result
        var origIds = tokenizer.Encode("hello").GetIds();
        var restoredIds = restored.Encode("hello").GetIds();
        CollectionAssert.AreEqual(origIds, restoredIds);
    }

    // ────────────────────────────────────────────────────────────────────────────
    //  Round-trip: special tokens behavior
    // ────────────────────────────────────────────────────────────────────────────

    [TestMethod]
    public void Serialize_SpecialTokens_DecodeAfterRoundTrip()
    {
        var tokenizer = TokenizerLoader.FromJson(SimpleBpeJson);

        var json = TokenizerSerializer.Serialize(tokenizer);
        var restored = TokenizerLoader.FromJson(json);

        // Decode the unk token ID — don't skip special tokens
        var decoded = restored.Decode([0], skipSpecialTokens: false);
        StringAssert.Contains(decoded, "<unk>");
    }

    [TestMethod]
    public void Serialize_SpecialTokens_EncodeSpecialTokensFlag()
    {
        var vocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0, ["hello"] = 1, ["world"] = 2
        };
        var model = new BpeModel.BpeBuilder().SetVocab(vocab).Build();

        var tokenizer = new Tokenizer(model);
        tokenizer.AddToken(new AddedToken("[CLS]", isSpecial: true));
        tokenizer.EncodeSpecialTokens = true;

        var json = TokenizerSerializer.Serialize(tokenizer);
        var restored = TokenizerLoader.FromJson(json);

        // Special token should be findable
        Assert.IsNotNull(restored.TokenToId("[CLS]"));
    }

    // ────────────────────────────────────────────────────────────────────────────
    //  Round-trip: complete full-circle
    // ────────────────────────────────────────────────────────────────────────────

    [TestMethod]
    public void Serialize_FullRoundTrip_BpeWithMergesAndAddedTokens()
    {
        // Build a BPE tokenizer with merges and added tokens
        var vocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0, ["h"] = 1, ["e"] = 2, ["l"] = 3, ["o"] = 4,
            ["he"] = 5, ["hel"] = 6, ["hell"] = 7, ["hello"] = 8,
            ["w"] = 9, ["r"] = 10, ["d"] = 11,
            ["wo"] = 12, ["wor"] = 13, ["worl"] = 14, ["world"] = 15
        };
        var merges = new List<(string, string)>
        {
            ("h", "e"), ("he", "l"), ("hel", "l"), ("hell", "o"),
            ("w", "o"), ("wo", "r"), ("wor", "l"), ("worl", "d")
        };

        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges(merges)
            .SetUnkToken("<unk>")
            .Build();

        var tokenizer = new Tokenizer(model);
        tokenizer.AddToken(new AddedToken("[CLS]", isSpecial: true));
        tokenizer.AddToken(new AddedToken("[SEP]", isSpecial: true));

        // Serialize
        var json = TokenizerSerializer.Serialize(tokenizer);

        // Verify JSON structure
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        // version
        Assert.AreEqual("1.0", root.GetProperty("version").GetString());

        // added_tokens
        Assert.IsTrue(root.TryGetProperty("added_tokens", out var addedTokensArr));
        var addedList = addedTokensArr.EnumerateArray().ToList();
        Assert.AreEqual(2, addedList.Count);

        // model
        var modelJson = root.GetProperty("model");
        Assert.AreEqual("BPE", modelJson.GetProperty("type").GetString());
        Assert.IsTrue(modelJson.TryGetProperty("merges", out _));
        Assert.IsTrue(modelJson.TryGetProperty("vocab", out _));

        // Deserialize
        var restored = TokenizerLoader.FromJson(json);

        // Verify encoding produces same result
        var origIds = tokenizer.Encode("hello").GetIds();
        var restoredIds = restored.Encode("hello").GetIds();
        CollectionAssert.AreEqual(origIds, restoredIds);

        // Verify added tokens
        Assert.IsNotNull(restored.TokenToId("[CLS]"));
        Assert.IsNotNull(restored.TokenToId("[SEP]"));
    }

    // ────────────────────────────────────────────────────────────────────────────
    //  Edge cases
    // ────────────────────────────────────────────────────────────────────────────

    [TestMethod]
    public void Serialize_NoAddedTokens_NoAddedTokensInJson()
    {
        var vocab = new Dictionary<string, uint> { ["hello"] = 0, ["world"] = 1 };
        var model = new BpeModel.BpeBuilder().SetVocab(vocab).Build();
        var tokenizer = new Tokenizer(model);

        var json = TokenizerSerializer.Serialize(tokenizer);

        using var doc = JsonDocument.Parse(json);
        // When no added tokens, the "added_tokens" key should not be present
        Assert.IsFalse(doc.RootElement.TryGetProperty("added_tokens", out _));
    }

    [TestMethod]
    public void Serialize_NoMerges_NoMergesInJson()
    {
        var vocab = new Dictionary<string, uint> { ["hello"] = 0, ["world"] = 1 };
        var model = new BpeModel.BpeBuilder().SetVocab(vocab).Build();
        var tokenizer = new Tokenizer(model);

        var json = TokenizerSerializer.Serialize(tokenizer);

        using var doc = JsonDocument.Parse(json);
        var modelJson = doc.RootElement.GetProperty("model");
        Assert.IsFalse(modelJson.TryGetProperty("merges", out _));
    }

    [TestMethod]
    public void Serialize_NoPrefixOrSuffix_PropertiesOmitted()
    {
        var vocab = new Dictionary<string, uint> { ["hello"] = 0 };
        var model = new BpeModel.BpeBuilder().SetVocab(vocab).Build();
        var tokenizer = new Tokenizer(model);

        var json = TokenizerSerializer.Serialize(tokenizer);

        using var doc = JsonDocument.Parse(json);
        var modelJson = doc.RootElement.GetProperty("model");
        Assert.IsFalse(modelJson.TryGetProperty("continuing_subword_prefix", out _));
        Assert.IsFalse(modelJson.TryGetProperty("end_of_word_suffix", out _));
    }

    [TestMethod]
    public void Serialize_PrettyPrint_ProducesReadableJson()
    {
        var vocab = new Dictionary<string, uint> { ["<unk>"] = 0, ["hello"] = 1 };
        var model = new BpeModel.BpeBuilder().SetVocab(vocab).Build();

        var tokenizer = new Tokenizer(model);
        tokenizer.AddToken(new AddedToken("[CLS]", isSpecial: true));

        var json = TokenizerSerializer.Serialize(tokenizer, pretty: true);

        // Should contain newlines (pretty-printed)
        Assert.IsTrue(json.Contains('\n'));
        // Should still be valid JSON
        var doc = JsonDocument.Parse(json);
        Assert.IsNotNull(doc);
    }

    // ────────────────────────────────────────────────────────────────────────────
    //  BPE with continuing_subword_prefix (GPT-2 style)
    // ────────────────────────────────────────────────────────────────────────────

    [TestMethod]
    public void Serialize_BpeWithContinuingSubwordPrefix_RoundTrip()
    {
        // Use a vocab where the prefix is baked into the tokens that use it
        var json = """
        {
          "version": "1.0",
          "added_tokens": [],
          "normalizer": null,
          "pre_tokenizer": null,
          "post_processor": null,
          "decoder": null,
          "model": {
            "type": "BPE",
            "unk_token": "<unk>",
            "continuing_subword_prefix": "##",
            "vocab": {
              "<unk>": 0, "h": 1, "##e": 2, "##l": 3, "##o": 4,
              "hello": 5
            },
            "merges": ["h ##e", "h##e ##l", "h##e##l ##l", "h##e##l##l ##o"]
          }
        }
        """;

        var tokenizer = TokenizerLoader.FromJson(json);
        var serialized = TokenizerSerializer.Serialize(tokenizer);

        using var doc = JsonDocument.Parse(serialized);
        var modelJson = doc.RootElement.GetProperty("model");
        Assert.AreEqual("##", modelJson.GetProperty("continuing_subword_prefix").GetString());

        // Round-trip
        var restored = TokenizerLoader.FromJson(serialized);
        var origIds = tokenizer.Encode("hello").GetIds();
        var restoredIds = restored.Encode("hello").GetIds();
        CollectionAssert.AreEqual(origIds, restoredIds);
    }

    // ────────────────────────────────────────────────────────────────────────────
    //  Test fixtures
    // ────────────────────────────────────────────────────────────────────────────

    private const string SimpleBpeJson = """
    {
      "version": "1.0",
      "added_tokens": [
        {
          "id": 0,
          "content": "<unk>",
          "single_word": false,
          "lstrip": false,
          "rstrip": false,
          "normalized": false,
          "special": true
        }
      ],
      "normalizer": null,
      "pre_tokenizer": {
        "type": "Whitespace"
      },
      "post_processor": null,
      "decoder": {
        "type": "ByteLevel"
      },
      "model": {
        "type": "BPE",
        "unk_token": "<unk>",
        "vocab": {
          "<unk>": 0,
          "h": 1,
          "e": 2,
          "l": 3,
          "o": 4,
          "hello": 5
        },
        "merges": [
          ["h", "e"],
          ["he", "l"],
          ["hel", "l"],
          ["hell", "o"]
        ]
      }
    }
    """;

    private const string SimpleWordPieceJson = """
    {
      "version": "1.0",
      "added_tokens": [
        {
          "id": 0,
          "content": "[UNK]",
          "single_word": false,
          "lstrip": false,
          "rstrip": false,
          "normalized": false,
          "special": true
        },
        {
          "id": 101,
          "content": "[CLS]",
          "single_word": false,
          "lstrip": false,
          "rstrip": false,
          "normalized": false,
          "special": true
        },
        {
          "id": 102,
          "content": "[SEP]",
          "single_word": false,
          "lstrip": false,
          "rstrip": false,
          "normalized": false,
          "special": true
        }
      ],
      "normalizer": {
        "type": "BertNormalizer",
        "strip_accents": true,
        "lowercase": true,
        "handle_chinese_chars": true,
        "strip_control_chars": true,
        "normalize_whitespace": true
      },
      "pre_tokenizer": {
        "type": "BertPreTokenizer"
      },
      "post_processor": {
        "type": "BertProcessing",
        "sep": ["[SEP]", 102],
        "cls": ["[CLS]", 101]
      },
      "decoder": {
        "type": "WordPiece",
        "prefix": "##",
        "cleanup": true
      },
      "model": {
        "type": "WordPiece",
        "unk_token": "[UNK]",
        "continuing_subword_prefix": "##",
        "max_input_chars_per_word": 100,
        "vocab": {
          "[UNK]": 0,
          "[CLS]": 101,
          "[SEP]": 102,
          "[PAD]": 103,
          "hello": 1,
          "world": 2,
          "##s": 3
        }
      }
    }
    """;

    /// <summary>
    /// WordPiece config without BertProcessing post-processor (which has a pre-existing
    /// serialization gap for sep/cls arrays).
    /// </summary>
    private const string SimpleWordPieceNoPostProcessorJson = """
    {
      "version": "1.0",
      "added_tokens": [
        {
          "id": 0,
          "content": "[UNK]",
          "single_word": false,
          "lstrip": false,
          "rstrip": false,
          "normalized": false,
          "special": true
        }
      ],
      "normalizer": null,
      "pre_tokenizer": {
        "type": "BertPreTokenizer"
      },
      "post_processor": null,
      "decoder": {
        "type": "WordPiece",
        "prefix": "##",
        "cleanup": true
      },
      "model": {
        "type": "WordPiece",
        "unk_token": "[UNK]",
        "continuing_subword_prefix": "##",
        "max_input_chars_per_word": 100,
        "vocab": {
          "[UNK]": 0,
          "hello": 1,
          "world": 2,
          "##s": 3
        }
      }
    }
    """;

    // ────────────────────────────────────────────────────────────────────────────
    //  Enum serialization format — must match Rust serde output
    // ────────────────────────────────────────────────────────────────────────────

    [TestMethod]
    public void Serialize_PunctuationPreTokenizer_BehaviorIsPascalCase()
    {
        // Rust SplitDelimiterBehavior has no rename_all → PascalCase
        var tokenizer = new Tokenizer(new WordPieceModel.WordPieceBuilder()
            .SetVocab(new Dictionary<string, uint> { ["[UNK]"] = 0, ["hello"] = 1 })
            .Build());
        tokenizer.PreTokenizer = new PreTokenizers.PunctuationPreTokenizer(
            SplitDelimiterBehavior.Isolated);

        var json = TokenizerSerializer.Serialize(tokenizer);
        using var doc = JsonDocument.Parse(json);
        var behavior = doc.RootElement
            .GetProperty("pre_tokenizer")
            .GetProperty("behavior")
            .GetString();

        Assert.AreEqual("Isolated", behavior); // PascalCase, not "isolated"
    }

    [TestMethod]
    public void Serialize_SplitPreTokenizer_BehaviorIsPascalCase()
    {
        var tokenizer = new Tokenizer(new WordPieceModel.WordPieceBuilder()
            .SetVocab(new Dictionary<string, uint> { ["[UNK]"] = 0, ["hello"] = 1 })
            .Build());
        tokenizer.PreTokenizer = new PreTokenizers.SplitPreTokenizer(
            new StringPattern(" "), SplitDelimiterBehavior.MergedWithPrevious);

        var json = TokenizerSerializer.Serialize(tokenizer);
        using var doc = JsonDocument.Parse(json);
        var behavior = doc.RootElement
            .GetProperty("pre_tokenizer")
            .GetProperty("behavior")
            .GetString();

        Assert.AreEqual("MergedWithPrevious", behavior); // PascalCase
    }
}
