using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Serialization;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Tests for tokenizer.json deserialization (TokenizerLoader).
/// </summary>
    [TestClass]
public class TokenizerLoaderTests
{
    // ────────────────────────────────────────────────────────────────────────────
    //  BPE model (GPT-2 style)
    // ────────────────────────────────────────────────────────────────────────────

    private const string BpeTokenizerJson = """
    {
      "version": "1.0",
      "truncation": null,
      "padding": null,
      "added_tokens": [
        {
          "id": 0,
          "content": "<unk>",
          "single_word": false,
          "lstrip": false,
          "rstrip": false,
          "normalized": false,
          "special": true
        },
        {
          "id": 6,
          "content": " world",
          "single_word": false,
          "lstrip": false,
          "rstrip": false,
          "normalized": false,
          "special": false
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
        "dropout": null,
        "unk_token": "<unk>",
        "continuing_subword_prefix": null,
        "end_of_word_suffix": null,
        "fuse_unk": false,
        "byte_fallback": false,
        "ignore_merges": false,
        "vocab": {
          "<unk>": 0,
          "h": 1,
          "e": 2,
          "l": 3,
          "o": 4,
          "hello": 5,
          " world": 6
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

    [TestMethod]
    public void FromJson_BpeModel_LoadsSuccessfully()
    {
        var tokenizer = TokenizerLoader.FromJson(BpeTokenizerJson);

        Assert.IsNotNull(tokenizer);
        Assert.IsNotNull(tokenizer.Model);
        Assert.IsNotNull(tokenizer.PreTokenizer);
        Assert.IsNotNull(tokenizer.Decoder);
    }

    [TestMethod]
    public void FromJson_BpeModel_EncodesText()
    {
        var tokenizer = TokenizerLoader.FromJson(BpeTokenizerJson);

        var encoding = tokenizer.Encode("hello world");

        Assert.IsNotNull(encoding);
        Assert.IsTrue(encoding.Length > 0);
    }

    [TestMethod]
    public void FromJson_BpeModel_AddedTokensLoaded()
    {
        var tokenizer = TokenizerLoader.FromJson(BpeTokenizerJson);

        // <unk> (id=0) should be in added vocabulary
        var unkResult = tokenizer.Decode([0]);
        Assert.IsNotNull(unkResult);
    }

    // ────────────────────────────────────────────────────────────────────────────
    //  WordPiece model (BERT style)
    // ────────────────────────────────────────────────────────────────────────────

    private const string WordPieceTokenizerJson = """
    {
      "version": "1.0",
      "truncation": {
        "type": "LongestFirst",
        "max_length": 512,
        "stride": 0,
        "direction": "Right"
      },
      "padding": {
        "type": "BatchLongest",
        "pad_id": 0,
        "pad_type_id": 0,
        "pad_token": "[PAD]",
        "direction": "Right"
      },
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

    [TestMethod]
    public void FromJson_WordPieceModel_LoadsSuccessfully()
    {
        var tokenizer = TokenizerLoader.FromJson(WordPieceTokenizerJson);

        Assert.IsNotNull(tokenizer);
        Assert.IsNotNull(tokenizer.Model);
        Assert.IsNotNull(tokenizer.Normalizer);
        Assert.IsNotNull(tokenizer.PreTokenizer);
        Assert.IsNotNull(tokenizer.PostProcessor);
        Assert.IsNotNull(tokenizer.Decoder);
        Assert.IsNotNull(tokenizer.Truncation);
        Assert.IsNotNull(tokenizer.Padding);
    }

    [TestMethod]
    public void FromJson_WordPieceModel_TruncationConfigLoaded()
    {
        var tokenizer = TokenizerLoader.FromJson(WordPieceTokenizerJson);

        Assert.IsNotNull(tokenizer.Truncation);
        Assert.AreEqual(512, tokenizer.Truncation!.MaxLength);
        Assert.AreEqual(TruncationStrategy.LongestFirst, tokenizer.Truncation.Strategy);
    }

    [TestMethod]
    public void FromJson_WordPieceModel_PaddingConfigLoaded()
    {
        var tokenizer = TokenizerLoader.FromJson(WordPieceTokenizerJson);

        Assert.IsNotNull(tokenizer.Padding);
        Assert.AreEqual("[PAD]", tokenizer.Padding!.PadToken);
        Assert.AreEqual(0u, tokenizer.Padding.PadId);
    }

    [TestMethod]
    public void FromJson_WordPieceModel_EncodesText()
    {
        var tokenizer = TokenizerLoader.FromJson(WordPieceTokenizerJson);

        var encoding = tokenizer.Encode("hello worlds");

        Assert.IsNotNull(encoding);
        Assert.IsTrue(encoding.Length > 0);
    }

    [TestMethod]
    public void FromJson_WordPieceModel_PostProcessorAddsSpecialTokens()
    {
        var tokenizer = TokenizerLoader.FromJson(WordPieceTokenizerJson);

        var encoding = tokenizer.Encode("hello world");

        // Should have [CLS] at start and [SEP] at end
        var tokens = encoding.GetTokens();
        Assert.IsTrue(tokens.Length >= 3);
        Assert.AreEqual("[CLS]", tokens[0]);
        Assert.AreEqual("[SEP]", tokens[^1]);
    }

    // ────────────────────────────────────────────────────────────────────────────
    //  Edge cases
    // ────────────────────────────────────────────────────────────────────────────

    [TestMethod]
    public void FromJson_NullInput_ThrowsArgumentException()
    {
        Assert.ThrowsExactly<ArgumentException>(() => TokenizerLoader.FromJson(null!));
    }

    [TestMethod]
    public void FromJson_EmptyInput_ThrowsArgumentException()
    {
        Assert.ThrowsExactly<ArgumentException>(() => TokenizerLoader.FromJson(""));
    }

    [TestMethod]
    public void FromJson_MissingModel_ThrowsInvalidOperation()
    {
        var json = """{"version": "1.0", "model": null}""";
        Assert.ThrowsExactly<InvalidOperationException>(() => TokenizerLoader.FromJson(json));
    }

    [TestMethod]
    public void FromFile_NonExistentPath_ThrowsFileNotFound()
    {
        Assert.ThrowsExactly<FileNotFoundException>(() =>
            TokenizerLoader.FromFile("/nonexistent/tokenizer.json"));
    }

    [TestMethod]
    public void FromJson_SequenceNormalizer_RecursivelyResolves()
    {
        var json = """
        {
          "version": "1.0",
          "normalizer": {
            "type": "Sequence",
            "normalizers": [
              {"type": "NFC"},
              {"type": "Lowercase"}
            ]
          },
          "pre_tokenizer": null,
          "post_processor": null,
          "decoder": null,
          "model": {
            "type": "WordLevel",
            "vocab": {"<unk>": 0, "hello": 1},
            "unk_token": "<unk>"
          }
        }
        """;

        var tokenizer = TokenizerLoader.FromJson(json);
        Assert.IsNotNull(tokenizer.Normalizer);
    }

    [TestMethod]
    public void FromJson_SequenceDecoder_RecursivelyResolves()
    {
        var json = """
        {
          "version": "1.0",
          "normalizer": null,
          "pre_tokenizer": null,
          "post_processor": null,
          "decoder": {
            "type": "Sequence",
            "decoders": [
              {"type": "ByteFallback"},
              {"type": "Metaspace", "replacement": "▁", "add_prefix_space": true, "prepend_scheme": "always"}
            ]
          },
          "model": {
            "type": "WordLevel",
            "vocab": {"<unk>": 0, "hello": 1},
            "unk_token": "<unk>"
          }
        }
        """;

        var tokenizer = TokenizerLoader.FromJson(json);
        Assert.IsNotNull(tokenizer.Decoder);
    }

    // ────────────────────────────────────────────────────────────────────────────
    //  Round-trip: load from JSON → encode → decode
    // ────────────────────────────────────────────────────────────────────────────

    [TestMethod]
    public void FromJson_RoundTrip_EncodeDecode()
    {
        var tokenizer = TokenizerLoader.FromJson(BpeTokenizerJson);

        var text = "hello";
        var encoding = tokenizer.Encode(text);
        var decoded = tokenizer.Decode(encoding.GetIds());

        Assert.IsNotNull(decoded);
        // The decoded text should contain recognizable content
        Assert.IsTrue(decoded.Count() > 0);
    }
}
