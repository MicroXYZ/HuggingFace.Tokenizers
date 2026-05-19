using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Decoders;
using HuggingFace.Tokenizers.Models.BPE;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Tests for BPE byte_fallback behavior.
/// When byte_fallback=true, unknown characters are encoded as &lt;0xXX&gt; hex tokens.
/// </summary>
    [TestClass]
public class ByteFallbackTests
{
    private static Tokenizer CreateByteFallbackTokenizer()
    {
        // Build vocab with byte tokens <0x00> through <0xFF> and some known words
        var vocab = new Dictionary<string, uint>();
        uint id = 0;

        // Add all 256 byte tokens
        for (int b = 0; b < 256; b++)
        {
            vocab[$"<0x{b:X2}>"] = id++;
        }

        // Add some regular tokens
        vocab["hello"] = id++;
        vocab["world"] = id++;

        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges([])
            .SetByteFallback(true)
            .Build();

        var tokenizer = new Tokenizer(model)
        {
            Decoder = new ByteFallbackDecoder()
        };

        return tokenizer;
    }

    [TestMethod]
    public void ByteFallback_UnknownChars_EncodedAsHexTokens()
    {
        var tokenizer = CreateByteFallbackTokenizer();

        // "你好" in UTF-8 is 6 bytes: E4 BD A0 E5 A5 BD
        var encoding = tokenizer.Encode("你好");

        var tokens = encoding.GetTokens();
        // Each byte should be a <0xXX> token
        foreach (var t in tokens) StringAssert.Matches(t, new System.Text.RegularExpressions.Regex(@"^<0x[0-9A-Fa-f]{2}>$"));
        Assert.AreEqual(6, tokens.Length);
    }

    [TestMethod]
    public void ByteFallback_ByteTokensInVocab()
    {
        var tokenizer = CreateByteFallbackTokenizer();

        // Verify byte tokens exist in vocab
        Assert.IsNotNull(tokenizer.TokenToId("<0x00>"));
        Assert.IsNotNull(tokenizer.TokenToId("<0xFF>"));
        Assert.IsNotNull(tokenizer.TokenToId("<0x41>")); // 'A'
    }

    [TestMethod]
    public void ByteFallback_DecodeRoundTrip_BytesRestored()
    {
        var tokenizer = CreateByteFallbackTokenizer();

        // Encode Chinese characters → byte tokens
        var encoding = tokenizer.Encode("你好");
        var ids = encoding.GetIds();

        // Decode back → should restore original bytes → UTF-8 → "你好"
        var decoded = tokenizer.Decode(ids);

        Assert.AreEqual("你好", decoded);
    }

    [TestMethod]
    public void ByteFallback_AsciiChars_EncodedCorrectly()
    {
        var tokenizer = CreateByteFallbackTokenizer();

        // ASCII 'A' = 0x41
        var encoding = tokenizer.Encode("A");

        var tokens = encoding.GetTokens();
        Assert.AreEqual(1, tokens.Count());
        Assert.AreEqual("<0x41>", tokens[0]);
    }

    [TestMethod]
    public void ByteFallback_Decoder_ChainProcessByteTokens()
    {
        var decoder = new ByteFallbackDecoder();

        // "你" in UTF-8 = E4 BD A0
        var tokens = new[] { "<0xE4>", "<0xBD>", "<0xA0>" };
        var chain = decoder.DecodeChain(tokens);

        Assert.AreEqual(1, chain.Count());
        Assert.AreEqual("你", chain[0]);
    }

    [TestMethod]
    public void ByteFallback_Decoder_MixedTokensAndBytes()
    {
        var decoder = new ByteFallbackDecoder();

        // "A你" = 0x41 + E4 BD A0 — all are byte tokens, so buffered and flushed together
        var tokens = new[] { "<0x41>", "<0xE4>", "<0xBD>", "<0xA0>" };
        var chain = decoder.DecodeChain(tokens);

        // All byte tokens get buffered into one UTF-8 decode → single result "A你"
        Assert.AreEqual(1, chain.Count());
        Assert.AreEqual("A你", chain[0]);
    }

    [TestMethod]
    public void ByteFallback_Decoder_NonByteTokensPassedThrough()
    {
        var decoder = new ByteFallbackDecoder();

        var tokens = new[] { "hello", "<0x20>", "world" };
        var result = decoder.Decode(tokens);

        // "hello" + space (0x20) + "world"
        Assert.AreEqual("hello world", result);
    }

    [TestMethod]
    public void ByteFallback_MixedAsciiAndUnicode_RoundTrip()
    {
        var tokenizer = CreateByteFallbackTokenizer();

        var text = "Hi你好";
        var encoding = tokenizer.Encode(text);
        var decoded = tokenizer.Decode(encoding.GetIds());

        Assert.AreEqual(text, decoded);
    }
}
