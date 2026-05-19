using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Decoders;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Tests for BpeDecoder correctness.
/// </summary>
    [TestClass]
public class BpeDecoderTests
{
    [TestMethod]
    public void Decode_SuffixEndOfWord_TokensSeparatedBySpace()
    {
        var decoder = new BpeDecoder(suffix: "</w>");
        var tokens = new[] { "hello</w>", "world</w>" };

        var result = decoder.Decode(tokens);

        Assert.AreEqual("hello world", result);
    }

    [TestMethod]
    public void Decode_SuffixEndOfWord_SingleToken_NoTrailingSpace()
    {
        var decoder = new BpeDecoder(suffix: "</w>");
        var tokens = new[] { "hello</w>" };

        var result = decoder.Decode(tokens);

        Assert.AreEqual("hello", result);
    }

    [TestMethod]
    public void Decode_EmptySuffix_TokensConcatenated()
    {
        var decoder = new BpeDecoder(suffix: "");
        var tokens = new[] { "a", "b" };

        var result = decoder.Decode(tokens);

        Assert.AreEqual("ab", result);
    }

    [TestMethod]
    public void Decode_EmptySuffix_MultipleTokens_AllJoined()
    {
        var decoder = new BpeDecoder(suffix: "");
        var tokens = new[] { "hel", "lo", " ", "wor", "ld" };

        var result = decoder.Decode(tokens);

        Assert.AreEqual("hello world", result);
    }

    [TestMethod]
    public void DecodeChain_SuffixEndOfWord_LastTokenSuffixRemoved_OthersReplacedWithSpace()
    {
        var decoder = new BpeDecoder(suffix: "</w>");
        var tokens = new[] { "hello</w>", "world</w>" };

        var chain = decoder.DecodeChain(tokens);

        Assert.AreEqual(2, chain.Count);
        Assert.AreEqual("hello ", chain[0]);  // non-last: suffix → " " (trailing space)
        Assert.AreEqual("world", chain[1]);   // last: suffix → "" (removed)
    }

    [TestMethod]
    public void DecodeChain_SuffixEndOfWord_MiddleTokensGetSpace()
    {
        var decoder = new BpeDecoder(suffix: "</w>");
        var tokens = new[] { "a</w>", "b</w>", "c</w>" };

        var chain = decoder.DecodeChain(tokens);

        Assert.AreEqual(3, chain.Count);
        Assert.AreEqual("a ", chain[0]);  // non-last: suffix → " "
        Assert.AreEqual("b ", chain[1]);  // non-last: suffix → " "
        Assert.AreEqual("c", chain[2]);   // last: suffix → ""
    }

    [TestMethod]
    public void DecodeChain_EmptySuffix_ReturnsTokensUnchanged()
    {
        var decoder = new BpeDecoder(suffix: "");
        var tokens = new[] { "x", "y", "z" };

        var chain = decoder.DecodeChain(tokens);

        CollectionAssert.AreEqual(new string[] { "x", "y", "z" }, chain.ToArray());
    }

    [TestMethod]
    public void DecodeChain_EmptyInput_ReturnsEmpty()
    {
        var decoder = new BpeDecoder(suffix: "</w>");

        var chain = decoder.DecodeChain([]);

        Assert.AreEqual(0, chain.Count());
    }

    [TestMethod]
    public void Decode_ThreeTokens_SuffixEndOfWord_SpacesInsertedCorrectly()
    {
        var decoder = new BpeDecoder(suffix: "</w>");
        var tokens = new[] { "i</w>", "love</w>", "cats</w>" };

        var result = decoder.Decode(tokens);

        Assert.AreEqual("i love cats", result);
    }

    [TestMethod]
    public void Decode_CustomSuffix_ReplacedCorrectly()
    {
        var decoder = new BpeDecoder(suffix: "##");
        var tokens = new[] { "un", "##break", "##able" };

        var result = decoder.Decode(tokens);

        // "un" has no "##" → unchanged; "##break" → " break"; last "##able" → "able"
        Assert.AreEqual("un breakable", result);
    }

    [TestMethod]
    public void Decode_TokensWithoutSuffix_Unchanged()
    {
        var decoder = new BpeDecoder(suffix: "</w>");
        var tokens = new[] { "no", "suffix", "here" };

        var result = decoder.Decode(tokens);

        // No token contains </w>, so no replacements happen
        Assert.AreEqual("nosuffixhere", result);
    }
}
