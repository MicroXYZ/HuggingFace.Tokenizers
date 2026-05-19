using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Decoders;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Tests for MetaspaceDecoder correctness.
/// Updated to match Rust Metaspace: uses prepend_scheme only (no separate addPrefixSpace).
/// </summary>
    [TestClass]
public class MetaspaceDecoderTests
{
    [TestMethod]
    public void DecodeChain_Always_FirstTokenKeepsSpace_SubsequentTokensGetSpace()
    {
        var decoder = new MetaspaceDecoder(
            replacement: '\u2581',
            prependScheme: PrependScheme.Always);

        var tokens = new[] { "\u2581Hey", "\u2581friend!" };
        var chain = decoder.DecodeChain(tokens);

        Assert.AreEqual(2, chain.Count);
        // Always mode: ▁ replaced with space; since decoded already starts with space, it's kept
        Assert.AreEqual(" Hey", chain[0]);
        Assert.AreEqual(" friend!", chain[1]);
    }

    [TestMethod]
    public void DecodeChain_Never_LeadingSpaceStrippedFromFirstToken()
    {
        var decoder = new MetaspaceDecoder(
            replacement: '\u2581',
            prependScheme: PrependScheme.Never);

        var tokens = new[] { "\u2581Hey", "\u2581friend!" };
        var chain = decoder.DecodeChain(tokens);

        Assert.AreEqual(2, chain.Count);
        // Never mode: leading space trimmed from first token
        Assert.AreEqual("Hey", chain[0]);
        Assert.AreEqual(" friend!", chain[1]);
    }

    [TestMethod]
    public void Decode_Always_JoinedCorrectly()
    {
        var decoder = new MetaspaceDecoder(
            replacement: '\u2581',
            prependScheme: PrependScheme.Always);

        var tokens = new[] { "\u2581Hey", "\u2581friend!" };
        var result = decoder.Decode(tokens);

        Assert.AreEqual(" Hey friend!", result);
    }

    [TestMethod]
    public void Decode_Never_JoinedCorrectly()
    {
        var decoder = new MetaspaceDecoder(
            replacement: '\u2581',
            prependScheme: PrependScheme.Never);

        var tokens = new[] { "\u2581Hey", "\u2581friend!" };
        var result = decoder.Decode(tokens);

        Assert.AreEqual("Hey friend!", result);
    }

    [TestMethod]
    public void DecodeChain_First_TokenAlreadyStartsWithReplacement()
    {
        var decoder = new MetaspaceDecoder(
            replacement: '\u2581',
            prependScheme: PrependScheme.First);

        var tokens = new[] { "\u2581Hey", "\u2581friend!" };
        var chain = decoder.DecodeChain(tokens);

        // First scheme: only add prefix space if token doesn't already start with replacement
        // "\u2581Hey" starts with replacement, so no extra space added
        Assert.AreEqual(" Hey", chain[0]);
        Assert.AreEqual(" friend!", chain[1]);
    }

    [TestMethod]
    public void DecodeChain_First_TokenDoesNotStartWithReplacement()
    {
        var decoder = new MetaspaceDecoder(
            replacement: '\u2581',
            prependScheme: PrependScheme.First);

        var tokens = new[] { "Hey", "\u2581friend!" };
        var chain = decoder.DecodeChain(tokens);

        // "Hey" doesn't start with ▁ and doesn't start with space → prepend space
        Assert.AreEqual(" Hey", chain[0]);
        Assert.AreEqual(" friend!", chain[1]);
    }

    [TestMethod]
    public void DecodeChain_SingleToken_Always()
    {
        var decoder = new MetaspaceDecoder(
            replacement: '\u2581',
            prependScheme: PrependScheme.Always);

        var tokens = new[] { "\u2581Hello" };
        var chain = decoder.DecodeChain(tokens);

        Assert.AreEqual(1, chain.Count());
        Assert.AreEqual(" Hello", chain[0]);
    }

    [TestMethod]
    public void DecodeChain_SingleToken_Never()
    {
        var decoder = new MetaspaceDecoder(
            replacement: '\u2581',
            prependScheme: PrependScheme.Never);

        var tokens = new[] { "\u2581Hello" };
        var chain = decoder.DecodeChain(tokens);

        Assert.AreEqual(1, chain.Count());
        Assert.AreEqual("Hello", chain[0]);
    }

    [TestMethod]
    public void DecodeChain_NoReplacementChar_TokensUnchanged()
    {
        var decoder = new MetaspaceDecoder(
            replacement: '\u2581',
            prependScheme: PrependScheme.Always);

        var tokens = new[] { "hello", "world" };
        var chain = decoder.DecodeChain(tokens);

        // No ▁ to replace; Always mode checks if decoded starts with ' '
        // "hello" doesn't start with ' ', so prepends ' '
        Assert.AreEqual(" hello", chain[0]);
        Assert.AreEqual("world", chain[1]);
    }

    [TestMethod]
    public void DecodeChain_EmptyInput_ReturnsEmpty()
    {
        var decoder = new MetaspaceDecoder();

        var chain = decoder.DecodeChain([]);

        Assert.AreEqual(0, chain.Count());
    }
}
