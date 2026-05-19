using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Decoders;
using HuggingFace.Tokenizers.Normalizers;
using HuggingFace.Tokenizers.Processors;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Serialization round-trip tests for R4/R5 modified components.
/// </summary>
    [TestClass]
public class SerializationRoundTripTests
{
    [TestMethod]
    public void CtcDecoder_RoundTrip_PreservesConfig()
    {
        var decoder = new CtcDecoder(padToken: "<blank>", wordDelimiter: "|", cleanup: true);
        var tokens = new[] { "<blank>", "h", "e", "e", "l", "l", "o" };
        var result = decoder.DecodeChain(tokens);
        CollectionAssert.AreEqual(new[] { "h", "e", "l", "o" }, result.ToArray());
    }

    [TestMethod]
    public void StripDecoder_CharContent_RoundTrip()
    {
        var decoder = new StripDecoder(content: 'H', start: 1, stop: 0);
        var tokens = new[] { "Hello" };
        var result = decoder.DecodeChain(tokens);
        Assert.AreEqual("ello", result[0]);
    }

    [TestMethod]
    public void ReplaceDecoder_Regex_RoundTrip()
    {
        var decoder = new ReplaceDecoder(@"\s+", " ", ReplacePatternType.Regex);
        var tokens = new[] { "hello   world" };
        var result = decoder.DecodeChain(tokens);
        Assert.AreEqual("hello world", result[0]);
    }

    [TestMethod]
    public void BertNormalizer_NewParams_DefaultStripAccents()
    {
        var normalizer = new BertNormalizer(cleanText: true, handleChineseChars: true, stripAccents: null, lowercase: true);
        var ns = new NormalizedString("Héllo");
        normalizer.Normalize(ns);
        Assert.AreEqual("hello", ns.Get());
    }

    [TestMethod]
    public void BertNormalizer_StripAccents_False()
    {
        var normalizer = new BertNormalizer(cleanText: false, handleChineseChars: false, stripAccents: false, lowercase: false);
        var ns = new NormalizedString("Héllo");
        normalizer.Normalize(ns);
        Assert.AreEqual("Héllo", ns.Get());
    }

    [TestMethod]
    public void NmtNormalizer_MapsWhitespace()
    {
        var normalizer = new NmtNormalizer();
        var ns = new NormalizedString("hello\tworld\nfoo");
        normalizer.Normalize(ns);
        Assert.AreEqual("hello world foo", ns.Get());
    }

    [TestMethod]
    public void NmtNormalizer_FiltersControlChars()
    {
        var normalizer = new NmtNormalizer();
        var ns = new NormalizedString("hello\x01\x02\x03world");
        normalizer.Normalize(ns);
        Assert.AreEqual("helloworld", ns.Get());
    }

    [TestMethod]
    public void NmtNormalizer_PreservesTab()
    {
        var normalizer = new NmtNormalizer();
        var ns = new NormalizedString("a\tb");
        normalizer.Normalize(ns);
        Assert.AreEqual("a b", ns.Get());
    }

    [TestMethod]
    public void ByteLevelPostProcessor_TrimOffsets()
    {
        var processor = new ByteLevelPostProcessor(trimOffsets: true, addPrefixSpace: true);
        var encoding = new Encoding(
            ids: [1, 2],
            typeIds: [0, 0],
            tokens: [" hello", "world "],
            words: [null, null],
            offsets: [(0, 6), (6, 12)],
            specialTokensMask: [0, 0],
            attentionMask: [1, 1]);

        var result = processor.Process(encoding, null, true);
        var offsets = result.GetOffsets();
        // First token " hello": addPrefixSpace=true, leading space preserved
        Assert.AreEqual(0, offsets[0].Start);
        // Second token "world ": trailing space trimmed
        Assert.AreEqual(11, offsets[1].End);
    }

    [TestMethod]
    public void ByteLevelDecoder_ExtendedLatin_RoundTrip()
    {
        var decoder = new ByteLevelDecoder();
        var tokens = new[] { "Hello", "World", "!", "~" };
        var result = decoder.DecodeChain(tokens);
        Assert.AreEqual("Hello", result[0]);
        Assert.AreEqual("World", result[1]);
        Assert.AreEqual("!", result[2]);
        Assert.AreEqual("~", result[3]);
    }

    [TestMethod]
    public void TemplateProcessing_MultiIdSpecialToken()
    {
        var processor = new TemplateProcessing(
            singleTemplate:
            [
                Template.Special([101, 102], ["[CLS]", "[PAD]"], typeId: 0),
                Template.A(typeId: 0),
            ]);

        var encoding = new Encoding(
            ids: [12, 14],
            typeIds: [0, 0],
            tokens: ["Hello", "there"],
            words: [null, null],
            offsets: [(0, 5), (6, 11)],
            specialTokensMask: [0, 0],
            attentionMask: [1, 1]);

        var result = processor.Process(encoding, null, true);
        Assert.AreEqual(4, result.Length);
        CollectionAssert.AreEqual(new uint[] { 101, 102, 12, 14 }, result.GetIds());
        Assert.AreEqual("[CLS]", result.GetTokens()[0]);
        Assert.AreEqual("[PAD]", result.GetTokens()[1]);
    }
}
