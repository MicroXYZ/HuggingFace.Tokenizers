using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Normalizers;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Tests for normalizer implementations.
/// </summary>
    [TestClass]
public class NormalizerTests
{
    [TestMethod]
    public void NfcNormalizer_NormalizesUnicode()
    {
        var normalizer = new NfcNormalizer();
        var ns = new NormalizedString("café");  // é as decomposed
        normalizer.Normalize(ns);
        Assert.AreEqual("café", ns.Get());
    }

    [TestMethod]
    public void NfdNormalizer_DecomposesUnicode()
    {
        var normalizer = new NfdNormalizer();
        var ns = new NormalizedString("café");  // é as single char
        normalizer.Normalize(ns);
        // NFD decomposes é into e + combining accent
        Assert.IsTrue(ns.Get().Length >= "café".Length);
    }

    [TestMethod]
    public void LowercaseNormalizer_ConvertsToLower()
    {
        var normalizer = new LowercaseNormalizer();
        var ns = new NormalizedString("Hello WORLD");
        normalizer.Normalize(ns);
        Assert.AreEqual("hello world", ns.Get());
    }

    [TestMethod]
    public void BertNormalizer_DefaultSettings_HandlesChineseAndAccents()
    {
        var normalizer = new BertNormalizer();
        var ns = new NormalizedString("Héllo 你好World");
        normalizer.Normalize(ns);
        // Should lowercase, strip accents, handle Chinese chars
        var result = ns.Get();
        StringAssert.Contains(result, "你");
        CollectionAssert.DoesNotContain(result.ToList(), "H"); // lowercased
    }

    [TestMethod]
    public void StripNormalizer_RemovesWhitespace()
    {
        var normalizer = new StripNormalizer();
        var ns = new NormalizedString("  hello  ");
        normalizer.Normalize(ns);
        Assert.AreEqual("hello", ns.Get());
    }

    [TestMethod]
    public void SequenceNormalizer_ChainsMultipleNormalizers()
    {
        var normalizer = new SequenceNormalizer([
            new LowercaseNormalizer(),
            new StripNormalizer()
        ]);
        var ns = new NormalizedString("  HELLO  ");
        normalizer.Normalize(ns);
        Assert.AreEqual("hello", ns.Get());
    }

    [TestMethod]
    public void PrependNormalizer_AddsPrefix()
    {
        var normalizer = new PrependNormalizer("▁");
        var ns = new NormalizedString("hello");
        normalizer.Normalize(ns);
        StringAssert.StartsWith(ns.Get(), "▁");
    }

    [TestMethod]
    public void BertNormalizer_CJK_SpacesAroundIdeographs()
    {
        var normalizer = new BertNormalizer(
            handleChineseChars: true,
            lowercase: false,
            stripAccents: false,
            cleanText: false);

        var ns = new NormalizedString("你好");
        normalizer.Normalize(ns);

        // Each CJK char gets space before AND after: " 你 " + " 好 " = " 你  好 "
        // Adjacent CJK chars produce double spaces between them
        Assert.AreEqual(" 你  好 ", ns.Get());
    }

    [TestMethod]
    public void BertNormalizer_CJK_MixedWithLatin()
    {
        var normalizer = new BertNormalizer(
            handleChineseChars: true,
            lowercase: false,
            stripAccents: false,
            cleanText: false);

        var ns = new NormalizedString("hello你好world");
        normalizer.Normalize(ns);

        // Each CJK char gets space before AND after; adjacent CJK chars → double space
        Assert.AreEqual("hello 你  好 world", ns.Get());
    }

    [TestMethod]
    public void BertNormalizer_CJK_PunctuationNotTreatedAsIdeograph()
    {
        var normalizer = new BertNormalizer(
            handleChineseChars: true,
            lowercase: false,
            stripAccents: false,
            cleanText: false);

        // Chinese punctuation: 。(U+3002) 、(U+3001) ！(U+FF01)
        var ns = new NormalizedString("你好！世界。");
        normalizer.Normalize(ns);

        // CJK ideographs get spaces, but punctuation does not
        var result = ns.Get();
        StringAssert.Contains(result, " 你 ");
        StringAssert.Contains(result, " 好 ");
        StringAssert.Contains(result, " 世 ");
        StringAssert.Contains(result, " 界 ");
        // Punctuation should NOT have spaces around it
        StringAssert.Contains(result, "！");
        StringAssert.Contains(result, "。");
    }

    [TestMethod]
    public void BertNormalizer_CJK_MultipleCharacters_AllSpaced()
    {
        var normalizer = new BertNormalizer(
            handleChineseChars: true,
            lowercase: false,
            stripAccents: false,
            cleanText: false);

        var ns = new NormalizedString("春夏秋冬");
        normalizer.Normalize(ns);

        // Each CJK char gets space before AND after; adjacent CJK chars → double space
        Assert.AreEqual(" 春  夏  秋  冬 ", ns.Get());
    }

    [TestMethod]
    public void BertNormalizer_CJK_WithLowercase()
    {
        var normalizer = new BertNormalizer(
            handleChineseChars: true,
            lowercase: true,
            stripAccents: false,
            cleanText: false);

        var ns = new NormalizedString("Hello你好WORLD");
        normalizer.Normalize(ns);

        // CJK chars get double spaces between them; lowercase applied to Latin
        Assert.AreEqual("hello 你  好 world", ns.Get());
    }
}
