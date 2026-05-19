using HuggingFace.Tokenizers.Internal;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.PreTokenizers;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Tests for all pre-tokenizer implementations.
/// </summary>
    [TestClass]
public class PreTokenizerTests
{
    private static List<string> GetSplitTexts(string input, IPreTokenizer preTokenizer)
    {
        var pts = new PreTokenizedString(input);
        preTokenizer.PreTokenize(pts);
        return pts.GetSplits().Select(s => s.Normalized.Get()).ToList();
    }

    // ========== BertPreTokenizer ==========

    [TestMethod]
    public void BertPreTokenizer_SplitsWhitespaceAndPunctuation()
    {
        var splits = GetSplitTexts("Hello, world!", new BertPreTokenizer());
        CollectionAssert.AreEqual(new[] { "Hello", ",", "world", "!" }, splits);
    }

    [TestMethod]
    public void BertPreTokenizer_HandlesMultipleSpaces()
    {
        var splits = GetSplitTexts("Hello   world", new BertPreTokenizer());
        CollectionAssert.AreEqual(new[] { "Hello", "world" }, splits);
    }

    [TestMethod]
    public void BertPreTokenizer_HandlesEmptyString()
    {
        var splits = GetSplitTexts("", new BertPreTokenizer());
        Assert.AreEqual(0, splits.Count());
    }

    // ========== WhitespacePreTokenizer ==========

    [TestMethod]
    public void WhitespacePreTokenizer_SplitsLikeBert()
    {
        var splits = GetSplitTexts("Hello, world!", new WhitespacePreTokenizer());
        CollectionAssert.AreEqual(new[] { "Hello", ",", "world", "!" }, splits);
    }

    // ========== WhitespaceSplitPreTokenizer ==========

    [TestMethod]
    public void WhitespaceSplitPreTokenizer_DoesNotSplitPunctuation()
    {
        var splits = GetSplitTexts("Hello, world!", new WhitespaceSplitPreTokenizer());
        CollectionAssert.AreEqual(new[] { "Hello,", "world!" }, splits);
    }

    [TestMethod]
    public void WhitespaceSplitPreTokenizer_SplitsOnWhitespace()
    {
        var splits = GetSplitTexts("a b c", new WhitespaceSplitPreTokenizer());
        CollectionAssert.AreEqual(new[] { "a", "b", "c" }, splits);
    }

    // ========== DigitsPreTokenizer ==========

    [TestMethod]
    public void DigitsPreTokenizer_GroupedDigits()
    {
        var splits = GetSplitTexts("abc123def", new DigitsPreTokenizer(individualDigits: false));
        CollectionAssert.AreEqual(new[] { "abc", "123", "def" }, splits);
    }

    [TestMethod]
    public void DigitsPreTokenizer_IndividualDigits()
    {
        var splits = GetSplitTexts("abc123def", new DigitsPreTokenizer(individualDigits: true));
        CollectionAssert.AreEqual(new[] { "abc", "1", "2", "3", "def" }, splits);
    }

    [TestMethod]
    public void DigitsPreTokenizer_NoDigits()
    {
        var splits = GetSplitTexts("hello", new DigitsPreTokenizer());
        CollectionAssert.AreEqual(new[] { "hello" }, splits);
    }

    // ========== PunctuationPreTokenizer ==========

    [TestMethod]
    public void PunctuationPreTokenizer_Isolated()
    {
        var splits = GetSplitTexts("Hello, world!", new PunctuationPreTokenizer(SplitDelimiterBehavior.Isolated));
        // Punctuation is isolated, but surrounding whitespace stays with adjacent segments
        CollectionAssert.AreEqual(new[] { "Hello", ",", " world", "!" }, splits);
    }

    [TestMethod]
    public void PunctuationPreTokenizer_Removed()
    {
        var splits = GetSplitTexts("Hello, world!", new PunctuationPreTokenizer(SplitDelimiterBehavior.Removed));
        CollectionAssert.AreEqual(new[] { "Hello", " world" }, splits);
    }

    // ========== DelimiterSplitPreTokenizer ==========

    [TestMethod]
    public void DelimiterSplitPreTokenizer_SplitsOnChar()
    {
        var splits = GetSplitTexts("a|b|c", new DelimiterSplitPreTokenizer('|'));
        CollectionAssert.AreEqual(new[] { "a", "b", "c" }, splits);
    }

    [TestMethod]
    public void DelimiterSplitPreTokenizer_NoDelimiter()
    {
        var splits = GetSplitTexts("abc", new DelimiterSplitPreTokenizer('|'));
        CollectionAssert.AreEqual(new[] { "abc" }, splits);
    }

    // ========== MetaspacePreTokenizer ==========

    [TestMethod]
    public void MetaspacePreTokenizer_DefaultBehavior()
    {
        var splits = GetSplitTexts("Hello World", new MetaspacePreTokenizer());
        CollectionAssert.AreEqual(new[] { "▁Hello", "▁World" }, splits);
    }

    [TestMethod]
    public void MetaspacePreTokenizer_LeadingSpace()
    {
        var splits = GetSplitTexts(" Hello", new MetaspacePreTokenizer());
        CollectionAssert.AreEqual(new[] { "▁Hello" }, splits);
    }

    [TestMethod]
    public void MetaspacePreTokenizer_NoPrefix()
    {
        var metaspace = new MetaspacePreTokenizer(addPrefixSpace: false, prependScheme: PrependScheme.Never);
        var splits = GetSplitTexts("Hello World", metaspace);
        // Replacement char becomes prefix of the next word, not suffix of the previous
        CollectionAssert.AreEqual(new[] { "Hello", "▁World" }, splits);
    }

    [TestMethod]
    public void MetaspacePreTokenizer_CustomReplacement()
    {
        var metaspace = new MetaspacePreTokenizer(replacement: '^', addPrefixSpace: true, prependScheme: PrependScheme.First);
        var splits = GetSplitTexts("Hello World", metaspace);
        CollectionAssert.AreEqual(new[] { "^Hello", "^World" }, splits);
    }

    // ========== SequencePreTokenizer ==========

    [TestMethod]
    public void SequencePreTokenizer_ChainsPreTokenizers()
    {
        var seq = new SequencePreTokenizer(
            new WhitespacePreTokenizer(),
            new DigitsPreTokenizer(individualDigits: true)
        );
        var splits = GetSplitTexts("hello123 world", seq);
        CollectionAssert.AreEqual(new[] { "hello", "1", "2", "3", "world" }, splits);
    }

    [TestMethod]
    public void SequencePreTokenizer_RequiresAtLeastOne()
    {
        Assert.ThrowsExactly<ArgumentException>(() => new SequencePreTokenizer());
    }

    // ========== SplitPreTokenizer ==========

    [TestMethod]
    public void SplitPreTokenizer_WithRegexPattern()
    {
        var pattern = new RegexPattern(new System.Text.RegularExpressions.Regex(@"\d+"));
        var split = new SplitPreTokenizer(pattern, SplitDelimiterBehavior.Isolated);
        var splits = GetSplitTexts("abc123def456", split);
        CollectionAssert.AreEqual(new[] { "abc", "123", "def", "456" }, splits);
    }

    [TestMethod]
    public void SplitPreTokenizer_WithCharPattern()
    {
        var pattern = new CharPattern(',');
        var split = new SplitPreTokenizer(pattern, SplitDelimiterBehavior.Removed);
        var splits = GetSplitTexts("a,b,c", split);
        CollectionAssert.AreEqual(new[] { "a", "b", "c" }, splits);
    }

    // ========== FixedLengthPreTokenizer ==========

    [TestMethod]
    public void FixedLengthPreTokenizer_EvenSplit()
    {
        var splits = GetSplitTexts("abcdefgh", new FixedLengthPreTokenizer(4));
        CollectionAssert.AreEqual(new[] { "abcd", "efgh" }, splits);
    }

    [TestMethod]
    public void FixedLengthPreTokenizer_UnevenSplit()
    {
        var splits = GetSplitTexts("abcde", new FixedLengthPreTokenizer(2));
        CollectionAssert.AreEqual(new[] { "ab", "cd", "e" }, splits);
    }

    [TestMethod]
    public void FixedLengthPreTokenizer_LengthMustBePositive()
    {
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => new FixedLengthPreTokenizer(0));
        Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => new FixedLengthPreTokenizer(-1));
    }

    // ========== UnicodeScriptsPreTokenizer ==========

    [TestMethod]
    public void UnicodeScriptsPreTokenizer_SplitsScriptBoundaries()
    {
        var splits = GetSplitTexts("helloмир", new UnicodeScriptsPreTokenizer());
        CollectionAssert.AreEqual(new[] { "hello", "мир" }, splits);
    }

    [TestMethod]
    public void UnicodeScriptsPreTokenizer_KeepsSameScriptTogether()
    {
        var splits = GetSplitTexts("hello world", new UnicodeScriptsPreTokenizer());
        Assert.AreEqual(1, splits.Count());
        Assert.AreEqual("hello world", splits[0]);
    }

    [TestMethod]
    public void UnicodeScriptsPreTokenizer_SplitsCJKFromLatin()
    {
        var splits = GetSplitTexts("hello你好", new UnicodeScriptsPreTokenizer());
        CollectionAssert.AreEqual(new[] { "hello", "你好" }, splits);
    }

    // ========== PreTokenizerWrapper ==========

    [TestMethod]
    public void PreTokenizerWrapper_DelegatesToInner()
    {
        var wrapper = new PreTokenizerWrapper(new BertPreTokenizer());
        var splits = GetSplitTexts("Hello, world!", wrapper);
        CollectionAssert.AreEqual(new[] { "Hello", ",", "world", "!" }, splits);
    }

    [TestMethod]
    public void PreTokenizerWrapper_RequiresNonNull()
    {
        Assert.ThrowsExactly<ArgumentNullException>(() => new PreTokenizerWrapper(null!));
    }

    // ========== ByteLevelPreTokenizer ==========

    [TestMethod]
    public void ByteLevelPreTokenizer_SplitsOnPattern()
    {
        var splits = GetSplitTexts("Hello world", new ByteLevelPreTokenizer());
        Assert.IsTrue(splits.Count > 0);
        Assert.IsTrue(splits.Any(s => s.Contains("Hello")));
    }

    [TestMethod]
    public void ByteLevelPreTokenizer_AddPrefixSpace()
    {
        var pre = new ByteLevelPreTokenizer(addPrefixSpace: true);
        var splits = GetSplitTexts("Hello", pre);
        Assert.IsTrue(splits.Count > 0, "应产生至少一个 split");
        // ByteLevel 编码中，空格映射为 Ġ (0xC4 0xA0)
        Assert.IsTrue(splits[0].StartsWith("Ġ"), $"addPrefixSpace=true 时首 split 应以 Ġ 开头，实际: '{splits[0]}'");
    }

    [TestMethod]
    public void ByteLevelPreTokenizer_EncodeDecode_Roundtrip()
    {
        var original = "Hello, World! 你好";
        var encoded = ByteLevelPreTokenizer.Encode(original);
        var decoded = ByteLevelPreTokenizer.Decode(encoded);
        Assert.AreEqual(original, decoded);
    }
}
