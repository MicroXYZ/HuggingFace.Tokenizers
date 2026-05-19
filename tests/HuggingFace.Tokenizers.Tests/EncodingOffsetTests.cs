using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Tests for Encoding offset/word mapping methods.
/// </summary>
    [TestClass]
public class EncodingOffsetTests
{
    /// <summary>
    /// Helper: "hello world" tokenized into two words with known offsets.
    /// tokens: ["hello", " world"]  offsets: [(0,5),(5,11)]  words: [0,1]
    /// </summary>
    private static Encoding CreateHelloWorld()
    {
        return new Encoding(
            ids:               [101, 102],
            typeIds:           [0, 0],
            tokens:            ["hello", " world"],
            words:             [0, 1],
            offsets:           [(0, 5), (5, 11)],
            specialTokensMask: [0, 0],
            attentionMask:     [1, 1]);
    }

    // ── TokenToChars ──────────────────────────────

    [TestMethod]
    public void TokenToChars_ReturnsOffset_ForValidIndex()
    {
        var enc = CreateHelloWorld();

        Assert.AreEqual((0, 5), enc.TokenToChars(0));
        Assert.AreEqual((5, 11), enc.TokenToChars(1));
    }

    [TestMethod]
    public void TokenToChars_ReturnsNull_ForOutOfBounds()
    {
        var enc = CreateHelloWorld();

        Assert.IsNull(enc.TokenToChars(-1));
        Assert.IsNull(enc.TokenToChars(2));
    }

    // ── CharToToken ───────────────────────────────

    [TestMethod]
    public void CharToToken_FindsToken_ForCharInsideFirstWord()
    {
        var enc = CreateHelloWorld();

        Assert.AreEqual(0, enc.CharToToken(0));   // 'h'
        Assert.AreEqual(0, enc.CharToToken(4));   // 'o'
    }

    [TestMethod]
    public void CharToToken_FindsToken_ForCharInsideSecondWord()
    {
        var enc = CreateHelloWorld();

        Assert.AreEqual(1, enc.CharToToken(5));   // ' '
        Assert.AreEqual(1, enc.CharToToken(10));  // 'd'
    }

    [TestMethod]
    public void CharToToken_ReturnsNull_ForOutOfBounds()
    {
        var enc = CreateHelloWorld();

        Assert.IsNull(enc.CharToToken(-1));
        Assert.IsNull(enc.CharToToken(11));  // past end
    }

    // ── TokenToWord ───────────────────────────────

    [TestMethod]
    public void TokenToWord_ReturnsWordId()
    {
        var enc = CreateHelloWorld();

        Assert.AreEqual(0u, enc.TokenToWord(0));
        Assert.AreEqual(1u, enc.TokenToWord(1));
    }

    [TestMethod]
    public void TokenToWord_ReturnsNull_ForOutOfBounds()
    {
        var enc = CreateHelloWorld();

        Assert.IsNull(enc.TokenToWord(-1));
        Assert.IsNull(enc.TokenToWord(2));
    }

    // ── CharToWord ────────────────────────────────

    [TestMethod]
    public void CharToWord_MapsCharToWordId()
    {
        var enc = CreateHelloWorld();

        Assert.AreEqual(0u, enc.CharToWord(2));  // inside "hello"
        Assert.AreEqual(1u, enc.CharToWord(7));  // inside " world"
    }

    [TestMethod]
    public void CharToWord_ReturnsNull_ForOutOfBounds()
    {
        var enc = CreateHelloWorld();

        Assert.IsNull(enc.CharToWord(-1));
        Assert.IsNull(enc.CharToWord(100));
    }

    // ── WordToTokens ──────────────────────────────

    [TestMethod]
    public void WordToTokens_ReturnsRange_ForMatchingWord()
    {
        // 4 tokens, words: [0, 0, 1, 1] → word 0 spans tokens 0-1
        var enc = new Encoding(
            ids:               [10, 11, 12, 13],
            typeIds:           [0, 0, 0, 0],
            tokens:            ["hel", "lo", "wor", "ld"],
            words:             [0, 0, 1, 1],
            offsets:           [(0, 3), (3, 5), (6, 9), (9, 11)],
            specialTokensMask: [0, 0, 0, 0],
            attentionMask:     [1, 1, 1, 1]);

        var result = enc.WordToTokens(0);
        Assert.IsNotNull(result);
        Assert.AreEqual((0, 2), result!.Value);

        var result1 = enc.WordToTokens(1);
        Assert.IsNotNull(result1);
        Assert.AreEqual((2, 2), result1!.Value);
    }

    [TestMethod]
    public void WordToTokens_ReturnsNull_ForMissingWord()
    {
        var enc = CreateHelloWorld();

        Assert.IsNull(enc.WordToTokens(99));
    }

    // ── Multi-sequence scenario ───────────────────

    [TestMethod]
    public void CharToToken_RespectsSequenceRange()
    {
        // Two sequences merged: seq 0 = tokens 0..2, seq 1 = tokens 2..4
        // seq 0 offsets: (0,3)(3,5)  seq 1 offsets: (0,4)(4,7)
        var enc = new Encoding(
            ids:               [10, 11, 20, 21],
            typeIds:           [0, 0, 1, 1],
            tokens:            ["abc", "de", "xyz", "w"],
            words:             [0, 0, 1, 1],
            offsets:           [(0, 3), (3, 5), (0, 4), (4, 7)],
            specialTokensMask: [0, 0, 0, 0],
            attentionMask:     [1, 1, 1, 1],
            sequenceRanges:    new Dictionary<int, Range> { [0] = 0..2, [1] = 2..4 });

        // char 2 in seq 0 → token 0 (offset (0,3))
        Assert.AreEqual(0, enc.CharToToken(2, sequenceId: 0));
        // char 0 in seq 1 → token 2 (offset (0,4), within range 2..4)
        Assert.AreEqual(2, enc.CharToToken(0, sequenceId: 1));
        // char 100 in seq 0 → null
        Assert.IsNull(enc.CharToToken(100, sequenceId: 0));
    }

    // ── TokenToSequence ───────────────────────────

    [TestMethod]
    public void TokenToSequence_ReturnsCorrectSequenceId()
    {
        var enc = new Encoding(
            ids:               [10, 11, 20, 21],
            typeIds:           [0, 0, 1, 1],
            tokens:            ["abc", "de", "xyz", "w"],
            words:             [0, 0, 1, 1],
            offsets:           [(0, 3), (3, 5), (0, 4), (4, 7)],
            specialTokensMask: [0, 0, 0, 0],
            attentionMask:     [1, 1, 1, 1],
            sequenceRanges:    new Dictionary<int, Range> { [0] = 0..2, [1] = 2..4 });

        Assert.AreEqual(0, enc.TokenToSequence(0));
        Assert.AreEqual(0, enc.TokenToSequence(1));
        Assert.AreEqual(1, enc.TokenToSequence(2));
        Assert.AreEqual(1, enc.TokenToSequence(3));
    }

    [TestMethod]
    public void TokenToSequence_ReturnsZero_ForSingleSequence()
    {
        var enc = CreateHelloWorld();

        Assert.AreEqual(0, enc.TokenToSequence(0));
        Assert.AreEqual(0, enc.TokenToSequence(1));
    }

    [TestMethod]
    public void TokenToSequence_ReturnsNull_ForOutOfBounds()
    {
        var enc = CreateHelloWorld();

        Assert.IsNull(enc.TokenToSequence(-1));
        Assert.IsNull(enc.TokenToSequence(99));
    }
}
