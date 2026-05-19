using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Models.BPE;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Tests for truncation behavior.
/// </summary>
    [TestClass]
public class TruncationTests
{
    private static Tokenizer CreateTokenizerWithTruncation(
        TruncationStrategy strategy,
        int maxLength,
        int stride = 0)
    {
        var vocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0,
            ["a"] = 1, ["b"] = 2, ["c"] = 3, ["d"] = 4, ["e"] = 5,
            ["f"] = 6, ["g"] = 7, ["h"] = 8, ["i"] = 9, ["j"] = 10,
            [" "] = 11
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges([])
            .SetUnkToken("<unk>")
            .Build();

        var tokenizer = new Tokenizer(model);
        tokenizer.Truncation = new TruncationParams
        {
            Strategy = strategy,
            MaxLength = maxLength,
            Stride = stride
        };

        return tokenizer;
    }

    [TestMethod]
    public void LongestFirst_NoPair_TruncatesFirst()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.LongestFirst, maxLength: 3);

        var encoding = tokenizer.Encode("abcde");

        Assert.AreEqual(3, encoding.Length);
    }

    [TestMethod]
    public void LongestFirst_Pair_TruncatesLongerSequence()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.LongestFirst, maxLength: 5);

        // First: "abcde" → 5 tokens, Second: "ab" → 2 tokens
        // Total = 7, overflow = 2, truncate first (longer)
        var encoding = tokenizer.EncodePair("abcde", "ab");

        // After truncation: first = 3 tokens, second = 2 tokens, total = 5
        Assert.AreEqual(5, encoding.Length);
    }

    [TestMethod]
    public void LongestFirst_Pair_SecondIsLonger_TruncatesSecond()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.LongestFirst, maxLength: 4);

        // First: "ab" → 2 tokens, Second: "abcde" → 5 tokens
        // Total = 7, overflow = 3, truncate second (longer)
        var encoding = tokenizer.EncodePair("ab", "abcde");

        // After truncation: first = 2, second = 2, total = 4
        Assert.AreEqual(4, encoding.Length);
    }

    [TestMethod]
    public void OnlyFirst_TruncatesFirstSequence()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.OnlyFirst, maxLength: 3);

        // "abcde" → 5 tokens, overflow = 2
        var encoding = tokenizer.EncodePair("abcde", null);

        Assert.AreEqual(3, encoding.Length);
    }

    [TestMethod]
    public void OnlyFirst_Pair_OnlyFirstTruncated()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.OnlyFirst, maxLength: 4);

        // First: "abcde" → 5, Second: "ab" → 2, total = 7, overflow = 3
        // OnlyFirst: truncate first by 3 → first = 2, second = 2, total = 4
        var encoding = tokenizer.EncodePair("abcde", "ab");

        Assert.AreEqual(4, encoding.Length);
    }

    [TestMethod]
    public void OnlySecond_TruncatesSecondSequence()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.OnlySecond, maxLength: 4);

        // First: "ab" → 2, Second: "abcde" → 5, total = 7, overflow = 3
        // OnlySecond: truncate second by 3 → first = 2, second = 2, total = 4
        var encoding = tokenizer.EncodePair("ab", "abcde");

        Assert.AreEqual(4, encoding.Length);
    }

    [TestMethod]
    public void OnlySecond_NoPair_DoesNothing()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.OnlySecond, maxLength: 3);

        // "abcde" → 5 tokens, but OnlySecond with no pair → no truncation
        var encoding = tokenizer.Encode("abcde");

        // No pair to truncate, so first stays at 5
        Assert.AreEqual(5, encoding.Length);
    }

    [TestMethod]
    public void Truncation_UnderMaxLength_NoTruncation()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.LongestFirst, maxLength: 10);

        var encoding = tokenizer.Encode("abc");

        Assert.AreEqual(3, encoding.Length);
    }

    [TestMethod]
    public void Truncation_ExactMaxLength_NoTruncation()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.LongestFirst, maxLength: 5);

        var encoding = tokenizer.Encode("abcde");

        Assert.AreEqual(5, encoding.Length);
    }

    [TestMethod]
    public void Truncation_VeryLongSequence_TruncatedToMaxLength()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.LongestFirst, maxLength: 2);

        var encoding = tokenizer.Encode("abcde");

        Assert.AreEqual(2, encoding.Length);
    }

    // ──────────────────────────────────────────────
    //  Left direction truncation tests
    // ──────────────────────────────────────────────

    [TestMethod]
    public void Left_SingleSequence_KeepsTailTokens()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.OnlyFirst, maxLength: 3);
        tokenizer.Truncation!.Direction = TruncationDirection.Left;

        // "abcde" → [a,b,c,d,e], truncate left → keep [c,d,e]
        var encoding = tokenizer.Encode("abcde");

        Assert.AreEqual(3, encoding.Length);
        var ids = encoding.GetIds();
        // Should keep the last 3 tokens: c=3, d=4, e=5
        Assert.AreEqual(3u, ids[0]); // c
        Assert.AreEqual(4u, ids[1]); // d
        Assert.AreEqual(5u, ids[2]); // e
    }

    [TestMethod]
    public void Left_Pair_TruncatesFirstFromHead()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.OnlyFirst, maxLength: 4);
        tokenizer.Truncation!.Direction = TruncationDirection.Left;

        // First: "abcde" → 5, Second: "ab" → 2, total = 7, overflow = 3
        // OnlyFirst: truncate first to maxLen=2 (from left), keep [d,e]
        var encoding = tokenizer.EncodePair("abcde", "ab");

        Assert.AreEqual(4, encoding.Length); // 2 + 2
        var ids = encoding.GetIds();
        // First part: d=4, e=5 (kept from tail)
        Assert.AreEqual(4u, ids[0]);
        Assert.AreEqual(5u, ids[1]);
    }

    [TestMethod]
    public void Left_Overflowing_ContainsHeadTokens()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.OnlyFirst, maxLength: 2);
        tokenizer.Truncation!.Direction = TruncationDirection.Left;

        // "abcde" → [a,b,c,d,e], truncate left to 2 → keep [d,e]
        // Left windows (offset=2): (3,5), (1,3), (0,1)
        // Main: [d,e], Overflow: [b,c], [a]
        var encoding = tokenizer.Encode("abcde");

        Assert.AreEqual(2, encoding.Length);
        Assert.AreEqual(2, encoding.GetOverflowing().Count);

        var overflow0 = encoding.GetOverflowing()[0]; // [b,c]
        Assert.AreEqual(2, overflow0.Length);
        Assert.AreEqual(2u, overflow0.GetIds()[0]); // b
        Assert.AreEqual(3u, overflow0.GetIds()[1]); // c

        var overflow1 = encoding.GetOverflowing()[1]; // [a]
        Assert.AreEqual(1, overflow1.GetIds().Count());
        Assert.AreEqual(1u, overflow1.GetIds()[0]); // a
    }

    // ──────────────────────────────────────────────
    //  Stride + Overflowing tests
    // ──────────────────────────────────────────────

    [TestMethod]
    public void Stride_CreatesOverflowingEncoding()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.OnlyFirst, maxLength: 4, stride: 2);
        tokenizer.Truncation!.Direction = TruncationDirection.Right;

        // "abcde" → [a,b,c,d,e], truncate to 4 with stride=2
        // Main: [a,b,c,d], overflow: [c,d,e] (offset = 4-2 = 2)
        var encoding = tokenizer.Encode("abcde");

        Assert.AreEqual(4, encoding.Length);
        Assert.AreEqual(1, encoding.GetOverflowing().Count());

        var overflow = encoding.GetOverflowing()[0];
        Assert.AreEqual(3, overflow.Length); // [c,d,e]
        Assert.AreEqual(3u, overflow.GetIds()[0]); // c
        Assert.AreEqual(4u, overflow.GetIds()[1]); // d
        Assert.AreEqual(5u, overflow.GetIds()[2]); // e
    }

    [TestMethod]
    public void Stride_Right_MultipleOverflowingWindows()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.OnlyFirst, maxLength: 3, stride: 1);
        tokenizer.Truncation!.Direction = TruncationDirection.Right;

        // "abcdefghij" → 10 tokens, truncate to 3 with stride=1
        // offset = 3-1 = 2
        // Windows: (0,3), (2,5), (4,7), (6,9), (8,10)
        // Main: [a,b,c], Overflow: [c,d,e], [e,f,g], [g,h,i], [i,j]
        var encoding = tokenizer.Encode("abcdefghij");

        Assert.AreEqual(3, encoding.Length);
        Assert.AreEqual(4, encoding.GetOverflowing().Count);

        Assert.AreEqual(3u, encoding.GetOverflowing()[0].GetIds()[0]); // c
        Assert.AreEqual(5u, encoding.GetOverflowing()[1].GetIds()[0]); // e
        Assert.AreEqual(7u, encoding.GetOverflowing()[2].GetIds()[0]); // g
        Assert.AreEqual(9u, encoding.GetOverflowing()[3].GetIds()[0]); // i
    }

    [TestMethod]
    public void Stride_Left_CreatesOverflowingFromHead()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.OnlyFirst, maxLength: 4, stride: 2);
        tokenizer.Truncation!.Direction = TruncationDirection.Left;

        // "abcde" → [a,b,c,d,e], truncate left to 4 with stride=2
        // offset = 4-2 = 2
        // Left windows from end: stop=5,start=1 → (1,5), stop=3,start=0 → (0,3)
        // Main: [b,c,d,e], Overflow: [a,b,c]
        var encoding = tokenizer.Encode("abcde");

        Assert.AreEqual(4, encoding.Length);
        var ids = encoding.GetIds();
        Assert.AreEqual(2u, ids[0]); // b
        Assert.AreEqual(5u, ids[3]); // e

        Assert.AreEqual(1, encoding.GetOverflowing().Count());
        var overflow = encoding.GetOverflowing()[0];
        Assert.AreEqual(3, overflow.Length);
        Assert.AreEqual(1u, overflow.GetIds()[0]); // a
        Assert.AreEqual(3u, overflow.GetIds()[2]); // c
    }

    [TestMethod]
    public void Stride_NoStride_StillCreatesOverflowing()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.OnlyFirst, maxLength: 3, stride: 0);
        tokenizer.Truncation!.Direction = TruncationDirection.Right;

        // "abcde" → [a,b,c,d,e], truncate to 3, no stride
        // Main: [a,b,c], Overflow: [d,e]
        var encoding = tokenizer.Encode("abcde");

        Assert.AreEqual(3, encoding.Length);
        Assert.AreEqual(1, encoding.GetOverflowing().Count());

        var overflow = encoding.GetOverflowing()[0];
        Assert.AreEqual(2, overflow.Length);
        Assert.AreEqual(4u, overflow.GetIds()[0]); // d
        Assert.AreEqual(5u, overflow.GetIds()[1]); // e
    }

    // ──────────────────────────────────────────────
    //  LongestFirst n1/n2 fine algorithm tests
    // ──────────────────────────────────────────────

    [TestMethod]
    public void LongestFirst_Pair_BothTruncated_WhenBothLong()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.LongestFirst, maxLength: 4);

        // First: "abcde" → 5, Second: "abcde" → 5, total = 10, overflow = 6
        // n1=5, n2=5, swap=false, n1<=maxLen? 5>4 → n2=5, n1+n2=10>4
        // n1=4/2=2, n2=2+4%2=2+0=2, swap=false → (2, 2)
        var encoding = tokenizer.EncodePair("abcde", "abcde");

        Assert.AreEqual(4, encoding.Length); // 2 + 2
    }

    [TestMethod]
    public void LongestFirst_Pair_ShortFirst_PreservesAndTruncatesSecond()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.LongestFirst, maxLength: 6);

        // First: "ab" → 2, Second: "abcdefghij" → 10, total = 12, overflow = 6
        // n1=2, n2=10, swap=true → n1=2, n2=10, n1<=maxLen? yes → n2=max(2, 6-2)=4
        // n1+n2=2+4=6 ≤ 6, no further truncation. swap back → (4, 2)
        // Wait: swap was true, so n1=n2_orig=10→4, n2=n1_orig=2
        // Result: encoding=4, pairEncoding=2, total=6
        var encoding = tokenizer.EncodePair("ab", "abcdefghij");

        Assert.AreEqual(6, encoding.Length); // 4 + 2
    }

    [TestMethod]
    public void LongestFirst_Pair_EqualLength_SplitsEvenly()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.LongestFirst, maxLength: 3);

        // First: "abcd" → 4, Second: "abcd" → 4, total = 8, overflow = 5
        // n1=4, n2=4, swap=false, n1>maxLen? 4>3 → n2=4
        // n1+n2=8>3 → n1=3/2=1, n2=1+3%2=1+1=2, swap=false → (1, 2)
        var encoding = tokenizer.EncodePair("abcd", "abcd");

        Assert.AreEqual(3, encoding.Length); // 1 + 2
    }

    // ──────────────────────────────────────────────
    //  OnlySecond with null pair tests
    // ──────────────────────────────────────────────

    [TestMethod]
    public void OnlySecond_NullPair_NoTruncation()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.OnlySecond, maxLength: 2);

        // "abcde" → 5 tokens, OnlySecond with null pair → no truncation
        var encoding = tokenizer.Encode("abcde");

        Assert.AreEqual(5, encoding.Length);
    }

    [TestMethod]
    public void OnlySecond_NullPair_EncodePair()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.OnlySecond, maxLength: 2);

        // EncodePair with null second → same as OnlySecond with null pair
        var encoding = tokenizer.EncodePair("abcde", null);

        Assert.AreEqual(5, encoding.Length);
    }

    // ──────────────────────────────────────────────
    //  Overflowing content verification
    // ──────────────────────────────────────────────

    [TestMethod]
    public void Overflowing_TokensMatchOriginal()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.OnlyFirst, maxLength: 3, stride: 1);
        tokenizer.Truncation!.Direction = TruncationDirection.Right;

        // "abcde" → [a,b,c,d,e], truncate to 3, stride=1
        // offset = 3-1 = 2
        // Main: [a,b,c], Overflow: [c,d,e]
        var encoding = tokenizer.Encode("abcde");

        var mainIds = encoding.GetIds();
        var mainTokens = encoding.GetTokens();
        Assert.AreEqual(3, mainIds.Length);
        Assert.AreEqual("a", mainTokens[0]);
        Assert.AreEqual("b", mainTokens[1]);
        Assert.AreEqual("c", mainTokens[2]);

        var overflow = encoding.GetOverflowing()[0];
        var ovIds = overflow.GetIds();
        var ovTokens = overflow.GetTokens();
        Assert.AreEqual(3, ovIds.Length);
        Assert.AreEqual("c", ovTokens[0]);
        Assert.AreEqual("d", ovTokens[1]);
        Assert.AreEqual("e", ovTokens[2]);
    }

    [TestMethod]
    public void Overflowing_OffsetsPreserved()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.OnlyFirst, maxLength: 3, stride: 0);
        tokenizer.Truncation!.Direction = TruncationDirection.Right;

        // "abcde" → [a,b,c,d,e], truncate to 3, no stride
        // Main: [a,b,c], Overflow: [d,e]
        var encoding = tokenizer.Encode("abcde");

        var mainOffsets = encoding.GetOffsets();
        Assert.AreEqual((0, 1), mainOffsets[0]); // a
        Assert.AreEqual((1, 2), mainOffsets[1]); // b
        Assert.AreEqual((2, 3), mainOffsets[2]); // c

        var overflow = encoding.GetOverflowing()[0];
        var ovOffsets = overflow.GetOffsets();
        Assert.AreEqual((3, 4), ovOffsets[0]); // d
        Assert.AreEqual((4, 5), ovOffsets[1]); // e
    }

    [TestMethod]
    public void TruncateToEmpty_MovesAllToOverflowing()
    {
        var tokenizer = CreateTokenizerWithTruncation(TruncationStrategy.OnlyFirst, maxLength: 0);
        tokenizer.Truncation!.Direction = TruncationDirection.Right;

        // "abcde" → [a,b,c,d,e], truncate to 0 → empty main, all in overflow
        var encoding = tokenizer.EncodePair("abcde", null);

        Assert.AreEqual(0, encoding.Length);
        Assert.AreEqual(1, encoding.GetOverflowing().Count());
        Assert.AreEqual(5, encoding.GetOverflowing()[0].Length);
    }
}
