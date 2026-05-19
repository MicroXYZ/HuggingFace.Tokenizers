using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Padding;

namespace HuggingFace.Tokenizers.Tests;

    [TestClass]
public class PaddingTests
{
    private static Encoding CreateEncoding(uint[] ids)
    {
        var tokens = ids.Select(id => $"tok_{id}").ToArray();
        var words = ids.Select((_, i) => (uint?)i).ToArray();
        var offsets = ids.Select((_, i) => (i * 2, i * 2 + 2)).ToArray();
        var typeIds = new uint[ids.Length];
        var specialMask = new uint[ids.Length];
        var attention = Enumerable.Repeat(1u, ids.Length).ToArray();
        return new Encoding(ids, typeIds, tokens, words, offsets, specialMask, attention);
    }

    [TestMethod]
    public void Pad_Right_PadsCorrectly()
    {
        var encoding = CreateEncoding([1, 2, 3]);
        encoding.Pad(5, padId: 99, direction: PaddingDirection.Right);

        Assert.AreEqual(5, encoding.Length);
        CollectionAssert.AreEqual(new uint[] { 1u, 2u, 3u, 99u, 99u }, encoding.GetIds());
        CollectionAssert.AreEqual(new[] { "tok_1", "tok_2", "tok_3", "[PAD]", "[PAD]" }, encoding.GetTokens());
        // Words should be null for padding positions
        var wordIds = encoding.GetWordIds();
        Assert.AreEqual(0u, wordIds[0]);
        Assert.IsNull(wordIds[3]);
        Assert.IsNull(wordIds[4]);
        // Attention mask: real tokens = 1, padding = 0
        CollectionAssert.AreEqual(new uint[] { 1u, 1u, 1u, 0u, 0u }, encoding.GetAttentionMask());
    }

    [TestMethod]
    public void Pad_Left_PadsCorrectly()
    {
        var encoding = CreateEncoding([1, 2, 3]);
        encoding.Pad(5, padId: 99, direction: PaddingDirection.Left);

        Assert.AreEqual(5, encoding.Length);
        CollectionAssert.AreEqual(new uint[] { 99u, 99u, 1u, 2u, 3u }, encoding.GetIds());
        CollectionAssert.AreEqual(new[] { "[PAD]", "[PAD]", "tok_1", "tok_2", "tok_3" }, encoding.GetTokens());
    }

    [TestMethod]
    public void Pad_Left_AttentionMask_PaddingIsZero()
    {
        var encoding = CreateEncoding([1, 2, 3]);
        encoding.Pad(6, padId: 0, direction: PaddingDirection.Left);

        Assert.AreEqual(6, encoding.Length);
        // Left padding: [PAD PAD PAD 1 2 3] → attention [0 0 0 1 1 1]
        CollectionAssert.AreEqual(new uint[] { 0u, 0u, 0u, 1u, 1u, 1u }, encoding.GetAttentionMask());
    }

    [TestMethod]
    public void Pad_Left_FullVerification()
    {
        var encoding = CreateEncoding([10, 20]);
        encoding.Pad(5, padId: 0, padTypeId: 1, padToken: "<P>", direction: PaddingDirection.Left);

        Assert.AreEqual(5, encoding.Length);
        // IDs: [0, 0, 10, 20]
        CollectionAssert.AreEqual(new uint[] { 0u, 0u, 0u, 10u, 20u }, encoding.GetIds());
        // Tokens
        Assert.AreEqual("<P>", encoding.GetTokens()[0]);
        Assert.AreEqual("<P>", encoding.GetTokens()[1]);
        Assert.AreEqual("<P>", encoding.GetTokens()[2]);
        Assert.AreEqual("tok_10", encoding.GetTokens()[3]);
        Assert.AreEqual("tok_20", encoding.GetTokens()[4]);
        // Attention mask: padding=0, real=1
        CollectionAssert.AreEqual(new uint[] { 0u, 0u, 0u, 1u, 1u }, encoding.GetAttentionMask());
        // Type IDs: padding gets padTypeId=1
        CollectionAssert.AreEqual(new uint[] { 1u, 1u, 1u, 0u, 0u }, encoding.GetTypeIds());
    }

    [TestMethod]
    public void Pad_Right_AttentionMask_AllRealTokensOne()
    {
        var encoding = CreateEncoding([5, 6, 7, 8]);
        encoding.Pad(7, padId: 0, direction: PaddingDirection.Right);

        CollectionAssert.AreEqual(new uint[] { 1u, 1u, 1u, 1u, 0u, 0u, 0u }, encoding.GetAttentionMask());
    }

    [TestMethod]
    public void Pad_PadToMultipleOf_RoundsUpToMultiple()
    {
        var encoding = CreateEncoding([1, 2, 3, 4, 5]);
        encoding.Pad(8, padId: 0);

        // 8 is a multiple of 4, so length should be 8
        Assert.AreEqual(8, encoding.Length);
        CollectionAssert.AreEqual(new uint[] { 1u, 1u, 1u, 1u, 1u, 0u, 0u, 0u }, encoding.GetAttentionMask());
    }

    [TestMethod]
    public void Pad_AlreadyLonger_DoesNothing()
    {
        var encoding = CreateEncoding([1, 2, 3, 4, 5]);
        encoding.Pad(3);

        Assert.AreEqual(5, encoding.Length);
        CollectionAssert.AreEqual(new uint[] { 1u, 2u, 3u, 4u, 5u }, encoding.GetIds());
    }

    [TestMethod]
    public void Pad_CustomPadToken_UsedCorrectly()
    {
        var encoding = CreateEncoding([1]);
        encoding.Pad(3, padId: 0, padToken: "<pad>");

        Assert.AreEqual(3, encoding.Length);
        Assert.AreEqual("<pad>", encoding.GetTokens()[1]);
        Assert.AreEqual("<pad>", encoding.GetTokens()[2]);
    }

    [TestMethod]
    public void PadEncodings_BatchLongest_AllSameLength()
    {
        var encodings = new List<Encoding>
        {
            CreateEncoding([1, 2]),
            CreateEncoding([3, 4, 5, 6]),
            CreateEncoding([7]),
        };

        var paddingParams = new PaddingParams
        {
            Strategy = PaddingStrategy.BatchLongest,
            PadId = 0,
        };

        PaddingHelper.PadEncodings(encodings, paddingParams);

        foreach (var e in encodings) Assert.AreEqual(4, e.Length);
        CollectionAssert.AreEqual(new uint[] { 1u, 2u, 0u, 0u }, encodings[0].GetIds());
        CollectionAssert.AreEqual(new uint[] { 3u, 4u, 5u, 6u }, encodings[1].GetIds());
        CollectionAssert.AreEqual(new uint[] { 7u, 0u, 0u, 0u }, encodings[2].GetIds());
    }

    [TestMethod]
    public void PadEncodings_Fixed_PadsToSpecifiedLength()
    {
        var encodings = new List<Encoding>
        {
            CreateEncoding([1, 2]),
            CreateEncoding([3]),
        };

        var paddingParams = new PaddingParams
        {
            Strategy = PaddingStrategy.Fixed,
            MaxLength = 5,
            PadId = 0,
        };

        PaddingHelper.PadEncodings(encodings, paddingParams);

        foreach (var e in encodings) Assert.AreEqual(5, e.Length);
        CollectionAssert.AreEqual(new uint[] { 1u, 2u, 0u, 0u, 0u }, encodings[0].GetIds());
        CollectionAssert.AreEqual(new uint[] { 3u, 0u, 0u, 0u, 0u }, encodings[1].GetIds());
    }

    [TestMethod]
    public void PadEncodings_Fixed_ShorterThanLongest_UsesLongest()
    {
        var encodings = new List<Encoding>
        {
            CreateEncoding([1, 2, 3, 4, 5, 6, 7]),
        };

        var paddingParams = new PaddingParams
        {
            Strategy = PaddingStrategy.Fixed,
            MaxLength = 3, // shorter than longest
            PadId = 0,
        };

        PaddingHelper.PadEncodings(encodings, paddingParams);

        // Should not truncate, length stays at 7
        Assert.AreEqual(7, encodings[0].Length);
    }

    [TestMethod]
    public void PadEncodings_EmptyList_DoesNothing()
    {
        var encodings = new List<Encoding>();
        var paddingParams = new PaddingParams { Strategy = PaddingStrategy.BatchLongest };
        PaddingHelper.PadEncodings(encodings, paddingParams); // should not throw
    }

    [TestMethod]
    public void PadEncodings_PadToMultipleOf_RoundsUp()
    {
        var encodings = new List<Encoding>
        {
            CreateEncoding([1, 2, 3]),
        };

        var paddingParams = new PaddingParams
        {
            Strategy = PaddingStrategy.BatchLongest,
            PadId = 0,
            PadToMultipleOf = 8,
        };

        PaddingHelper.PadEncodings(encodings, paddingParams);

        Assert.AreEqual(8, encodings[0].Length);
    }

    [TestMethod]
    public void PadEncodings_PadToMultipleOf_AlreadyAligned()
    {
        var encodings = new List<Encoding>
        {
            CreateEncoding([1, 2, 3, 4]),
        };

        var paddingParams = new PaddingParams
        {
            Strategy = PaddingStrategy.BatchLongest,
            PadId = 0,
            PadToMultipleOf = 4,
        };

        PaddingHelper.PadEncodings(encodings, paddingParams);

        Assert.AreEqual(4, encodings[0].Length);
    }

    [TestMethod]
    public void PadEncodings_BatchLongest_WithPadToMultipleOf()
    {
        var encodings = new List<Encoding>
        {
            CreateEncoding([1, 2]),
            CreateEncoding([3, 4, 5]),
        };

        var paddingParams = new PaddingParams
        {
            Strategy = PaddingStrategy.BatchLongest,
            PadId = 0,
            PadToMultipleOf = 4,
        };

        PaddingHelper.PadEncodings(encodings, paddingParams);

        // Longest is 3, round up to multiple of 4 = 4
        foreach (var e in encodings) Assert.AreEqual(4, e.Length);
    }

    [TestMethod]
    public void PadEncodings_Left_PadsOnLeft()
    {
        var encodings = new List<Encoding>
        {
            CreateEncoding([1, 2]),
            CreateEncoding([3, 4, 5]),
        };

        var paddingParams = new PaddingParams
        {
            Strategy = PaddingStrategy.BatchLongest,
            PadId = 99,
            Direction = PaddingDirection.Left,
        };

        PaddingHelper.PadEncodings(encodings, paddingParams);

        Assert.AreEqual(3, encodings[0].Length);
        CollectionAssert.AreEqual(new uint[] { 99u, 1u, 2u }, encodings[0].GetIds());
        CollectionAssert.AreEqual(new uint[] { 3u, 4u, 5u }, encodings[1].GetIds());
    }

    // ── R1C: Pad Left + SequenceRanges ────────────────────────────────────

    [TestMethod]
    public void Pad_Left_SequenceRanges_Updated()
    {
        var encoding = CreateEncoding([1, 2, 3]);
        // Set sequence ranges before padding
        encoding.SetSequenceId(0);
        var rangeBefore = encoding.SequenceRange(0);
        Assert.AreEqual(0, rangeBefore.Start.Value);
        Assert.AreEqual(3, rangeBefore.End.Value);

        encoding.Pad(6, padId: 99, direction: PaddingDirection.Left);

        // After left padding by 3, sequence range should shift right by 3
        var rangeAfter = encoding.SequenceRange(0);
        Assert.AreEqual(3, rangeAfter.Start.Value);
        Assert.AreEqual(6, rangeAfter.End.Value);

        // CharToToken works with CHARACTER positions in the original string.
        // CreateEncoding offsets: token 0→(0,2), token 1→(2,4), token 2→(4,6)
        // After left padding, real tokens are at indices 3,4,5 in the encoding.
        // CharToToken searches within SequenceRange(0) = 3..6
        Assert.AreEqual(3, encoding.CharToToken(0, 0)); // char 0 → token 3 (offsets 0,2)
        Assert.AreEqual(3, encoding.CharToToken(1, 0)); // char 1 → token 3 (offsets 0,2)
        Assert.AreEqual(4, encoding.CharToToken(2, 0)); // char 2 → token 4 (offsets 2,4)
        Assert.AreEqual(4, encoding.CharToToken(3, 0)); // char 3 → token 4 (offsets 2,4)
        Assert.AreEqual(5, encoding.CharToToken(4, 0)); // char 4 → token 5 (offsets 4,6)
        Assert.AreEqual(5, encoding.CharToToken(5, 0)); // char 5 → token 5 (offsets 4,6)
    }

    [TestMethod]
    public void Pad_Right_SequenceRanges_Unchanged()
    {
        var encoding = CreateEncoding([1, 2, 3]);
        encoding.SetSequenceId(0);

        encoding.Pad(6, padId: 99, direction: PaddingDirection.Right);

        // Right padding: sequence range should stay at 0..3
        var range = encoding.SequenceRange(0);
        Assert.AreEqual(0, range.Start.Value);
        Assert.AreEqual(3, range.End.Value);
    }
}
