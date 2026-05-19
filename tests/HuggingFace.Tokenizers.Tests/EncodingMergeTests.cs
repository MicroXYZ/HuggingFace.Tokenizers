using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Tests for Encoding.MergeWith and Encoding.Merge.
/// </summary>
    [TestClass]
public class EncodingMergeTests
{
    private static Encoding CreateEncoding(uint[] ids, (int Start, int End)[] offsets)
    {
        var tokens = ids.Select(id => $"tok_{id}").ToArray();
        var words = ids.Select((_, i) => (uint?)i).ToArray();
        var typeIds = new uint[ids.Length];
        var specialMask = new uint[ids.Length];
        var attention = Enumerable.Repeat(1u, ids.Length).ToArray();
        return new Encoding(ids, typeIds, tokens, words, offsets, specialMask, attention);
    }

    [TestMethod]
    public void MergeWith_GrowingOffsets_True_OffsetsAdjusted()
    {
        var enc1 = CreateEncoding([1, 2], [(0, 5), (6, 11)]);
        var enc2 = CreateEncoding([3, 4], [(0, 3), (4, 7)]);

        enc1.MergeWith(enc2, growingOffsets: true);

        Assert.AreEqual(4, enc1.Length);
        CollectionAssert.AreEqual(new uint[] { 1u, 2u, 3u, 4u }, enc1.GetIds());

        var offsets = enc1.GetOffsets();
        // enc1's last offset ends at 11, so enc2's offsets are shifted by 11
        Assert.AreEqual((0, 5), offsets[0]);     // unchanged
        Assert.AreEqual((6, 11), offsets[1]);    // unchanged
        Assert.AreEqual((0 + 11, 3 + 11), offsets[2]);  // (11, 14)
        Assert.AreEqual((4 + 11, 7 + 11), offsets[3]);  // (15, 18)
    }

    [TestMethod]
    public void MergeWith_GrowingOffsets_False_OffsetsUnchanged()
    {
        var enc1 = CreateEncoding([1, 2], [(0, 5), (6, 11)]);
        var enc2 = CreateEncoding([3, 4], [(0, 3), (4, 7)]);

        enc1.MergeWith(enc2, growingOffsets: false);

        Assert.AreEqual(4, enc1.Length);
        CollectionAssert.AreEqual(new uint[] { 1u, 2u, 3u, 4u }, enc1.GetIds());

        var offsets = enc1.GetOffsets();
        Assert.AreEqual((0, 5), offsets[0]);
        Assert.AreEqual((6, 11), offsets[1]);
        Assert.AreEqual((0, 3), offsets[2]);   // unchanged from enc2
        Assert.AreEqual((4, 7), offsets[3]);   // unchanged from enc2
    }

    [TestMethod]
    public void MergeWith_GrowingOffsets_MergesTokens()
    {
        var enc1 = CreateEncoding([10, 20], [(0, 2), (3, 5)]);
        var enc2 = CreateEncoding([30], [(0, 4)]);

        enc1.MergeWith(enc2, growingOffsets: true);

        Assert.AreEqual(3, enc1.Length);
        CollectionAssert.AreEqual(new uint[] { 10u, 20u, 30u }, enc1.GetIds());

        var tokens = enc1.GetTokens();
        Assert.AreEqual("tok_10", tokens[0]);
        Assert.AreEqual("tok_20", tokens[1]);
        Assert.AreEqual("tok_30", tokens[2]);
    }

    [TestMethod]
    public void MergeStatic_TwoEncodings_GrowingOffsets()
    {
        var enc1 = CreateEncoding([1, 2], [(0, 3), (4, 7)]);
        var enc2 = CreateEncoding([3], [(0, 5)]);

        var merged = Encoding.Merge([enc1, enc2], growingOffsets: true);

        Assert.AreEqual(3, merged.Length);
        var offsets = merged.GetOffsets();
        Assert.AreEqual((0, 3), offsets[0]);
        Assert.AreEqual((4, 7), offsets[1]);
        Assert.AreEqual((0 + 7, 5 + 7), offsets[2]);  // shifted by enc1's last end = 7
    }

    [TestMethod]
    public void MergeStatic_MultipleEncodings_AllMerged()
    {
        var enc1 = CreateEncoding([1], [(0, 2)]);
        var enc2 = CreateEncoding([2], [(0, 3)]);
        var enc3 = CreateEncoding([3], [(0, 1)]);

        var merged = Encoding.Merge([enc1, enc2, enc3], growingOffsets: false);

        Assert.AreEqual(3, merged.Length);
        CollectionAssert.AreEqual(new uint[] { 1u, 2u, 3u }, merged.GetIds());
    }

    [TestMethod]
    public void MergeStatic_MultipleEncodings_GrowingOffsets_AccumulatesShift()
    {
        var enc1 = CreateEncoding([1], [(0, 5)]);
        var enc2 = CreateEncoding([2], [(0, 3)]);
        var enc3 = CreateEncoding([3], [(0, 4)]);

        var merged = Encoding.Merge([enc1, enc2, enc3], growingOffsets: true);

        Assert.AreEqual(3, merged.Length);
        var offsets = merged.GetOffsets();
        Assert.AreEqual((0, 5), offsets[0]);          // enc1: unchanged
        Assert.AreEqual((0 + 5, 3 + 5), offsets[1]);  // enc2: shifted by 5
        Assert.AreEqual((0 + 8, 4 + 8), offsets[2]);  // enc3: shifted by 5+3=8
    }

    [TestMethod]
    public void MergeStatic_SingleEncoding_ReturnsSame()
    {
        var enc = CreateEncoding([1, 2], [(0, 3), (4, 6)]);

        var merged = Encoding.Merge([enc], growingOffsets: true);

        Assert.AreEqual(2, merged.Length);
        CollectionAssert.AreEqual(new uint[] { 1u, 2u }, merged.GetIds());
    }

    [TestMethod]
    public void MergeStatic_EmptyList_ReturnsEmpty()
    {
        var merged = Encoding.Merge([], growingOffsets: true);

        Assert.IsTrue(merged.IsEmpty);
        Assert.AreEqual(0, merged.Length);
    }

    [TestMethod]
    public void MergeWith_AttentionMask_Merged()
    {
        var enc1 = CreateEncoding([1, 2], [(0, 2), (3, 5)]);
        var enc2 = CreateEncoding([3], [(0, 4)]);

        enc1.MergeWith(enc2, growingOffsets: false);

        var attention = enc1.GetAttentionMask();
        CollectionAssert.AreEqual(new uint[] { 1u, 1u, 1u }, attention);
    }

    [TestMethod]
    public void MergeWith_SpecialTokensMask_Merged()
    {
        var ids1 = new uint[] { 1 };
        var ids2 = new uint[] { 2 };
        var offsets = new (int, int)[] { (0, 1) };

        var enc1 = new Encoding(ids1, new uint[1], ["t1"], [(uint?)0],
            offsets, [1u], [1u]);
        var enc2 = new Encoding(ids2, new uint[1], ["t2"], [(uint?)0],
            offsets, [0u], [1u]);

        enc1.MergeWith(enc2, growingOffsets: false);

        CollectionAssert.AreEqual(new uint[] { 1u, 0u }, enc1.GetSpecialTokensMask());
    }

    // ── R1B: Overflowing merge tests ──────────────────────────────────────

    [TestMethod]
    public void MergeWith_Overflowing_CrossProduct()
    {
        // enc1 has overflow o1, enc2 has overflow o2
        // Result should have: o1+enc2, o1+o2, enc1+o2 (3 overflowings)
        var enc1 = CreateEncoding([1], [(0, 2)]);
        var o1 = CreateEncoding([10], [(0, 3)]);
        enc1.GetOverflowing().Add(o1);

        var enc2 = CreateEncoding([2], [(0, 4)]);
        var o2 = CreateEncoding([20], [(0, 5)]);
        enc2.GetOverflowing().Add(o2);

        enc1.MergeWith(enc2, growingOffsets: false);

        Assert.AreEqual(2, enc1.Length); // main: [1, 2]
        Assert.AreEqual(3, enc1.GetOverflowing().Count);

        // o1 + enc2 = [10, 2]
        CollectionAssert.AreEqual(new uint[] { 10u, 2u }, enc1.GetOverflowing()[0].GetIds());
        // o1 + o2 = [10, 20]
        CollectionAssert.AreEqual(new uint[] { 10u, 20u }, enc1.GetOverflowing()[1].GetIds());
        // enc1(orig) + o2 = [1, 20]
        CollectionAssert.AreEqual(new uint[] { 1u, 20u }, enc1.GetOverflowing()[2].GetIds());
    }

    [TestMethod]
    public void MergeWith_NoOverflowing_NoChange()
    {
        var enc1 = CreateEncoding([1], [(0, 2)]);
        var enc2 = CreateEncoding([2], [(0, 3)]);

        enc1.MergeWith(enc2, growingOffsets: false);

        Assert.AreEqual(0, enc1.GetOverflowing().Count());
    }

    [TestMethod]
    public void MergeWith_Overflowing_GrowingOffsets()
    {
        var enc1 = CreateEncoding([1], [(0, 5)]);
        var o1 = CreateEncoding([10], [(0, 3)]);
        enc1.GetOverflowing().Add(o1);

        var enc2 = CreateEncoding([2], [(0, 4)]);

        enc1.MergeWith(enc2, growingOffsets: true);

        // overflow o1+enc2: o1 ends at 3, so enc2 offsets shift by 3
        var overflow = enc1.GetOverflowing()[0];
        CollectionAssert.AreEqual(new uint[] { 10u, 2u }, overflow.GetIds());
        Assert.AreEqual((0, 3), overflow.GetOffsets()[0]);  // o1 unchanged
        Assert.AreEqual((0 + 3, 4 + 3), overflow.GetOffsets()[1]);  // enc2 shifted by 3
    }

    [TestMethod]
    public void Clone_CreatesDeepCopy()
    {
        var enc = CreateEncoding([1, 2], [(0, 3), (4, 7)]);
        enc.GetOverflowing().Add(CreateEncoding([99], [(0, 1)]));

        var clone = enc.Clone();

        Assert.AreEqual(enc.Length, clone.Length);
        CollectionAssert.AreEqual(enc.GetIds(), clone.GetIds());
        Assert.AreEqual(1, clone.GetOverflowing().Count());

        // Modify clone shouldn't affect original
        clone.GetIds()[0] = 999;
        Assert.AreEqual(1u, enc.GetIds()[0]);
    }
}
