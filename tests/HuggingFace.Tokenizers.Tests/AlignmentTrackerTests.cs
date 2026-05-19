using System.Runtime.InteropServices;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Internal;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// AlignmentTracker 直接单元测试。
/// 覆盖构造、Transform、TransformRange、ExpandAlignments、ReplaceAlignments。
/// </summary>
[TestClass]
public class AlignmentTrackerTests
{
    [TestMethod]
    public void Constructor_AsciiString_InitializesCorrectly()
    {
        var tracker = new AlignmentTracker("hello");

        Assert.AreEqual(5, tracker.NormalizedUtf8Length);
        Assert.AreEqual(5, tracker.OriginalUtf8Length);
        Assert.AreEqual(5, tracker.Alignments.Count);
        Assert.AreEqual(0, tracker.OriginalShift);
    }

    [TestMethod]
    public void Constructor_EmptyString_InitializesWithZeroLength()
    {
        var tracker = new AlignmentTracker("");

        Assert.AreEqual(0, tracker.NormalizedUtf8Length);
        Assert.AreEqual(0, tracker.OriginalUtf8Length);
        Assert.AreEqual(0, tracker.Alignments.Count);
    }

    [TestMethod]
    public void Constructor_UnicodeString_CalculatesUtf8LengthCorrectly()
    {
        // "你好" = 2 CJK chars, each 3 bytes in UTF-8
        var tracker = new AlignmentTracker("你好");

        Assert.AreEqual(6, tracker.NormalizedUtf8Length);
        Assert.AreEqual(6, tracker.OriginalUtf8Length);
    }

    [TestMethod]
    public void Constructor_SupplementaryPlane_CalculatesUtf8LengthCorrectly()
    {
        // "😀" = U+1F600, 4 bytes in UTF-8, 2 chars in UTF-16
        var tracker = new AlignmentTracker("😀");

        Assert.AreEqual(4, tracker.NormalizedUtf8Length);
        Assert.AreEqual(4, tracker.OriginalUtf8Length);
    }

    [TestMethod]
    public void Alignments_InitialState_MapsEachByteToItself()
    {
        var tracker = new AlignmentTracker("abc");
        var alignments = tracker.Alignments;

        Assert.AreEqual(3, alignments.Count);
        Assert.AreEqual((0, 1), alignments[0]);
        Assert.AreEqual((1, 2), alignments[1]);
        Assert.AreEqual((2, 3), alignments[2]);
    }

    [TestMethod]
    public void GetNormalizedUtf8Bytes_ReturnsCorrectBytes()
    {
        var tracker = new AlignmentTracker("AB");
        var bytes = tracker.GetNormalizedUtf8Bytes();

        Assert.AreEqual(2, bytes.Length);
        Assert.AreEqual((byte)'A', bytes[0]);
        Assert.AreEqual((byte)'B', bytes[1]);
    }

    [TestMethod]
    public void GetOriginalUtf8Bytes_ReturnsCorrectBytes()
    {
        var tracker = new AlignmentTracker("AB");
        var bytes = tracker.GetOriginalUtf8Bytes();

        Assert.AreEqual(2, bytes.Length);
        Assert.AreEqual((byte)'A', bytes[0]);
        Assert.AreEqual((byte)'B', bytes[1]);
    }

    [TestMethod]
    public void Transform_ReplaceChar_UpdatesNormalizedAndAlignments()
    {
        var tracker = new AlignmentTracker("abc");
        // Replace 'b' with 'X' (change=0 means 1:1 replacement)
        var transforms = new List<(char, int)> { ('X', 0) };
        tracker.Transform(CollectionsMarshal.AsSpan(transforms), 1);
        var result = tracker.GetNormalizedString();

        Assert.AreEqual("X", result);
        Assert.AreEqual(1, tracker.Alignments.Count);
        Assert.AreEqual((1, 2), tracker.Alignments[0]); // 'X' maps to original 'b' position
    }

    [TestMethod]
    public void Transform_InsertChar_ExpandsAlignments()
    {
        var tracker = new AlignmentTracker("ac");
        // Insert 'b' between 'a' and 'c': 'a' change=0, 'b' change=1 (insert), 'c' change=0
        var transforms = new List<(char, int)> { ('a', 0), ('b', 1), ('c', 0) };
        tracker.Transform(CollectionsMarshal.AsSpan(transforms), 0);
        var result = tracker.GetNormalizedString();

        Assert.AreEqual("abc", result);
        Assert.AreEqual(3, tracker.Alignments.Count);
    }

    [TestMethod]
    public void Transform_RemoveChar_SkipsOldChars()
    {
        var tracker = new AlignmentTracker("abc");
        // Remove 'b': 'a' change=0, 'c' change=-1 (skip 1 old char)
        var transforms = new List<(char, int)> { ('a', 0), ('c', -1) };
        tracker.Transform(CollectionsMarshal.AsSpan(transforms), 0);
        var result = tracker.GetNormalizedString();

        Assert.AreEqual("ac", result);
        Assert.AreEqual(2, tracker.Alignments.Count);
    }

    [TestMethod]
    public void Transform_WithInitialOffset_SkipsPrefix()
    {
        var tracker = new AlignmentTracker("abcd");
        // Transform starting from index 2: replace 'c' with 'X'
        var transforms = new List<(char, int)> { ('X', 0) };
        tracker.Transform(CollectionsMarshal.AsSpan(transforms), 2);
        var result = tracker.GetNormalizedString();

        Assert.AreEqual("X", result);
    }

    [TestMethod]
    public void TransformRange_NormalizedReferential_TransformsRange()
    {
        var tracker = new AlignmentTracker("hello world");
        var transforms = new List<(char, int)> { ('X', 0), ('Y', 0), ('Z', 0) };
        Func<OffsetReferential, Range, Range?> converter = (r, range) => range;

        tracker.TransformRange(
            OffsetReferential.Normalized,
            0..3,
            CollectionsMarshal.AsSpan(transforms),
            0,
            11,
            converter);

        var result = tracker.GetNormalizedString();
        // Replaces "hel" with "XYZ"
        Assert.AreEqual("XYZlo world", result);
    }

    [TestMethod]
    public void ExpandAlignments_ValidRange_ReturnsCorrectRange()
    {
        var alignments = new List<(int, int)>
        {
            (0, 1), (1, 2), (2, 3), (3, 4)
        };

        var result = AlignmentTracker.ExpandAlignments(alignments, 1, 3);

        Assert.IsNotNull(result);
        Assert.AreEqual(1, result.Value.Start.GetOffset(4));
        Assert.AreEqual(3, result.Value.End.GetOffset(4));
    }

    [TestMethod]
    public void ExpandAlignments_EmptyRange_ReturnsNull()
    {
        var alignments = new List<(int, int)> { (0, 1) };
        var result = AlignmentTracker.ExpandAlignments(alignments, 1, 1);

        Assert.IsNull(result);
    }

    [TestMethod]
    public void ReplaceAlignments_UpdatesAlignmentsAndUtf8()
    {
        var tracker = new AlignmentTracker("hello");
        var newAlignments = new List<(int, int)>
        {
            (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)
        };

        var buffer = "HELLO".ToCharArray();
        tracker.ReplaceAlignments(newAlignments, buffer, 5);
        var result = tracker.GetNormalizedString();

        Assert.AreEqual("HELLO", result);
        Assert.AreEqual(5, tracker.Alignments.Count);
        Assert.AreEqual(5, tracker.NormalizedUtf8Length);
    }

    [TestMethod]
    public void SecondConstructor_WithShift_SetsOriginalShift()
    {
        var alignments = new List<(int, int)> { (5, 6), (6, 7) };
        var tracker = new AlignmentTracker("original", "norm", alignments, 5);

        Assert.AreEqual(5, tracker.OriginalShift);
        Assert.AreEqual(2, tracker.Alignments.Count);
    }

    [TestMethod]
    public void Transform_MultipleOperations_PreservesAlignmentConsistency()
    {
        var tracker = new AlignmentTracker("abcde");
        // Replace 'b' with 'XY' (insert Y after X)
        var transforms = new List<(char, int)>
        {
            ('X', 0),  // replace 'b'
            ('Y', 1)   // insert 'Y'
        };
        tracker.Transform(CollectionsMarshal.AsSpan(transforms), 1);
        var result = tracker.GetNormalizedString();

        Assert.AreEqual("XY", result);
        Assert.AreEqual(2, tracker.Alignments.Count);
        // Both X and Y should map to 'b' area
        Assert.AreEqual((1, 2), tracker.Alignments[0]);
    }
}
