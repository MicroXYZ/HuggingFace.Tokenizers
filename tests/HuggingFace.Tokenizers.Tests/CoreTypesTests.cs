using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Tests for core data structures.
/// </summary>
    [TestClass]
public class CoreTypesTests
{
    [TestMethod]
    public void NormalizedString_TracksAlignments()
    {
        var ns = new NormalizedString("hello world");
        ns.Lowercase();
        Assert.AreEqual("hello world", ns.Get());
    }

    [TestMethod]
    public void NormalizedString_StripRemovesWhitespace()
    {
        var ns = new NormalizedString("  hello  ");
        ns.Strip();
        Assert.AreEqual("hello", ns.Get());
    }

    [TestMethod]
    public void NormalizedString_NfcNormalization()
    {
        var ns = new NormalizedString("test");
        ns.Nfc();
        Assert.AreEqual("test", ns.Get());
    }

    [TestMethod]
    public void NormalizedString_ReplacePattern()
    {
        var ns = new NormalizedString("hello world");
        int count = ns.Replace("world", "there");
        Assert.AreEqual("hello there", ns.Get());
        Assert.AreEqual(1, count);
    }

    [TestMethod]
    public void PreTokenizedString_SplitAndTokenize()
    {
        var pts = new PreTokenizedString("hello world");
        pts.Split((i, ns) =>
        {
            var str = ns.Get();
            var parts = str.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            return parts.Select(p => new NormalizedString(p)).ToArray();
        });

        Assert.AreEqual(2, pts.GetSplits().Count);
    }

    [TestMethod]
    public void Encoding_Merge_CombinesTwoEncodings()
    {
        var e1 = new Encoding(
            [1, 2], [0, 0], ["a", "b"], [0, 1],
            [(0, 1), (1, 2)], [0, 0], [1, 1]);
        var e2 = new Encoding(
            [3, 4], [1, 1], ["c", "d"], [0, 1],
            [(0, 1), (1, 2)], [0, 0], [1, 1]);

        var merged = Encoding.Merge([e1, e2], false);

        Assert.AreEqual(4, merged.Length);
        CollectionAssert.AreEqual(new uint[] { 1, 2, 3, 4 }, merged.GetIds());
    }

    [TestMethod]
    public void Encoding_Empty_HasZeroLength()
    {
        var empty = Encoding.Empty;
        Assert.IsTrue(empty.IsEmpty);
        Assert.AreEqual(0, empty.Length);
    }

    [TestMethod]
    public void Token_HasCorrectOffsets()
    {
        var token = new Token(42, "hello", 0, 5);
        Assert.AreEqual(42u, token.Id);
        Assert.AreEqual("hello", token.Value);
        Assert.AreEqual((0, 5), token.Offsets);
    }

    [TestMethod]
    public void AddedToken_Equality()
    {
        var t1 = new AddedToken("[CLS]", isSpecial: true);
        var t2 = new AddedToken("[CLS]", isSpecial: false);
        Assert.AreEqual(t1, t2);  // Equality based on content
    }
}
