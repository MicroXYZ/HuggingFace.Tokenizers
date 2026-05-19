using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// StringTransforms 直接单元测试。
/// 覆盖 BuildFilter、BuildStrip、BuildPrepend、BuildAppend、BuildMap。
/// </summary>
[TestClass]
public class StringTransformsTests
{
    // ── BuildFilter ──

    [TestMethod]
    public void BuildFilter_FilterNone_ReturnsAllChars()
    {
        var (transforms, initialOffset) = StringTransforms.BuildFilter("abc", _ => true);
        Assert.AreEqual(0, initialOffset);
        Assert.AreEqual(3, transforms.Count);
    }

    [TestMethod]
    public void BuildFilter_FilterAll_ReturnsEmptyWithOffset()
    {
        // When all chars are filtered, no char is ever kept, so removedStart stays 0
        var (transforms, initialOffset) = StringTransforms.BuildFilter("abc", _ => false);
        Assert.AreEqual(0, initialOffset);
        Assert.AreEqual(0, transforms.Count);
    }

    [TestMethod]
    public void BuildFilter_FilterMiddle_RemovesMiddleChars()
    {
        // "abc" filtering out 'b': 'a' sets removedStart=0, 'b' increments removed=1
        // 'c' triggers result.Add(('a', -1)), removed resets to 0, lastC='c'
        // End: result.Add(('c', 0)) — no pending removal after last kept char
        var (transforms, initialOffset) = StringTransforms.BuildFilter("abc", r => r.Value != 'b');
        Assert.AreEqual(0, initialOffset);
        Assert.AreEqual(2, transforms.Count);
        Assert.AreEqual(('a', -1), transforms[0]); // carries pending removal of 'b'
        Assert.AreEqual(('c', 0), transforms[1]);  // no pending removal after last char
    }

    [TestMethod]
    public void BuildFilter_FilterLeading_RemovesLeadingChars()
    {
        // Keep 'bc', filter 'a'
        var (transforms, initialOffset) = StringTransforms.BuildFilter("abc", r => r.Value != 'a');
        Assert.AreEqual(1, initialOffset); // 'a' removed from start
        Assert.AreEqual(2, transforms.Count);
    }

    [TestMethod]
    public void BuildFilter_SupplementaryPlane_HandlesCorrectly()
    {
        // Keep emoji and 'y', filter 'x' (1 char removed from start)
        var (transforms, initialOffset) = StringTransforms.BuildFilter("x😀y", r => r.Value != 'x');
        Assert.AreEqual(1, initialOffset); // 'x' removed from start
        Assert.IsTrue(transforms.Count > 0);
    }

    // ── BuildStrip ──

    [TestMethod]
    public void BuildStrip_BothSides_RemovesWhitespace()
    {
        var (transforms, initialOffset) = StringTransforms.BuildStrip("  hello  ", left: true, right: true);
        Assert.AreEqual(2, initialOffset);
        Assert.AreEqual(5, transforms.Count);
    }

    [TestMethod]
    public void BuildStrip_LeftOnly_RemovesLeadingWhitespace()
    {
        var (transforms, initialOffset) = StringTransforms.BuildStrip("  hello  ", left: true, right: false);
        Assert.AreEqual(2, initialOffset);
        Assert.AreEqual(7, transforms.Count); // "hello  "
    }

    [TestMethod]
    public void BuildStrip_RightOnly_RemovesTrailingWhitespace()
    {
        var (transforms, initialOffset) = StringTransforms.BuildStrip("  hello  ", left: false, right: true);
        Assert.AreEqual(0, initialOffset);
        Assert.AreEqual(7, transforms.Count); // "  hello"
    }

    [TestMethod]
    public void BuildStrip_NoWhitespace_ReturnsEmpty()
    {
        var (transforms, initialOffset) = StringTransforms.BuildStrip("hello", left: true, right: true);
        Assert.AreEqual(0, initialOffset);
        Assert.AreEqual(0, transforms.Count);
    }

    [TestMethod]
    public void BuildStrip_EmptyString_ReturnsEmpty()
    {
        var (transforms, initialOffset) = StringTransforms.BuildStrip("", left: true, right: true);
        Assert.AreEqual(0, initialOffset);
        Assert.AreEqual(0, transforms.Count);
    }

    [TestMethod]
    public void BuildStrip_AllWhitespace_ReturnsEmpty()
    {
        var (transforms, initialOffset) = StringTransforms.BuildStrip("   ", left: true, right: true);
        Assert.AreEqual(3, initialOffset);
        Assert.AreEqual(0, transforms.Count);
    }

    // ── BuildPrepend ──

    [TestMethod]
    public void BuildPrepend_SingleChar_ReturnsCorrectTransforms()
    {
        var transforms = StringTransforms.BuildPrepend("▁", "h");
        // ▁ change=0 (replace first), h change=1 (insert)
        Assert.AreEqual(2, transforms.Count);
        Assert.AreEqual(('▁', 0), transforms[0]);
        Assert.AreEqual(('h', 1), transforms[1]);
    }

    [TestMethod]
    public void BuildPrepend_MultiCharContent_FirstCharReplacesRestInsert()
    {
        var transforms = StringTransforms.BuildPrepend("##", "h");
        Assert.AreEqual(3, transforms.Count);
        Assert.AreEqual(('#', 0), transforms[0]); // first char replaces
        Assert.AreEqual(('#', 1), transforms[1]); // second char inserts
        Assert.AreEqual(('h', 1), transforms[2]); // original first char inserts
    }

    // ── BuildAppend ──

    [TestMethod]
    public void BuildAppend_SingleChar_ReturnsCorrectTransforms()
    {
        var transforms = StringTransforms.BuildAppend("!", "o");
        // 'o' change=0 (replace last), '!' change=1 (insert)
        Assert.AreEqual(2, transforms.Count);
        Assert.AreEqual(('o', 0), transforms[0]);
        Assert.AreEqual(('!', 1), transforms[1]);
    }

    [TestMethod]
    public void BuildAppend_MultiCharContent_LastCharReplacesRestInsert()
    {
        var transforms = StringTransforms.BuildAppend("##", "o");
        Assert.AreEqual(3, transforms.Count);
        Assert.AreEqual(('o', 0), transforms[0]);  // last char replaces
        Assert.AreEqual(('#', 1), transforms[1]);   // first append char inserts
        Assert.AreEqual(('#', 1), transforms[2]);   // second append char inserts
    }

    // ── BuildMap ──

    [TestMethod]
    public void BuildMap_Identity_ReturnsUnchangedChars()
    {
        var transforms = StringTransforms.BuildMap("abc", r => r);
        Assert.AreEqual(3, transforms.Count);
        Assert.AreEqual(('a', 0), transforms[0]);
        Assert.AreEqual(('b', 0), transforms[1]);
        Assert.AreEqual(('c', 0), transforms[2]);
    }

    [TestMethod]
    public void BuildMap_Uppercase_ReturnsMappedChars()
    {
        var transforms = StringTransforms.BuildMap("abc", r => Rune.ToUpperInvariant(r));
        Assert.AreEqual(3, transforms.Count);
        Assert.AreEqual(('A', 0), transforms[0]);
        Assert.AreEqual(('B', 0), transforms[1]);
        Assert.AreEqual(('C', 0), transforms[2]);
    }

    [TestMethod]
    public void BuildMap_ShrinkingMapping_HandlesCorrectly()
    {
        // Map 2-char rune to 1-char rune: supplementary plane to ASCII
        // '😀' (2 UTF-16 chars) → 'X' (1 char)
        var transforms = StringTransforms.BuildMap("😀", _ => new Rune('X'));
        Assert.AreEqual(1, transforms.Count);
        Assert.AreEqual(('X', -1), transforms[0]); // -1 because consumed 1 extra old char
    }

    [TestMethod]
    public void BuildMap_EmptyString_ReturnsEmpty()
    {
        var transforms = StringTransforms.BuildMap("", r => r);
        Assert.AreEqual(0, transforms.Count);
    }
}
