using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// UnicodeNormalizer 直接单元测试。
/// 覆盖 NFD/NFC/NFKD/NFKC 规范化变换构建。
/// </summary>
[TestClass]
public class UnicodeNormalizerTests
{
    // ── NFD 分解 ──

    [TestMethod]
    public void BuildNormalizationTransform_NFD_AsciiUnchanged()
    {
        // ASCII 字符无需分解
        var transforms = UnicodeNormalizer.BuildNormalizationTransform("abc", "abc", NormalizationForm.FormD);
        Assert.AreEqual(3, transforms.Count);
        Assert.AreEqual(('a', 0), transforms[0]);
        Assert.AreEqual(('b', 0), transforms[1]);
        Assert.AreEqual(('c', 0), transforms[2]);
    }

    [TestMethod]
    public void BuildNormalizationTransform_NFD_AccentDecomposes()
    {
        // 'é' (U+00E9) → 'e' + combining acute (U+0301)
        var input = "é";
        var expected = input.Normalize(NormalizationForm.FormD);
        var transforms = UnicodeNormalizer.BuildNormalizationTransform(input, expected, NormalizationForm.FormD);

        Assert.AreEqual(2, transforms.Count);
        Assert.AreEqual(('e', 0), transforms[0]);      // base char replaces
        Assert.AreEqual(('\u0301', 1), transforms[1]);  // combining mark inserts
    }

    [TestMethod]
    public void BuildNormalizationTransform_NFD_EmptyString_ReturnsEmpty()
    {
        var transforms = UnicodeNormalizer.BuildNormalizationTransform("", "", NormalizationForm.FormD);
        Assert.AreEqual(0, transforms.Count);
    }

    [TestMethod]
    public void BuildNormalizationTransform_NFD_MultipleAccents()
    {
        // 'ệ' → 'e' + combining circumflex + combining dot below
        var input = "ệ";
        var expected = input.Normalize(NormalizationForm.FormD);
        var transforms = UnicodeNormalizer.BuildNormalizationTransform(input, expected, NormalizationForm.FormD);

        // Should decompose into multiple characters
        Assert.IsTrue(transforms.Count >= 2);
        Assert.AreEqual(0, transforms[0].Change); // first char always change=0
    }

    // ── NFKD 分解 ──

    [TestMethod]
    public void BuildNormalizationTransform_NFKD_CompatibilityDecomposes()
    {
        // 'ﬁ' (U+FB01, fi ligature) → 'f' + 'i'
        var input = "ﬁ";
        var expected = input.Normalize(NormalizationForm.FormKD);
        var transforms = UnicodeNormalizer.BuildNormalizationTransform(input, expected, NormalizationForm.FormKD);

        Assert.AreEqual(2, transforms.Count);
        Assert.AreEqual(('f', 0), transforms[0]);
        Assert.AreEqual(('i', 1), transforms[1]);
    }

    // ── NFC 组合 ──

    [TestMethod]
    public void BuildNormalizationTransform_NFC_AsciiUnchanged()
    {
        var transforms = UnicodeNormalizer.BuildNormalizationTransform("abc", "abc", NormalizationForm.FormC);
        Assert.AreEqual(3, transforms.Count);
        Assert.AreEqual(('a', 0), transforms[0]);
        Assert.AreEqual(('b', 0), transforms[1]);
        Assert.AreEqual(('c', 0), transforms[2]);
    }

    [TestMethod]
    public void BuildNormalizationTransform_NFC_CombinesAccents()
    {
        // 'e' + combining acute → 'é'
        var input = "e\u0301";
        var expected = input.Normalize(NormalizationForm.FormC);
        var transforms = UnicodeNormalizer.BuildNormalizationTransform(input, expected, NormalizationForm.FormC);

        // Should have fewer output chars than input
        Assert.IsTrue(transforms.Count <= input.Length);
    }

    [TestMethod]
    public void BuildNormalizationTransform_NFC_AlreadyComposed_Unchanged()
    {
        // 'é' is already NFC, but composition via NFD intermediate:
        // NFD: 'e' + '\u0301' (2 runes), NFC: 'é' (1 rune)
        // So change = -(2-1) = -1
        var input = "é";
        var expected = input.Normalize(NormalizationForm.FormC);
        var transforms = UnicodeNormalizer.BuildNormalizationTransform(input, expected, NormalizationForm.FormC);

        Assert.AreEqual(1, transforms.Count);
        Assert.AreEqual(('é', -1), transforms[0]);
    }

    // ── NFKC 组合 ──

    [TestMethod]
    public void BuildNormalizationTransform_NFKC_CombinesCompatibility()
    {
        // 'ﬁ' → NFKD → 'f' + 'i', NFKC → 'fi'
        // Composition via NFKD intermediate: 'f' consumed=1 change=0, 'i' consumed=1 change=0
        var input = "ﬁ";
        var expected = input.Normalize(NormalizationForm.FormKC);
        var transforms = UnicodeNormalizer.BuildNormalizationTransform(input, expected, NormalizationForm.FormKC);

        Assert.AreEqual(2, transforms.Count);
        Assert.AreEqual(('f', 0), transforms[0]);
        Assert.AreEqual(('i', 0), transforms[1]);
    }

    // ── 通用 ──

    [TestMethod]
    public void BuildNormalizationTransform_NFD_ChangeValuesCorrect()
    {
        // Verify change semantics: 0=replace, 1=insert, negative=replace+remove
        var input = "é"; // decomposes to 'e' + '\u0301'
        var expected = input.Normalize(NormalizationForm.FormD);
        var transforms = UnicodeNormalizer.BuildNormalizationTransform(input, expected, NormalizationForm.FormD);

        foreach (var (c, change) in transforms)
        {
            Assert.IsTrue(change >= -1, $"Change value {change} should be >= -1");
        }
    }

    [TestMethod]
    public void BuildNormalizationTransform_AllForms_HandleEmptyString()
    {
        var forms = new[]
        {
            NormalizationForm.FormC,
            NormalizationForm.FormD,
            NormalizationForm.FormKC,
            NormalizationForm.FormKD
        };

        foreach (var form in forms)
        {
            var transforms = UnicodeNormalizer.BuildNormalizationTransform("", "", form);
            Assert.AreEqual(0, transforms.Count, $"Form {form} should return empty for empty input");
        }
    }

    [TestMethod]
    public void BuildNormalizationTransform_NFD_Japanese_CorrectDecomposition()
    {
        // Japanese hiragana: 'が' (ga) → 'か' + combining voiced mark
        var input = "が";
        var expected = input.Normalize(NormalizationForm.FormD);
        var transforms = UnicodeNormalizer.BuildNormalizationTransform(input, expected, NormalizationForm.FormD);

        Assert.AreEqual(2, transforms.Count);
        Assert.AreEqual(0, transforms[0].Change);
        Assert.AreEqual(1, transforms[1].Change);
    }

    [TestMethod]
    public void BuildNormalizationTransform_NFC_RoundTrip_PreservesLength()
    {
        var input = "Hello World 你好世界";
        var nfc = input.Normalize(NormalizationForm.FormC);
        var transforms = UnicodeNormalizer.BuildNormalizationTransform(input, nfc, NormalizationForm.FormC);

        // For already-normalized input, transform count should equal input length
        Assert.AreEqual(input.Length, transforms.Count);
    }
}
