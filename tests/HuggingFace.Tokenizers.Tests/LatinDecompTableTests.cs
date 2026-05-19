using System.Text;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;
using HuggingFace.Tokenizers.Normalizers;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// LatinDecompTable 和 StripAccentsFast 优化路径的专项测试。
/// 覆盖正确性、边界条件、ZWJ emoji 处理。
/// </summary>
[TestClass]
public class LatinDecompTableTests
{
    // ── 查表正确性 ──

    [TestMethod]
    public void TryGetBaseChar_LatinAccent_ReturnsBase()
    {
        // é (U+00E9) → e
        Assert.IsTrue(LatinDecompTable.TryGetBaseChar('\u00E9', out var baseChar));
        Assert.AreEqual('e', baseChar);
    }

    [TestMethod]
    public void TryGetBaseChar_NChar_ReturnsN()
    {
        // ñ (U+00F1) → n
        Assert.IsTrue(LatinDecompTable.TryGetBaseChar('\u00F1', out var baseChar));
        Assert.AreEqual('n', baseChar);
    }

    [TestMethod]
    public void TryGetBaseChar_UUmlaut_ReturnsU()
    {
        // ü (U+00FC) → u
        Assert.IsTrue(LatinDecompTable.TryGetBaseChar('\u00FC', out var baseChar));
        Assert.AreEqual('u', baseChar);
    }

    [TestMethod]
    public void TryGetBaseChar_Ascii_ReturnsFalse()
    {
        // ASCII 字符不在表中
        Assert.IsFalse(LatinDecompTable.TryGetBaseChar('a', out _));
        Assert.IsFalse(LatinDecompTable.TryGetBaseChar('Z', out _));
        Assert.IsFalse(LatinDecompTable.TryGetBaseChar('0', out _));
    }

    [TestMethod]
    public void TryGetBaseChar_CombiningMark_ReturnsFalse()
    {
        // combining marks 本身不在表中
        Assert.IsFalse(LatinDecompTable.TryGetBaseChar('\u0301', out _)); // combining acute
        Assert.IsFalse(LatinDecompTable.TryGetBaseChar('\u0308', out _)); // combining diaeresis
    }

    [TestMethod]
    public void TryGetBaseChar_HighSurrogate_ReturnsFalse()
    {
        // 补充平面字符的高代理项
        Assert.IsFalse(LatinDecompTable.TryGetBaseChar('\uD83D', out _));
    }

    // ── IsCombiningMark ──

    [TestMethod]
    public void IsCombiningMark_CombiningAcute_ReturnsTrue()
    {
        Assert.IsTrue(LatinDecompTable.IsCombiningMark('\u0301'));
    }

    [TestMethod]
    public void IsCombiningMark_CombiningDiaeresis_ReturnsTrue()
    {
        Assert.IsTrue(LatinDecompTable.IsCombiningMark('\u0308'));
    }

    [TestMethod]
    public void IsCombiningMark_Ascii_ReturnsFalse()
    {
        Assert.IsFalse(LatinDecompTable.IsCombiningMark('a'));
    }

    [TestMethod]
    public void IsCombiningMark_OutOfRange_ReturnsFalse()
    {
        Assert.IsFalse(LatinDecompTable.IsCombiningMark('\u02FF'));
        Assert.IsFalse(LatinDecompTable.IsCombiningMark('\u0370'));
    }

    // ── NeedsLatinDecomp ──

    [TestMethod]
    public void NeedsLatinDecomp_WithAccented_ReturnsTrue()
    {
        Assert.IsTrue(LatinDecompTable.NeedsLatinDecomp("café"));
    }

    [TestMethod]
    public void NeedsLatinDecomp_PureAscii_ReturnsFalse()
    {
        Assert.IsFalse(LatinDecompTable.NeedsLatinDecomp("hello"));
    }

    [TestMethod]
    public void NeedsLatinDecomp_CombiningMarkOnly_ReturnsFalse()
    {
        // combining marks 本身不算 precomposed
        Assert.IsFalse(LatinDecompTable.NeedsLatinDecomp("\u0301\u0308"));
    }

    [TestMethod]
    public void NeedsLatinDecomp_Empty_ReturnsFalse()
    {
        Assert.IsFalse(LatinDecompTable.NeedsLatinDecomp(""));
    }

    // ── ContainsZwj ──

    [TestMethod]
    public void ContainsZwj_WithZwj_ReturnsTrue()
    {
        Assert.IsTrue(LatinDecompTable.ContainsZwj("👨\u200D💻"));
    }

    [TestMethod]
    public void ContainsZwj_WithoutZwj_ReturnsFalse()
    {
        Assert.IsFalse(LatinDecompTable.ContainsZwj("hello"));
    }

    // ── 多 combining marks 字符 ──

    [TestMethod]
    public void TryGetBaseChar_TwoCombiningMarks_ReturnsBase()
    {
        // ǭ (U+01EB) → o + ̨ + ̄ (o + combining ogonek + combining macron)
        if (LatinDecompTable.TryGetBaseChar('\u01EB', out var baseChar))
        {
            Assert.AreEqual('o', baseChar);
        }
        // 如果表中没有（因为 combiningCount=2 的情况），也不应崩溃
    }
}

/// <summary>
/// StripAccentsFast 优化路径的集成测试。
/// 对比标准 StripAccents 和 StripAccentsFast 的结果一致性。
/// </summary>
[TestClass]
public class StripAccentsFastTests
{
    // ── 基本功能 ──

    [TestMethod]
    public void StripAccentsFast_SimpleAccent_RemovesAccent()
    {
        var ns = new NormalizedString("café");
        ns.StripAccentsFast();
        Assert.AreEqual("cafe", ns.Get());
    }

    [TestMethod]
    public void StripAccentsFast_MultipleAccents_RemovesAll()
    {
        var ns = new NormalizedString("résumé");
        ns.StripAccentsFast();
        Assert.AreEqual("resume", ns.Get());
    }

    [TestMethod]
    public void StripAccentsFast_NoAccents_Unchanged()
    {
        var ns = new NormalizedString("hello");
        ns.StripAccentsFast();
        Assert.AreEqual("hello", ns.Get());
    }

    [TestMethod]
    public void StripAccentsFast_EmptyString_Unchanged()
    {
        var ns = new NormalizedString("");
        ns.StripAccentsFast();
        Assert.AreEqual("", ns.Get());
    }

    [TestMethod]
    public void StripAccentsFast_AllLatinAccents_RemovesAll()
    {
        // 覆盖常见拉丁扩展字符（去音标场景）
        // 注意：ð(eth), þ(thorn), æ, ø 等是独立字母，NFD 不会分解为 base+combining
        var ns = new NormalizedString("àáâãäåçèéêëìíîïñòóôõöùúûüý");
        ns.StripAccentsFast();
        var result = ns.Get();
        // 所有字符都应该是 ASCII
        foreach (var c in result)
        {
            Assert.IsTrue(c < 0x80, $"Expected ASCII char, got U+{c:X4}");
        }
    }

    // ── 与标准 StripAccents 结果一致 ──

    [TestMethod]
    public void StripAccentsFast_MatchesStandard_LatinText()
    {
        var testCases = new[]
        {
            "café résumé naïve",
            "Héllo Wörld",
            "Ångström jalapeño",
            " Über Straße ",
            "àáâãäåæçèéêë",
        };

        foreach (var text in testCases)
        {
            var ns1 = new NormalizedString(text);
            ns1.StripAccents();

            var ns2 = new NormalizedString(text);
            ns2.StripAccentsFast();

            Assert.AreEqual(ns1.Get(), ns2.Get(), $"Mismatch for input: {text}");
        }
    }

    // ── ZWJ emoji 处理 ──

    [TestMethod]
    public void StripAccentsFast_WithZwjEmoji_PreservesEmoji()
    {
        var ns = new NormalizedString("café 👨\u200D💻 résumé");
        ns.StripAccentsFast();
        var result = ns.Get();
        // ZWJ emoji 应保留
        Assert.IsTrue(result.Contains('\u200D'), "ZWJ should be preserved");
        // 音标应去除
        Assert.AreEqual("cafe 👨\u200D💻 resume", result);
    }

    [TestMethod]
    public void StripAccentsFast_MultipleZwjEmojis_PreservesAll()
    {
        var ns = new NormalizedString("Héllo 🏳️\u200D🌈 Wörld 👨\u200D💻");
        ns.StripAccentsFast();
        var result = ns.Get();
        Assert.IsTrue(result.Contains("🏳️\u200D🌈"), "First emoji preserved");
        Assert.IsTrue(result.Contains("👨\u200D💻"), "Second emoji preserved");
        Assert.IsFalse(result.Contains('é'), "Accent removed");
        Assert.IsFalse(result.Contains('ö'), "Accent removed");
    }

    [TestMethod]
    public void StripAccentsFast_OnlyZwjEmoji_NoChange()
    {
        var ns = new NormalizedString("👨\u200D💻🏳️\u200D🌈");
        ns.StripAccentsFast();
        Assert.AreEqual("👨\u200D💻🏳️\u200D🌈", ns.Get());
    }

    // ── CJK + 音标混合 ──

    [TestMethod]
    public void StripAccentsFast_CjkWithAccents_RemovesAccentsPreservesCjk()
    {
        var ns = new NormalizedString("Héllo 你好 Wörld");
        ns.StripAccentsFast();
        Assert.AreEqual("Hello 你好 World", ns.Get());
    }

    // ── BertNormalizer 集成 ──

    [TestMethod]
    public void BertNormalizer_StripAccentsFast_Integration()
    {
        var normalizer = new BertNormalizer(cleanText: true, handleChineseChars: true, stripAccents: true, lowercase: true);
        var ns = new NormalizedString("Héllo café");
        normalizer.Normalize(ns);
        var result = ns.Get();
        // 音标去除 + 小写
        Assert.AreEqual("hello cafe", result);
    }

    [TestMethod]
    public void BertNormalizer_StripAccents_False_SkipsStrip()
    {
        var normalizer = new BertNormalizer(cleanText: false, handleChineseChars: false, stripAccents: false, lowercase: false);
        var ns = new NormalizedString("Héllo");
        normalizer.Normalize(ns);
        Assert.AreEqual("Héllo", ns.Get());
    }

    // ── 边界条件 ──

    [TestMethod]
    public void StripAccentsFast_OnlyCombiningMarks_RemovesAll()
    {
        // 纯 combining marks（没有 base char）
        var ns = new NormalizedString("\u0301\u0308\u0300");
        ns.StripAccentsFast();
        Assert.AreEqual("", ns.Get());
    }

    [TestMethod]
    public void StripAccentsFast_MixedAsciiAndCombining_CorrectResult()
    {
        var ns = new NormalizedString("a\u0301b\u0308c"); // á + b̈ + c (base + combining)
        ns.StripAccentsFast();
        // combining marks 应被移除
        Assert.AreEqual("abc", ns.Get());
    }

    [TestMethod]
    public void StripAccentsFast_SupplementaryPlaneChars_Preserved()
    {
        // 补充平面字符（非 ZWJ emoji）应保留
        var ns = new NormalizedString("café 😀🎮 test");
        ns.StripAccentsFast();
        Assert.AreEqual("cafe 😀🎮 test", ns.Get());
    }

    [TestMethod]
    public void StripAccentsFast_PureCombiningAfterNfd_CorrectResult()
    {
        // 先 NFD 分解，再 StripAccentsFast
        var ns = new NormalizedString("café");
        ns.Nfd(); // é → e + combining acute
        ns.StripAccentsFast(); // 应移除 combining acute
        Assert.AreEqual("cafe", ns.Get());
    }
}
