using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// SA-IS 后缀数组和增强后缀数组测试。
/// 验证与 Rust esaxx-rs 的输出一致性。
/// </summary>
[TestClass]
public class SuffixArrayTests
{
    [TestMethod]
    public void Sais_Abracadabra_CorrectSuffixArray()
    {
        // 与 Rust esaxx-rs 测试用例一致
        var chars = "abracadabra".Select(c => (int)c).ToArray();
        var sa = new int[chars.Length];

        Sais.Build(chars, sa);

        // 期望 SA: [10, 7, 0, 3, 5, 8, 1, 4, 6, 9, 2]
        CollectionAssert.AreEqual(new[] { 10, 7, 0, 3, 5, 8, 1, 4, 6, 9, 2 }, sa);
    }

    [TestMethod]
    public void EnhancedSuffixArray_Abracadabra_CorrectSubstrings()
    {
        // 与 Rust suffix_rs 迭代器输出一致
        var text = "abracadabra";
        var chars = text.Select(c => (int)c).ToArray();

        var entries = EnhancedSuffixArray.EnumerateSubstrings(chars);

        // 期望 5 个节点（与 Rust suffix_rs 迭代器输出一致）：
        // ("abra", freq=2), ("a", freq=5), ("bra", freq=2), ("ra", freq=2), ("", freq=11)
        Assert.AreEqual(5, entries.Count);

        // 通过 offset + length 提取子串内容进行验证（与 Rust 的 slice 比较方式一致）
        string Sub(EnhancedSuffixArray.SubstringEntry e) =>
            e.Length == 0 ? "" : new string(text.AsSpan(e.Offset, e.Length));

        Assert.AreEqual("abra", Sub(entries[0]));
        Assert.AreEqual(2, entries[0].Frequency);

        Assert.AreEqual("a", Sub(entries[1]));
        Assert.AreEqual(5, entries[1].Frequency);

        Assert.AreEqual("bra", Sub(entries[2]));
        Assert.AreEqual(2, entries[2].Frequency);

        Assert.AreEqual("ra", Sub(entries[3]));
        Assert.AreEqual(2, entries[3].Frequency);

        Assert.AreEqual("", Sub(entries[4]));
        Assert.AreEqual(11, entries[4].Frequency);
    }

    [TestMethod]
    public void Sais_SingleCharacter_Correct()
    {
        var chars = new[] { (int)'a' };
        var sa = new int[1];
        Sais.Build(chars, sa);
        Assert.AreEqual(0, sa[0]);
    }

    [TestMethod]
    public void Sais_RepeatedCharacters_Correct()
    {
        // "aaaa" 的 SA 应为 [3, 2, 1, 0]
        var chars = "aaaa".Select(c => (int)c).ToArray();
        var sa = new int[4];
        Sais.Build(chars, sa);
        CollectionAssert.AreEqual(new[] { 3, 2, 1, 0 }, sa);
    }

    [TestMethod]
    public void Sais_BananaDollar_Correct()
    {
        // "banana$" 的 SA 应为 [6, 5, 3, 1, 0, 4, 2]
        var chars = "banana$".Select(c => (int)c).ToArray();
        var sa = new int[7];
        Sais.Build(chars, sa);
        CollectionAssert.AreEqual(new[] { 6, 5, 3, 1, 0, 4, 2 }, sa);
    }

    [TestMethod]
    public void EnhancedSuffixArray_SupplementaryPlane_Correct()
    {
        // 包含 emoji（补充平面字符）的字符串
        var text = "a🎉a🎉";
        var chars = text.EnumerateRunes().Select(r => r.Value).ToArray();

        var entries = EnhancedSuffixArray.EnumerateSubstrings(chars);

        // 应该能正常处理，不崩溃
        Assert.IsTrue(entries.Count > 0, "补充平面字符应产生有效子串");
    }
}
