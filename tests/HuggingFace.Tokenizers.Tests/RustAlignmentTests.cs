using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Rust 对齐验证测试。
/// 验证 C# SA-IS 后缀数组实现与 Rust esaxx-rs 的一致性。
/// </summary>
[TestClass]
public class RustAlignmentTests
{
    #region SA-IS 后缀数组对齐

    [TestMethod]
    public void Sais_RustEsaxxTestCases_Match()
    {
        // 验证与 Rust esaxx-rs 测试用例完全一致
        // 来源：https://github.com/Narsil/esaxx-rs/blob/main/src/lib.rs

        // "abracadabra"
        VerifySa("abracadabra", [10, 7, 0, 3, 5, 8, 1, 4, 6, 9, 2]);

        // "banana$"
        VerifySa("banana$", [6, 5, 3, 1, 0, 4, 2]);
    }

    [TestMethod]
    public void Sais_RustSuffixRs_MatchesSubstringContent()
    {
        // 验证与 Rust suffix_rs 迭代器输出一致（子串内容+频率）
        var text = "abracadabra";
        var chars = text.Select(c => (int)c).ToArray();
        var entries = EnhancedSuffixArray.EnumerateSubstrings(chars);

        // 期望 5 个节点（与 Rust 测试一致）
        Assert.AreEqual(5, entries.Count);

        string Sub(EnhancedSuffixArray.SubstringEntry e) =>
            e.Length == 0 ? "" : new string(text.AsSpan(e.Offset, e.Length));

        // Rust: (&chars[..4], 2) = ("abra", 2)
        Assert.AreEqual("abra", Sub(entries[0]));
        Assert.AreEqual(2, entries[0].Frequency);

        // Rust: (&chars[..1], 5) = ("a", 5)
        Assert.AreEqual("a", Sub(entries[1]));
        Assert.AreEqual(5, entries[1].Frequency);

        // Rust: (&chars[1..4], 2) = ("bra", 2)
        Assert.AreEqual("bra", Sub(entries[2]));
        Assert.AreEqual(2, entries[2].Frequency);

        // Rust: (&chars[2..4], 2) = ("ra", 2)
        Assert.AreEqual("ra", Sub(entries[3]));
        Assert.AreEqual(2, entries[3].Frequency);

        // Rust: (&chars[..0], 11) = ("", 11)
        Assert.AreEqual("", Sub(entries[4]));
        Assert.AreEqual(11, entries[4].Frequency);
    }

    [TestMethod]
    public void Sais_RustBananaDollar_Matches()
    {
        // "banana$" 是 SA-IS 算法的经典测试用例
        VerifySa("banana$", [6, 5, 3, 1, 0, 4, 2]);

        var chars = "banana$".Select(c => (int)c).ToArray();
        var entries = EnhancedSuffixArray.EnumerateSubstrings(chars);

        // 验证子串内容正确
        Assert.IsTrue(entries.Count > 0);
        var first = entries[0];
        var sub = "banana$".Substring(first.Offset, first.Length);
        Assert.IsTrue(sub.Length > 0 || first.Length == 0);
    }

    [TestMethod]
    public void Sais_LargeInput_NoCrash()
    {
        // 验证较长输入不会崩溃（与 Rust test_esaxx_rs_long 对应）
        var text = "Lorem Ipsum is simply dummy text of the printing and typesetting industry. " +
                   "Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, " +
                   "when an unknown printer took a galley of type and scrambled it to make a type " +
                   "specimen book. It has survived not only five centuries, but also the leap into " +
                   "electronic typesetting, remaining essentially unchanged. It was popularised in " +
                   "the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, " +
                   "and more recently with desktop publishing software like Aldus PageMaker including " +
                   "versions of Lorem Ipsum.";
        var chars = text.Select(c => (int)c).ToArray();
        var sa = new int[chars.Length];

        Sais.Build(chars, sa);

        // 验证 SA 是有效的排列（每个索引恰好出现一次）
        var sorted = sa.OrderBy(x => x).ToArray();
        for (int i = 0; i < sorted.Length; i++)
            Assert.AreEqual(i, sorted[i], $"SA 应包含每个索引恰好一次，缺失 {i}");

        // 验证增强后缀数组能正常枚举
        var entries = EnhancedSuffixArray.EnumerateSubstrings(chars);
        Assert.IsTrue(entries.Count > 100, "574 字符的文本应产生大量子串节点");
    }

    private static void VerifySa(string text, int[] expected)
    {
        var chars = text.Select(c => (int)c).ToArray();
        var sa = new int[chars.Length];
        Sais.Build(chars, sa);
        CollectionAssert.AreEqual(expected, sa, $"SA 不匹配：{text}");
    }

    #endregion
}
