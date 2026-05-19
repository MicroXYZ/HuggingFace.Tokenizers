using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using System.Diagnostics.CodeAnalysis;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// AhoCorasickMatcher 直接单元测试。
/// 覆盖：单模式、多模式、重叠匹配、空输入、Unicode、LeftmostLongest 语义。
/// </summary>
[TestClass]
public class AhoCorasickMatcherTests
{
    // 通过反射访问 internal 类
    [UnconditionalSuppressMessage("Trimming", "IL2026", Justification = "测试代码，类型在程序集中确定存在")]
    [UnconditionalSuppressMessage("Trimming", "IL2072", Justification = "测试代码，返回类型有公共构造函数")]
    [UnconditionalSuppressMessage("Trimming", "IL2075", Justification = "测试代码，方法在运行时确定存在")]
    private static object CreateMatcher(string[] patterns)
    {
        var type = typeof(HuggingFace.Tokenizers.Models.BPE.BpeModel).Assembly
            .GetType("HuggingFace.Tokenizers.Abstractions.AhoCorasickMatcher");
        Assert.IsNotNull(type, "AhoCorasickMatcher 类型未找到");
        return Activator.CreateInstance(type, (IReadOnlyList<string>)patterns)!;
    }

    [UnconditionalSuppressMessage("Trimming", "IL2075", Justification = "测试代码，FindAll 方法确定存在")]
    private static List<(int patternIndex, int start, int length)> FindAll(object matcher, string text)
    {
        var method = matcher.GetType().GetMethod("FindAll")!;
        return (List<(int, int, int)>)method.Invoke(matcher, new object[] { text })!;
    }

    [UnconditionalSuppressMessage("Trimming", "IL2075", Justification = "测试代码，FindLeftmostLongest 方法确定存在")]
    private static List<(int patternIndex, int start, int length)> FindLeftmostLongest(object matcher, string text)
    {
        return ((AhoCorasickMatcher)matcher).FindLeftmostLongest(text.AsSpan());
    }

    [TestMethod]
    public void FindAll_SinglePattern_FindsAllOccurrences()
    {
        var matcher = CreateMatcher(["ab"]);
        var results = FindAll(matcher, "ababab");

        Assert.AreEqual(3, results.Count);
        Assert.AreEqual((0, 0, 2), results[0]);
        Assert.AreEqual((0, 2, 2), results[1]);
        Assert.AreEqual((0, 4, 2), results[2]);
    }

    [TestMethod]
    public void FindAll_MultiplePatterns_FindsAll()
    {
        var matcher = CreateMatcher(["ab", "bc", "abc"]);
        var results = FindAll(matcher, "abc");

        // "ab" at 0, "abc" at 0, "bc" at 1
        Assert.IsTrue(results.Count >= 3);
    }

    [TestMethod]
    public void FindAll_Overlapping_FindsOverlaps()
    {
        var matcher = CreateMatcher(["aa", "aaa"]);
        var results = FindAll(matcher, "aaaa");

        // "aa" at 0, "aa" at 1, "aa" at 2, "aaa" at 0, "aaa" at 1
        Assert.IsTrue(results.Count >= 4);
    }

    [TestMethod]
    public void FindAll_EmptyText_ReturnsEmpty()
    {
        var matcher = CreateMatcher(["a", "b"]);
        var results = FindAll(matcher, "");

        Assert.AreEqual(0, results.Count);
    }

    [TestMethod]
    public void FindAll_EmptyPatterns_ReturnsEmpty()
    {
        var matcher = CreateMatcher(Array.Empty<string>());
        var results = FindAll(matcher, "hello");

        Assert.AreEqual(0, results.Count);
    }

    [TestMethod]
    public void FindAll_NoMatch_ReturnsEmpty()
    {
        var matcher = CreateMatcher(["xyz"]);
        var results = FindAll(matcher, "hello world");

        Assert.AreEqual(0, results.Count);
    }

    [TestMethod]
    public void FindAll_Unicode_HandlesCorrectly()
    {
        var matcher = CreateMatcher(["中国", "国人"]);
        var results = FindAll(matcher, "中国人");

        // "中国" at 0, "国人" at 1
        Assert.AreEqual(2, results.Count);
        Assert.AreEqual((0, 0, 2), results[0]);
        Assert.AreEqual((1, 1, 2), results[1]);
    }

    [TestMethod]
    public void FindAll_Emoji_HandlesCorrectly()
    {
        var matcher = CreateMatcher(["👋", "🌍"]);
        var results = FindAll(matcher, "👋🌍");

        Assert.AreEqual(2, results.Count);
    }

    [TestMethod]
    public void FindLeftmostLongest_SameStart_KeepsLongest()
    {
        var matcher = CreateMatcher(["ab", "abc", "abcd"]);
        var results = FindLeftmostLongest(matcher, "abcdef");

        Assert.AreEqual(1, results.Count);
        Assert.AreEqual((2, 0, 4), results[0]); // "abcd" 最长
    }

    [TestMethod]
    public void FindLeftmostLongest_DifferentStarts_KeepsAll()
    {
        var matcher = CreateMatcher(["ab", "cd"]);
        var results = FindLeftmostLongest(matcher, "abcd");

        Assert.AreEqual(2, results.Count);
        Assert.AreEqual((0, 0, 2), results[0]); // "ab" at 0
        Assert.AreEqual((1, 2, 2), results[1]); // "cd" at 2
    }

    [TestMethod]
    public void FindLeftmostLongest_EmptyText_ReturnsEmpty()
    {
        var matcher = CreateMatcher(["a"]);
        var results = FindLeftmostLongest(matcher, "");

        Assert.AreEqual(0, results.Count);
    }

    [TestMethod]
    public void FindLeftmostLongest_Unicode_HandlesCorrectly()
    {
        var matcher = CreateMatcher(["中国", "中国人"]);
        var results = FindLeftmostLongest(matcher, "中国人民");

        Assert.AreEqual(1, results.Count);
        Assert.AreEqual((1, 0, 3), results[0]); // "中国人" 最长
    }
}
