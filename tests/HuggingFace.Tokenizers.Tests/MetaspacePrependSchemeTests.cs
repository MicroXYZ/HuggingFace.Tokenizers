using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.PreTokenizers;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// MetaspacePreTokenizer PrependScheme 测试。
/// 验证 PrependScheme.First 在不同场景下的行为。
/// </summary>
[TestClass]
public class MetaspacePrependSchemeTests
{
    [TestMethod]
    public void Metaspace_First_SingleSegment_PrependsPrefix()
    {
        // 验证 PrependScheme.First 对单个段落（起始位置为 0）正确添加前缀
        var pretokenizer = new MetaspacePreTokenizer(
            replacement: '▁',
            addPrefixSpace: true,
            prependScheme: PrependScheme.First);

        var pts = new PreTokenizedString("hello world");
        pretokenizer.PreTokenize(pts);

        var splits = pts.GetSplits().Select(s => s.Normalized.Get()).ToList();

        // 第一个 split 应包含前缀空格替换符
        Assert.IsTrue(splits[0].StartsWith("▁"),
            "PrependScheme.First 应在文本起始位置添加 ▁ 前缀");
    }

    [TestMethod]
    public void Metaspace_First_SpacesReplacedByReplacement()
    {
        // 验证空格被正确替换为 replacement 字符
        var pretokenizer = new MetaspacePreTokenizer(
            replacement: '▁',
            addPrefixSpace: true,
            prependScheme: PrependScheme.First);

        var pts = new PreTokenizedString("hello world");
        pretokenizer.PreTokenize(pts);

        var splits = pts.GetSplits().Select(s => s.Normalized.Get()).ToList();

        // 所有 split 中不应包含空格（已被替换）
        foreach (var split in splits)
        {
            Assert.IsFalse(split.Contains(' '),
                "split 中不应包含原始空格");
        }
    }

    [TestMethod]
    public void Metaspace_Always_AlwaysPrepends()
    {
        // 验证 PrependScheme.Always 总是添加前缀
        var pretokenizer = new MetaspacePreTokenizer(
            replacement: '▁',
            addPrefixSpace: true,
            prependScheme: PrependScheme.Always);

        var pts = new PreTokenizedString("hello world");
        pretokenizer.PreTokenize(pts);

        var splits = pts.GetSplits().Select(s => s.Normalized.Get()).ToList();

        // Always 模式下，第一个 split 应有前缀
        Assert.IsTrue(splits.Count > 0, "应至少有一个 split");
        Assert.IsTrue(splits[0].StartsWith("▁"),
            "PrependScheme.Always 应在起始位置添加 ▁ 前缀");
    }

    [TestMethod]
    public void Metaspace_Never_NeverPrepends()
    {
        // 验证 PrependScheme.Never 不添加前缀
        var pretokenizer = new MetaspacePreTokenizer(
            replacement: '▁',
            addPrefixSpace: true,
            prependScheme: PrependScheme.Never);

        var pts = new PreTokenizedString("hello world");
        pretokenizer.PreTokenize(pts);

        var splits = pts.GetSplits().Select(s => s.Normalized.Get()).ToList();

        Assert.IsTrue(splits.Count > 0, "应产生至少一个 split");

        // Never 模式下，首 split 不应以 ▁ 开头（文本不以空格开头）
        // 空格被替换为 ▁ 出现在 split 内部，但不会有额外的前缀 ▁
        Assert.IsFalse(splits[0].StartsWith("▁"),
            $"PrependScheme.Never 不应添加前缀 ▁，实际首 split: '{splits[0]}'");
    }

    [TestMethod]
    public void Metaspace_CustomReplacement_UsedCorrectly()
    {
        // 验证自定义替换字符被正确使用
        var pretokenizer = new MetaspacePreTokenizer(
            replacement: '^',
            addPrefixSpace: true,
            prependScheme: PrependScheme.First);

        var pts = new PreTokenizedString("hello world");
        pretokenizer.PreTokenize(pts);

        var splits = pts.GetSplits().Select(s => s.Normalized.Get()).ToList();

        // 应使用 ^ 而非 ▁
        Assert.IsTrue(splits.Any(s => s.Contains('^')),
            "应使用自定义替换字符 '^'");
    }
}
