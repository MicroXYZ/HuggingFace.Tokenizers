using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Models.BPE;
using HuggingFace.Tokenizers.Models.WordLevel;
using HuggingFace.Tokenizers.Padding;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// EncodeBatch + 填充策略测试。
/// 验证批量编码时 BatchLongest 和 Fixed 填充策略的正确性。
/// </summary>
[TestClass]
public class EncodeBatchTests
{
    /// <summary>
    /// 创建一个简单的 BPE 分词器用于测试。
    /// </summary>
    private static Tokenizer CreateSimpleBpeTokenizer()
    {
        var vocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0,
            ["a"] = 1, ["b"] = 2, ["c"] = 3, ["d"] = 4,
            ["e"] = 5, ["f"] = 6, ["g"] = 7, ["h"] = 8,
            ["i"] = 9, ["l"] = 10, ["n"] = 11, ["o"] = 12,
            ["r"] = 13, ["s"] = 14, ["t"] = 15, ["w"] = 16,
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges(new List<(string, string)>())
            .SetUnkToken("<unk>")
            .Build();

        return new Tokenizer(model);
    }

    [TestMethod]
    public void EncodeBatch_WithBatchLongest_AllSequencesSameLength()
    {
        var tokenizer = CreateSimpleBpeTokenizer();
        tokenizer.Padding = new PaddingParams
        {
            Strategy = PaddingStrategy.BatchLongest,
            PadId = 0,
            PadToken = "[PAD]",
        };

        var texts = new[] { "ab", "abcd", "a" };
        var results = tokenizer.EncodeBatch(texts);

        Assert.AreEqual(3, results.Length);

        int expectedLength = 4;
        foreach (var encoding in results)
        {
            Assert.AreEqual(expectedLength, encoding.Length,
                "所有 encoding 的长度应一致（等于最长序列）");
        }

        Assert.AreEqual(4, results[0].Length);
        CollectionAssert.AreEqual(new uint[] { 1, 2, 0, 0 }, results[0].GetIds());
        CollectionAssert.AreEqual(new uint[] { 1, 2, 3, 4 }, results[1].GetIds());
        CollectionAssert.AreEqual(new uint[] { 1, 0, 0, 0 }, results[2].GetIds());
    }

    [TestMethod]
    public void EncodeBatch_WithFixedPadding_AllSequencesSameLength()
    {
        var tokenizer = CreateSimpleBpeTokenizer();
        tokenizer.Padding = new PaddingParams
        {
            Strategy = PaddingStrategy.Fixed,
            MaxLength = 6,
            PadId = 0,
            PadToken = "[PAD]",
        };

        var texts = new[] { "ab", "abcd" };
        var results = tokenizer.EncodeBatch(texts);

        Assert.AreEqual(2, results.Length);

        foreach (var encoding in results)
        {
            Assert.AreEqual(6, encoding.Length);
        }

        CollectionAssert.AreEqual(new uint[] { 1, 2, 0, 0, 0, 0 }, results[0].GetIds());
        CollectionAssert.AreEqual(new uint[] { 1, 2, 3, 4, 0, 0 }, results[1].GetIds());
    }

    [TestMethod]
    public void EncodeBatch_WithBatchLongest_AttentionMaskCorrect()
    {
        var tokenizer = CreateSimpleBpeTokenizer();
        tokenizer.Padding = new PaddingParams
        {
            Strategy = PaddingStrategy.BatchLongest,
            PadId = 0,
        };

        var texts = new[] { "abc", "a" };
        var results = tokenizer.EncodeBatch(texts);

        CollectionAssert.AreEqual(new uint[] { 1, 1, 1 }, results[0].GetAttentionMask());
        CollectionAssert.AreEqual(new uint[] { 1, 0, 0 }, results[1].GetAttentionMask());
    }

    [TestMethod]
    public void EncodeBatch_EmptyList_ReturnsEmpty()
    {
        var tokenizer = CreateSimpleBpeTokenizer();
        var results = tokenizer.EncodeBatch(Array.Empty<string>());

        Assert.AreEqual(0, results.Length);
    }

    [TestMethod]
    public void EncodeBatch_SingleElement_NoPadding()
    {
        var tokenizer = CreateSimpleBpeTokenizer();
        tokenizer.Padding = new PaddingParams
        {
            Strategy = PaddingStrategy.BatchLongest,
            PadId = 0,
        };

        var results = tokenizer.EncodeBatch(new[] { "ab" });

        Assert.AreEqual(1, results.Length);
        Assert.AreEqual(2, results[0].Length);
        CollectionAssert.AreEqual(new uint[] { 1, 2 }, results[0].GetIds());
    }

    [TestMethod]
    public void EncodeBatch_WithoutPadding_NoPaddingApplied()
    {
        var tokenizer = CreateSimpleBpeTokenizer();

        var texts = new[] { "ab", "abcd" };
        var results = tokenizer.EncodeBatch(texts);

        Assert.AreEqual(2, results[0].Length);
        Assert.AreEqual(4, results[1].Length);
    }
}
