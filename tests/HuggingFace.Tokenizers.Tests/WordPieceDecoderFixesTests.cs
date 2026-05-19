using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Decoders;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// WordPieceDecoder 修复验证测试。
/// 验证缩写清理、标点间距等行为。
/// </summary>
[TestClass]
public class WordPieceDecoderFixesTests
{
    [TestMethod]
    public void Decode_Contraction_CleanedUp()
    {
        // 验证缩写被正确拼接（"don't" 形式）
        var decoder = new WordPieceDecoder(prefix: "##", cleanup: true);

        // 模拟 BERT 分词结果：do + n't
        var tokens = new List<string> { "do", "##n", "##'", "##t" };
        var result = decoder.Decode(tokens);

        // DecodeChain: [" do", "n", "'", "t"] → " don't"
        // CleanupSpacing: " n't" → "n't" → "don't"
        Assert.AreEqual("don't", result);
    }

    [TestMethod]
    public void Decode_Punctuation_NoSpaceBefore()
    {
        // 验证标点前无多余空格
        var decoder = new WordPieceDecoder(prefix: "##", cleanup: true);

        var tokens = new List<string> { "hello", "." };
        var result = decoder.Decode(tokens);

        Assert.AreEqual("hello.", result);
    }

    [TestMethod]
    public void Decode_ContinuationPrefix_Removed()
    {
        // 验证连续子词前缀被正确移除
        var decoder = new WordPieceDecoder(prefix: "##", cleanup: false);

        var tokens = new List<string> { "un", "##be", "##li", "##ev", "##able" };
        var result = decoder.Decode(tokens);

        // 与 Rust 一致：首 token 不加空格
        // DecodeChain: ["un", "be", "li", "ev", "able"] → "unbelievable"
        Assert.AreEqual("unbelievable", result);
    }

    [TestMethod]
    public void Decode_CleanupFalse_PreservesSpaces()
    {
        // 验证 cleanup=false 时保留原始间距（首个 token 前有空格）
        var decoder = new WordPieceDecoder(prefix: "##", cleanup: false);

        var tokens = new List<string> { "hello", ",", "world" };
        var result = decoder.Decode(tokens);

        // 与 Rust 一致：首 token 不加空格
        // DecodeChain: ["hello", ",", " world"] → "hello , world"
        Assert.AreEqual("hello , world", result);
    }

    [TestMethod]
    public void Decode_MultiplePunctuation_AllCleaned()
    {
        // 验证多个标点符号都被正确清理
        var decoder = new WordPieceDecoder(prefix: "##", cleanup: true);

        var tokens = new List<string> { "hello", ",", "world", "!" };
        var result = decoder.Decode(tokens);

        Assert.AreEqual("hello, world!", result);
    }

    [TestMethod]
    public void Decode_EmptyTokens_ReturnsEmpty()
    {
        // 验证空 token 列表返回空字符串
        var decoder = new WordPieceDecoder(prefix: "##", cleanup: true);

        var result = decoder.Decode(new List<string>());

        Assert.AreEqual("", result);
    }

    [TestMethod]
    public void Decode_SingleToken_NoPrefix()
    {
        // 验证单个 token 无前缀时前面加空格
        var decoder = new WordPieceDecoder(prefix: "##", cleanup: true);

        var result = decoder.Decode(new List<string> { "hello" });

        Assert.AreEqual("hello", result);
    }

    [TestMethod]
    public void DecodeChain_ReturnsCorrectParts()
    {
        // 验证 DecodeChain 返回正确的中间结果
        var decoder = new WordPieceDecoder(prefix: "##", cleanup: true);

        var tokens = new List<string> { "un", "##happy" };
        var chain = decoder.DecodeChain(tokens);

        Assert.AreEqual(2, chain.Count);
        Assert.AreEqual("un", chain[0]);    // 与 Rust 一致：首 token 不加空格
        Assert.AreEqual("happy", chain[1]); // 续接 token 去掉 prefix
    }
}
