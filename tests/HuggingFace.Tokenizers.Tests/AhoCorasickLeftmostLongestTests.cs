using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Normalizers;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// LeftmostLongest 匹配语义测试。
/// 通过 AddedVocabulary（公共 API）验证 AC 匹配行为，
/// 与 Rust daachorse::MatchKind::LeftmostLongest 语义一致。
/// </summary>
    [TestClass]
public class AhoCorasickLeftmostLongestTests
{
    /// <summary>
    /// 创建带 added tokens 的 Tokenizer，用于测试匹配语义。
    /// </summary>
    private static Tokenizer CreateTokenizerWithTokens(
        IReadOnlyList<AddedToken> tokens,
        INormalizer? normalizer = null)
    {
        // 使用简单 BPE 模型，词表只包含基本字符
        var vocab = new Dictionary<string, uint>
        {
            ["a"] = 0, ["b"] = 1, ["c"] = 2, ["d"] = 3,
            ["e"] = 4, ["f"] = 5, [" "] = 6,
            ["ab"] = 10, ["abc"] = 11, ["abcd"] = 12,
            ["中国"] = 20, ["中国人"] = 21, ["国"] = 22,
        };
        var model = new HuggingFace.Tokenizers.Models.BPE.BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges([])
            .Build();

        var tokenizer = new Tokenizer(model) { Normalizer = normalizer };
        tokenizer.AddTokens(tokens);
        return tokenizer;
    }

    [TestMethod]
    public void LeftmostLongest_同一位置多个匹配_保留最长()
    {
        // 添加 "ab", "a", "abcd" 三个 token
        // 输入 "abcd"，位置 0 有三个匹配，应保留最长的 "abcd"
        var tokenizer = CreateTokenizerWithTokens([
            new AddedToken("ab"),
            new AddedToken("a"),
            new AddedToken("abcd"),
        ]);

        var encoding = tokenizer.Encode("abcd", addSpecialTokens: false);
        var tokens = encoding.GetTokens();

        // LeftmostLongest: "abcd" 应作为单个 token 被匹配
        Assert.AreEqual(1, tokens.Count());
        Assert.AreEqual("abcd", tokens[0]);
    }

    [TestMethod]
    public void LeftmostLongest_重叠匹配_正确选择()
    {
        // 添加 "abc", "bc", "c"
        // 输入 "abc"，位置 0 有 "abc"，位置 1 有 "bc"，位置 2 有 "c"
        var tokenizer = CreateTokenizerWithTokens([
            new AddedToken("abc"),
            new AddedToken("bc"),
            new AddedToken("c"),
        ]);

        var encoding = tokenizer.Encode("abc", addSpecialTokens: false);
        var tokens = encoding.GetTokens();

        // LeftmostLongest: "abc" 在位置 0 被匹配为最长
        Assert.AreEqual(1, tokens.Count());
        Assert.AreEqual("abc", tokens[0]);
    }

    [TestMethod]
    public void LeftmostLongest_不重叠匹配_全部保留()
    {
        // 添加 "ab", "cd"
        // 输入 "abcd"，两个匹配不重叠
        var tokenizer = CreateTokenizerWithTokens([
            new AddedToken("ab"),
            new AddedToken("cd"),
        ]);

        var encoding = tokenizer.Encode("abcd", addSpecialTokens: false);
        var tokens = encoding.GetTokens();

        Assert.AreEqual(2, tokens.Length);
        Assert.AreEqual("ab", tokens[0]);
        Assert.AreEqual("cd", tokens[1]);
    }

    [TestMethod]
    public void LeftmostLongest_中文token_正确匹配()
    {
        // 添加 "中国", "中国人", "国"
        // 输入 "中国人"，位置 0 有 "中国" 和 "中国人"，应保留最长的 "中国人"
        var tokenizer = CreateTokenizerWithTokens([
            new AddedToken("中国"),
            new AddedToken("中国人"),
            new AddedToken("国"),
        ]);

        var encoding = tokenizer.Encode("中国人", addSpecialTokens: false);
        var tokens = encoding.GetTokens();

        Assert.AreEqual(1, tokens.Count());
        Assert.AreEqual("中国人", tokens[0]);
    }

    [TestMethod]
    public void LeftmostLongest_短pattern被长pattern覆盖_不单独出现()
    {
        // 添加 "a", "abc"
        // 输入 "abc"，"a" 在位置 0 被 "abc" 覆盖
        var tokenizer = CreateTokenizerWithTokens([
            new AddedToken("a"),
            new AddedToken("abc"),
        ]);

        var encoding = tokenizer.Encode("abc", addSpecialTokens: false);
        var tokens = encoding.GetTokens();

        Assert.AreEqual(1, tokens.Count());
        Assert.AreEqual("abc", tokens[0]);
    }

    [TestMethod]
    public void LeftmostLongest_标准化后匹配_正确工作()
    {
        // 添加标准化 token "abc"（normalized=true）
        // 使用 lowercase 标准化器，输入 "ABC" 应匹配 "abc"
        var tokenizer = CreateTokenizerWithTokens(
            [new AddedToken("abc", normalized: true)],
            normalizer: new LowercaseNormalizer());

        var encoding = tokenizer.Encode("ABC", addSpecialTokens: false);
        var tokens = encoding.GetTokens();

        // 标准化后 "ABC" → "abc"，应匹配 added token "abc"
        Assert.AreEqual(1, tokens.Count());
        Assert.AreEqual("abc", tokens[0]);
    }

    [TestMethod]
    public void LeftmostLongest_非标准化token优先_先于标准化token匹配()
    {
        // Phase 1 (非标准化): "<s>" 匹配原始文本
        // Phase 2 (标准化): "abc" 匹配标准化文本
        var tokenizer = CreateTokenizerWithTokens(
            [
                new AddedToken("<s>"),
                new AddedToken("abc", normalized: true),
            ],
            normalizer: new LowercaseNormalizer());

        var encoding = tokenizer.Encode("<s>abc", addSpecialTokens: false);
        var tokens = encoding.GetTokens();

        // "<s>" 是非标准化 token，在 Phase 1 匹配
        // "abc" 是标准化 token，在 Phase 2 匹配
        Assert.AreEqual(2, tokens.Length);
        Assert.AreEqual("<s>", tokens[0]);
        Assert.AreEqual("abc", tokens[1]);
    }
}
