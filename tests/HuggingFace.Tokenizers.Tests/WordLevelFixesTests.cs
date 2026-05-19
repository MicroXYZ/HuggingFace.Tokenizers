using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Models.WordLevel;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// WordLevel 模型修复验证测试。
/// 验证 WordLevel 的分词、UNK 处理、训练回写等行为。
/// </summary>
[TestClass]
public class WordLevelFixesTests
{
    [TestMethod]
    public void Tokenize_MultiWordInput_ReturnsSingleToken()
    {
        // 验证 WordLevel 将整个输入作为单个 token 查找（与 Rust 一致）
        var vocab = new Dictionary<string, uint>
        {
            ["hello world"] = 0,
            ["<unk>"] = 1
        };
        var model = new WordLevelModel.WordLevelBuilder()
            .SetVocab(vocab)
            .SetUnkToken("<unk>")
            .Build();

        var tokens = model.Tokenize("hello world");

        Assert.AreEqual(1, tokens.Count);
        Assert.AreEqual(0u, tokens[0].Id);
        Assert.AreEqual("hello world", tokens[0].Value);
    }

    [TestMethod]
    public void Tokenize_UnknownInput_ReturnsUnk()
    {
        // 验证未知输入返回 UNK token
        var vocab = new Dictionary<string, uint>
        {
            ["hello"] = 0,
            ["<unk>"] = 1
        };
        var model = new WordLevelModel.WordLevelBuilder()
            .SetVocab(vocab)
            .SetUnkToken("<unk>")
            .Build();

        var tokens = model.Tokenize("world");

        Assert.AreEqual(1, tokens.Count);
        Assert.AreEqual(1u, tokens[0].Id); // UNK id
        Assert.AreEqual("<unk>", tokens[0].Value);
    }

    [TestMethod]
    public void Tokenize_KnownInput_ReturnsCorrectToken()
    {
        // 验证已知输入返回正确的 token
        var vocab = new Dictionary<string, uint>
        {
            ["hello"] = 0,
            ["world"] = 1,
            ["<unk>"] = 2
        };
        var model = new WordLevelModel.WordLevelBuilder()
            .SetVocab(vocab)
            .SetUnkToken("<unk>")
            .Build();

        var tokens = model.Tokenize("hello");

        Assert.AreEqual(1, tokens.Count);
        Assert.AreEqual(0u, tokens[0].Id);
        Assert.AreEqual("hello", tokens[0].Value);
        Assert.AreEqual(0, tokens[0].Start);
        Assert.AreEqual(5, tokens[0].End);
    }

    [TestMethod]
    public void Tokenize_EmptyInput_ReturnsEmpty()
    {
        // 验证空输入返回空列表
        var vocab = new Dictionary<string, uint> { ["<unk>"] = 0 };
        var model = new WordLevelModel.WordLevelBuilder()
            .SetVocab(vocab)
            .SetUnkToken("<unk>")
            .Build();

        var tokens = model.Tokenize("");

        Assert.AreEqual(0, tokens.Count);
    }

    [TestMethod]
    public void DefaultUnkToken_IsAngleBrackets()
    {
        // 验证默认 UNK token 是 "<unk>" 而非 "[UNK]"
        var vocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0,
            ["hello"] = 1
        };
        // 使用默认 UnkToken（不显式设置）
        var model = new WordLevelModel.WordLevelBuilder()
            .SetVocab(vocab)
            .Build();

        Assert.AreEqual("<unk>", model.UnkToken);
    }

    [TestMethod]
    public void TokenToId_KnownToken_ReturnsId()
    {
        // 验证 TokenToId 查找已知 token
        var vocab = new Dictionary<string, uint>
        {
            ["hello"] = 42,
            ["<unk>"] = 0
        };
        var model = new WordLevelModel.WordLevelBuilder()
            .SetVocab(vocab)
            .Build();

        Assert.AreEqual(42u, model.TokenToId("hello"));
        Assert.IsNull(model.TokenToId("unknown"));
    }

    [TestMethod]
    public void IdToToken_ValidId_ReturnsToken()
    {
        // 验证 IdToToken 反向查找
        var vocab = new Dictionary<string, uint>
        {
            ["hello"] = 42,
            ["<unk>"] = 0
        };
        var model = new WordLevelModel.WordLevelBuilder()
            .SetVocab(vocab)
            .Build();

        Assert.AreEqual("hello", model.IdToToken(42));
        Assert.IsNull(model.IdToToken(999));
    }
}
