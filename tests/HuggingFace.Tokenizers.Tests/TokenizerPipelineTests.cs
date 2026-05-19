using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Models.BPE;
using HuggingFace.Tokenizers.Models.WordLevel;
using HuggingFace.Tokenizers.Models.WordPiece;
using HuggingFace.Tokenizers.Normalizers;
using HuggingFace.Tokenizers.PreTokenizers;
using HuggingFace.Tokenizers.Decoders;
using HuggingFace.Tokenizers.Processors;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Tests for the core tokenizer pipeline.
/// </summary>
    [TestClass]
public class TokenizerPipelineTests
{
    [TestMethod]
    public void Encode_SimpleBPE_ReturnsTokens()
    {
        var vocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0, ["hello"] = 1, ["world"] = 2,
            ["the"] = 3, [" "] = 4
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges([])
            .SetUnkToken("<unk>")
            .Build();

        var tokenizer = new Tokenizer(model);
        var encoding = tokenizer.Encode("hello world");

        Assert.IsNotNull(encoding);
        Assert.IsTrue(encoding.Length > 0);
    }

    [TestMethod]
    public void Encode_WithNormalizer_AppliesNormalization()
    {
        var vocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0, ["hello"] = 1, ["world"] = 2
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab).SetMerges([]).SetUnkToken("<unk>").Build();

        var tokenizer = new Tokenizer(model)
        {
            Normalizer = new LowercaseNormalizer()
        };

        var encoding = tokenizer.Encode("HELLO WORLD");
        Assert.IsNotNull(encoding);
        Assert.IsTrue(encoding.Length > 0);
    }

    [TestMethod]
    public void Encode_WithPreTokenizer_SplitsCorrectly()
    {
        var vocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0, ["hello"] = 1, ["world"] = 2, ["!"] = 3
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab).SetMerges([]).SetUnkToken("<unk>").Build();

        var tokenizer = new Tokenizer(model)
        {
            PreTokenizer = new BertPreTokenizer()
        };

        var encoding = tokenizer.Encode("hello world!");
        Assert.IsNotNull(encoding);
        var ids = encoding.GetIds();
        // BertPreTokenizer 按空白+标点拆分，BPE 无 merge 所以逐字符编码
        // "hello" → h,e,l,l,o (5), "world" → w,o,r,l,d (5), "!" → 1 → 共 11 个 token
        Assert.AreEqual(11, ids.Length, "BPE 无 merge 时逐字符编码，共 11 个 token");
        // "hello" 中的字符不在 vocab 中 → UNK(0)；"world" 中的字符也不在 → UNK(0)
        // "!" 在 vocab 中 → id 3
        Assert.AreEqual(3u, ids[10], "! → id 3");
    }

    [TestMethod]
    public void Builder_CreatesTokenizer()
    {
        var vocab = new Dictionary<string, uint> { ["<unk>"] = 0, ["test"] = 1 };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab).SetMerges([]).SetUnkToken("<unk>").Build();

        var tokenizer = new TokenizerBuilder()
            .WithModel(model)
            .WithNormalizer(new NfcNormalizer())
            .WithPreTokenizer(new WhitespacePreTokenizer())
            .WithDecoder(new WordPieceDecoder())
            .Build();

        Assert.IsNotNull(tokenizer);
        Assert.IsNotNull(tokenizer.Normalizer);
        Assert.IsNotNull(tokenizer.PreTokenizer);
        Assert.IsNotNull(tokenizer.Decoder);
    }

    [TestMethod]
    public void EncodeBatch_MultipleInputs_ReturnsAllEncodings()
    {
        var vocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0, ["a"] = 1, ["b"] = 2, [" "] = 3
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab).SetMerges([]).SetUnkToken("<unk>").Build();

        var tokenizer = new Tokenizer(model);
        var encodings = tokenizer.EncodeBatch(["a b", "b a", "a a"]);

        Assert.AreEqual(3, encodings.Length);
        foreach (var e in encodings) Assert.IsTrue(e.Length > 0);
    }

    [TestMethod]
    public void Decode_Tokens_ReturnsText()
    {
        var vocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0, ["hello"] = 1, ["world"] = 2
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab).SetMerges([]).SetUnkToken("<unk>").Build();

        var tokenizer = new Tokenizer(model)
        {
            Decoder = new WordPieceDecoder()
        };

        var text = tokenizer.Decode([1, 2]);
        Assert.IsNotNull(text);
        Assert.AreEqual("hello world", text, "解码 [1,2] 应得到 'hello world'");
    }
}
