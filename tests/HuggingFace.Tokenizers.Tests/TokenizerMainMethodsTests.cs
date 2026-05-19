using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Decoders;
using HuggingFace.Tokenizers.Internal;
using HuggingFace.Tokenizers.Models.BPE;
using HuggingFace.Tokenizers.PreTokenizers;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Tokenizer 主类方法测试。
/// 使用 BPE + ByteLevel（与 Rust 版 GPT-2 默认行为一致）覆盖核心方法。
/// </summary>
[TestClass]
public class TokenizerMainMethodsTests
{
    /// <summary>
    /// 创建 BPE + ByteLevel Tokenizer（模拟 GPT-2 默认配置）。
    /// ByteLevel pre-tokenizer 将 UTF-8 字节映射为可打印字符，
    /// BPE 在字节级 token 上执行合并。
    /// </summary>
    private static Tokenizer CreateTokenizer()
    {
        // GPT-2 风格字节级词表：256 个字节 token + 合并结果
        var vocab = new Dictionary<string, uint>();
        // 256 个字节级 token（ByteLevelMapping 的字符映射）
        for (int i = 0; i < 256; i++)
        {
            vocab[ByteLevelMapping.ByteToChar[i].ToString()] = (uint)i;
        }
        // 合并后的 token
        vocab["ll"] = 256;
        vocab["hello"] = 257;
        vocab["world"] = 258;

        // 合并规则：需要基于字节级 token 的形式
        // "hello" 的字节级 token 序列：h(0x68) e(0x65) l(0x6C) l(0x6C) o(0x6F)
        string hTok = ByteLevelMapping.ByteToChar[0x68].ToString();
        string eTok = ByteLevelMapping.ByteToChar[0x65].ToString();
        string lTok = ByteLevelMapping.ByteToChar[0x6C].ToString();
        string oTok = ByteLevelMapping.ByteToChar[0x6F].ToString();
        string wTok = ByteLevelMapping.ByteToChar[0x77].ToString();
        string rTok = ByteLevelMapping.ByteToChar[0x72].ToString();
        string dTok = ByteLevelMapping.ByteToChar[0x64].ToString();

        var merges = new List<(string, string)>
        {
            (lTok, lTok),     // ll
            (hTok, eTok),     // he
            ("he", lTok),     // hel
            ("hel", lTok),    // hell
            ("hell", oTok),   // hello
        };

        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges(merges)
            .Build();

        var tokenizer = new Tokenizer(model);
        tokenizer.PreTokenizer = new ByteLevelPreTokenizer(addPrefixSpace: false);
        tokenizer.Decoder = new ByteLevelDecoder();
        return tokenizer;
    }

    // ── Encode ──

    [TestMethod]
    public void Encode_ValidText_ReturnsNonEmptyEncoding()
    {
        var tokenizer = CreateTokenizer();
        var encoding = tokenizer.Encode("hello");

        Assert.IsNotNull(encoding);
        Assert.IsTrue(encoding.GetIds().Length > 0);
    }

    [TestMethod]
    public void Encode_EmptyString_ReturnsEncoding()
    {
        var tokenizer = CreateTokenizer();
        var encoding = tokenizer.Encode("");
        Assert.IsNotNull(encoding);
    }

    [TestMethod]
    public void Encode_SingleByte_ReturnsId()
    {
        var tokenizer = CreateTokenizer();
        var encoding = tokenizer.Encode("A");

        Assert.IsTrue(encoding.GetIds().Length > 0);
        // ByteLevel 映射后 id 由 ByteToChar 决定，不等于原始字节值
    }

    [TestMethod]
    public void Encode_ProducesConsistentResults()
    {
        var tokenizer = CreateTokenizer();
        var enc1 = tokenizer.Encode("hello");
        var enc2 = tokenizer.Encode("hello");

        CollectionAssert.AreEqual(enc1.GetIds(), enc2.GetIds());
    }

    // ── EncodeFast ──

    [TestMethod]
    public void EncodeFast_ValidText_ReturnsIds()
    {
        var tokenizer = CreateTokenizer();
        var ids = tokenizer.EncodeFast("A");

        Assert.IsNotNull(ids);
        Assert.IsTrue(ids.Length > 0);
    }

    [TestMethod]
    public void EncodeFast_EmptyString_ReturnsIds()
    {
        var tokenizer = CreateTokenizer();
        var ids = tokenizer.EncodeFast("");
        Assert.IsNotNull(ids);
    }

    [TestMethod]
    public void EncodeFast_SameAsEncode_IdsMatch()
    {
        var tokenizer = CreateTokenizer();
        var encoding = tokenizer.Encode("hello");
        var fastIds = tokenizer.EncodeFast("hello");

        CollectionAssert.AreEqual(encoding.GetIds(), fastIds);
    }

    // ── EncodeCharOffsets ──

    [TestMethod]
    public void EncodeCharOffsets_ReturnsEncodingWithOffsets()
    {
        var tokenizer = CreateTokenizer();
        var encoding = tokenizer.EncodeCharOffsets("hello");

        Assert.IsNotNull(encoding);
        Assert.IsTrue(encoding.GetIds().Length > 0);
        Assert.AreEqual(encoding.GetIds().Length, encoding.GetOffsets().Length);
    }

    [TestMethod]
    public void EncodeCharOffsets_EmptyString_ReturnsEncoding()
    {
        var tokenizer = CreateTokenizer();
        var encoding = tokenizer.EncodeCharOffsets("");
        Assert.IsNotNull(encoding);
    }

    [TestMethod]
    public void EncodeCharOffsets_OffsetsWithinBounds()
    {
        var tokenizer = CreateTokenizer();
        var input = "hello world";
        var encoding = tokenizer.EncodeCharOffsets(input);
        var offsets = encoding.GetOffsets();

        // 所有偏移应在 [0, input.Length] 范围内，且 Start <= End
        Assert.IsTrue(offsets.All(o => o.Start >= 0 && o.End <= input.Length && o.Start <= o.End));
    }

    // ── EncodePair / EncodeCharOffsetsPair ──

    [TestMethod]
    public void EncodePair_TwoInputs_ReturnsCombinedEncoding()
    {
        var tokenizer = CreateTokenizer();
        var encoding = tokenizer.EncodePair("hello", "world");

        Assert.IsNotNull(encoding);
        Assert.IsTrue(encoding.GetIds().Length > 0);
    }

    [TestMethod]
    public void EncodePair_SecondNull_ReturnsSingleEncoding()
    {
        var tokenizer = CreateTokenizer();
        var single = tokenizer.Encode("hello");
        var pair = tokenizer.EncodePair("hello", null);

        CollectionAssert.AreEqual(single.GetIds(), pair.GetIds());
    }

    [TestMethod]
    public void EncodeCharOffsetsPair_TwoInputs_ReturnsCombinedEncoding()
    {
        var tokenizer = CreateTokenizer();
        var encoding = tokenizer.EncodeCharOffsetsPair("hello", "world");

        Assert.IsNotNull(encoding);
        Assert.IsTrue(encoding.GetIds().Length > 0);
        Assert.AreEqual(encoding.GetIds().Length, encoding.GetOffsets().Length);
    }

    // ── EncodeBatch ──

    [TestMethod]
    public void EncodeBatch_MultipleInputs_ReturnsCorrectCount()
    {
        var tokenizer = CreateTokenizer();
        var inputs = new[] { "hello", "world", "hello world" };
        var results = tokenizer.EncodeBatch(inputs);

        Assert.AreEqual(3, results.Length);
    }

    [TestMethod]
    public void EncodeBatch_EmptyList_ReturnsEmptyList()
    {
        var tokenizer = CreateTokenizer();
        var results = tokenizer.EncodeBatch(Array.Empty<string>());

        Assert.AreEqual(0, results.Length);
    }

    [TestMethod]
    public void EncodeBatch_ConsistentWithSingleEncode()
    {
        var tokenizer = CreateTokenizer();
        var batch = tokenizer.EncodeBatch(new[] { "hello", "world" });
        var single1 = tokenizer.Encode("hello");
        var single2 = tokenizer.Encode("world");

        CollectionAssert.AreEqual(single1.GetIds(), batch[0].GetIds());
        CollectionAssert.AreEqual(single2.GetIds(), batch[1].GetIds());
    }

    // ── EncodeBatchFast ──

    [TestMethod]
    public void EncodeBatchFast_MultipleInputs_ReturnsCorrectCount()
    {
        var tokenizer = CreateTokenizer();
        var results = tokenizer.EncodeBatchFast(new[] { "hello", "world" });

        Assert.AreEqual(2, results.Length);
    }

    [TestMethod]
    public void EncodeBatchFast_EmptyList_ReturnsEmptyList()
    {
        var tokenizer = CreateTokenizer();
        var results = tokenizer.EncodeBatchFast(Array.Empty<string>());

        Assert.AreEqual(0, results.Length);
    }

    // ── EncodeBatchCharOffsets ──

    [TestMethod]
    public void EncodeBatchCharOffsets_MultipleInputs_ReturnsCorrectCount()
    {
        var tokenizer = CreateTokenizer();
        var results = tokenizer.EncodeBatchCharOffsets(new[] { "hello", "world" });

        Assert.AreEqual(2, results.Length);
        Assert.IsTrue(results[0].GetOffsets().Length > 0);
    }

    // ── EncodeBatchPair ──

    [TestMethod]
    public void EncodeBatchPair_MultiplePairs_ReturnsCorrectCount()
    {
        var tokenizer = CreateTokenizer();
        var firsts = new[] { "hello", "hello" };
        var seconds = new string?[] { "world", null };

        var results = tokenizer.EncodeBatchPair(firsts, seconds);

        Assert.AreEqual(2, results.Length);
    }

    // ── Decode ──

    [TestMethod]
    public void Decode_SingleId_ReturnsChar()
    {
        var tokenizer = CreateTokenizer();
        var encoding = tokenizer.Encode("A");
        var text = tokenizer.Decode(encoding.GetIds().ToList());

        Assert.AreEqual("A", text);
    }

    [TestMethod]
    public void Decode_EmptyIds_ReturnsEmptyString()
    {
        var tokenizer = CreateTokenizer();
        var text = tokenizer.Decode(new List<uint>());

        Assert.AreEqual("", text);
    }

    [TestMethod]
    public void Decode_ByteSequence_ReturnsText()
    {
        var tokenizer = CreateTokenizer();
        // 编码 "AB" 然后解码验证
        var encoding = tokenizer.Encode("AB");
        var text = tokenizer.Decode(encoding.GetIds().ToList());

        Assert.AreEqual("AB", text);
    }

    // ── DecodeBatch ──

    [TestMethod]
    public void DecodeBatch_MultipleInputs_ReturnsCorrectResults()
    {
        var tokenizer = CreateTokenizer();
        var encA = tokenizer.Encode("A");
        var encB = tokenizer.Encode("B");
        var inputs = new IReadOnlyList<uint>[]
        {
            encA.GetIds().ToList(),
            encB.GetIds().ToList()
        };
        var results = tokenizer.DecodeBatch(inputs);

        Assert.AreEqual(2, results.Length);
        Assert.AreEqual("A", results[0]);
        Assert.AreEqual("B", results[1]);
    }

    [TestMethod]
    public void DecodeBatch_EmptyList_ReturnsEmptyList()
    {
        var tokenizer = CreateTokenizer();
        var results = tokenizer.DecodeBatch(Array.Empty<IReadOnlyList<uint>>());

        Assert.AreEqual(0, results.Length);
    }

    // ── TokenToId / IdToToken ──

    [TestMethod]
    public void TokenToId_ByteToken_ReturnsId()
    {
        var tokenizer = CreateTokenizer();
        // Byte token for 'h' (0x68)
        string hToken = ByteLevelMapping.ByteToChar[0x68].ToString();
        var id = tokenizer.TokenToId(hToken);

        Assert.IsNotNull(id);
        Assert.AreEqual(0x68u, id.Value);
    }

    [TestMethod]
    public void TokenToId_UnknownToken_ReturnsNull()
    {
        var tokenizer = CreateTokenizer();
        var id = tokenizer.TokenToId("此token不存在");

        Assert.IsNull(id);
    }

    [TestMethod]
    public void IdToToken_KnownId_ReturnsToken()
    {
        var tokenizer = CreateTokenizer();
        // id=0x68 → byte token for 'h'
        var token = tokenizer.IdToToken(0x68);

        Assert.IsNotNull(token);
        Assert.AreEqual(ByteLevelMapping.ByteToChar[0x68].ToString(), token);
    }

    [TestMethod]
    public void IdToToken_UnknownId_ReturnsNull()
    {
        var tokenizer = CreateTokenizer();
        var token = tokenizer.IdToToken(999);

        Assert.IsNull(token);
    }

    [TestMethod]
    public void TokenToId_RoundTrip_PreservesToken()
    {
        var tokenizer = CreateTokenizer();
        string expected = ByteLevelMapping.ByteToChar[0x41].ToString(); // 'A'
        var id = tokenizer.TokenToId(expected);
        Assert.IsNotNull(id);

        var token = tokenizer.IdToToken(id.Value);
        Assert.AreEqual(expected, token);
    }

    // ── AddToken / AddTokens ──

    [TestMethod]
    public void AddToken_NewToken_IncreasesVocabSize()
    {
        var tokenizer = CreateTokenizer();
        int sizeBefore = tokenizer.GetVocabSizeWithAddedTokens();

        tokenizer.AddToken(new AddedToken("[NEW]"));
        int sizeAfter = tokenizer.GetVocabSizeWithAddedTokens();

        Assert.AreEqual(sizeBefore + 1, sizeAfter);
    }

    [TestMethod]
    public void AddTokens_MultipleTokens_IncreasesVocabSize()
    {
        var tokenizer = CreateTokenizer();
        int sizeBefore = tokenizer.GetVocabSizeWithAddedTokens();

        var tokens = new List<AddedToken>
        {
            new AddedToken("[T1]"),
            new AddedToken("[T2]"),
            new AddedToken("[T3]")
        };
        tokenizer.AddTokens(tokens);
        int sizeAfter = tokenizer.GetVocabSizeWithAddedTokens();

        Assert.AreEqual(sizeBefore + 3, sizeAfter);
    }

    [TestMethod]
    public void AddToken_NewToken_IsAccessibleViaTokenToId()
    {
        var tokenizer = CreateTokenizer();
        tokenizer.AddToken(new AddedToken("[CUSTOM]"));

        var id = tokenizer.TokenToId("[CUSTOM]");
        Assert.IsNotNull(id);
    }

    // ── GetVocabWithAddedTokens / GetVocabSizeWithAddedTokens ──

    [TestMethod]
    public void GetVocabWithAddedTokens_ReturnsAllTokens()
    {
        var tokenizer = CreateTokenizer();
        var vocab = tokenizer.GetVocabWithAddedTokens();

        Assert.IsNotNull(vocab);
        Assert.IsTrue(vocab.Count > 0);
    }

    [TestMethod]
    public void GetVocabSizeWithAddedTokens_ReturnsCorrectSize()
    {
        var tokenizer = CreateTokenizer();
        int size = tokenizer.GetVocabSizeWithAddedTokens();

        Assert.IsTrue(size > 0);
    }

    [TestMethod]
    public void GetVocabSizeWithAddedTokens_WithAddedTokens_IncludesAdded()
    {
        var tokenizer = CreateTokenizer();
        int sizeBefore = tokenizer.GetVocabSizeWithAddedTokens();

        tokenizer.AddToken(new AddedToken("[NEW]"));
        int sizeAfter = tokenizer.GetVocabSizeWithAddedTokens();

        Assert.AreEqual(sizeBefore + 1, sizeAfter);
    }

    // ── CreateDecodeStream ──

    [TestMethod]
    public void CreateDecodeStream_Default_ReturnsStream()
    {
        var tokenizer = CreateTokenizer();
        var stream = tokenizer.CreateDecodeStream();

        Assert.IsNotNull(stream);
    }

    [TestMethod]
    public void CreateDecodeStream_SkipSpecialTokensFalse_ReturnsStream()
    {
        var tokenizer = CreateTokenizer();
        var stream = tokenizer.CreateDecodeStream(skipSpecialTokens: false);

        Assert.IsNotNull(stream);
    }
}
