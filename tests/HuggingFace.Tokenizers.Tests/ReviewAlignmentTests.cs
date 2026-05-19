using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Decoders;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Models.BPE;
using HuggingFace.Tokenizers.Normalizers;
using HuggingFace.Tokenizers.PreTokenizers;
using HuggingFace.Tokenizers.Processors;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Review 补充测试：对齐 Rust 版本的测试用例。
/// 覆盖 CTC 解码器、WordPiece cleanup、EncodeFast、TemplateProcessing pair 等。
/// </summary>
[TestClass]
public class ReviewAlignmentTests
{
    // ─────────────────────────────────────────────
    //  CTC 解码器（对齐 Rust decoders/ctc.rs 测试）
    // ─────────────────────────────────────────────

    [TestMethod]
    public void CtcDecoder_HandmadeSample()
    {
        // 对齐 Rust: handmade_sample
        var decoder = new CtcDecoder();
        var tokens = "<pad> <pad> h e e l l <pad> l o o o <pad>"
            .Split(' ');
        var result = decoder.DecodeChain(tokens);

        Assert.AreEqual(5, result.Count);
        Assert.AreEqual("h", result[0]);
        Assert.AreEqual("e", result[1]);
        Assert.AreEqual("l", result[2]);
        Assert.AreEqual("l", result[3]);
        Assert.AreEqual("o", result[4]);
    }

    [TestMethod]
    public void CtcDecoder_HandmadeWithDelimiter()
    {
        // 对齐 Rust: handmade_with_delimiter_sample
        var decoder = new CtcDecoder();
        var tokens = "<pad> <pad> h e e l l <pad> l o o o <pad> <pad> | <pad> w o o o r <pad> <pad> l l d <pad> <pad> <pad> <pad>"
            .Split(' ');
        var result = decoder.DecodeChain(tokens);

        Assert.AreEqual(11, result.Count);
        Assert.AreEqual("h", result[0]);
        Assert.AreEqual("e", result[1]);
        Assert.AreEqual("l", result[2]);
        Assert.AreEqual("l", result[3]);
        Assert.AreEqual("o", result[4]);
        Assert.AreEqual(" ", result[5]); // word delimiter → space
        Assert.AreEqual("w", result[6]);
        Assert.AreEqual("o", result[7]);
        Assert.AreEqual("r", result[8]);
        Assert.AreEqual("l", result[9]);
        Assert.AreEqual("d", result[10]);
    }

    [TestMethod]
    public void CtcDecoder_LibrispeechSample()
    {
        // 对齐 Rust: librispeech_sample
        var decoder = new CtcDecoder();
        var tokens = "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> A | | <pad> M <pad> <pad> <pad> <pad> A <pad> <pad> N <pad> <pad> <pad> | | | <pad> <pad> <pad> <pad> S <pad> <pad> <pad> A I <pad> D D | | T T <pad> O <pad> | | T H E E | | | <pad> U U <pad> N N <pad> I <pad> <pad> V <pad> <pad> <pad> E R R <pad> <pad> <pad> S E E | | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> S S <pad> <pad> <pad> <pad> I <pad> R R <pad> <pad> | | | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> I <pad> <pad> <pad> | <pad> <pad> <pad> E X <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> I <pad> S <pad> <pad> T <pad> <pad> | | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>"
            .Split(' ');
        var result = decoder.DecodeChain(tokens);

        var expected = new[] { "A", " ", "M", "A", "N", " ", "S", "A", "I", "D", " ", "T", "O", " ", "T", "H", "E", " ", "U", "N", "I", "V", "E", "R", "S", "E", " ", "S", "I", "R", " ", "I", " ", "E", "X", "I", "S", "T", " " };
        Assert.AreEqual(expected.Length, result.Count);
        for (int i = 0; i < expected.Length; i++)
            Assert.AreEqual(expected[i], result[i], $"Mismatch at index {i}");
    }

    [TestMethod]
    public void CtcDecoder_AnotherLibrispeechSample()
    {
        // 对齐 Rust: another_librispeech_sample
        var decoder = new CtcDecoder();
        var tokens = "<pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> H <pad> I <pad> S S | | <pad> <pad> <pad> I N <pad> <pad> S <pad> T T <pad> <pad> A N C C T <pad> | | | | | <pad> <pad> <pad> <pad> P <pad> <pad> <pad> <pad> A <pad> <pad> N N N <pad> <pad> I <pad> C <pad> <pad> | | <pad> W <pad> <pad> A S <pad> | | <pad> <pad> <pad> F <pad> <pad> O L <pad> <pad> L L O O W E E D | | <pad> B <pad> <pad> <pad> Y <pad> | | | A | | <pad> S S S <pad> M M <pad> <pad> <pad> A L L <pad> <pad> <pad> <pad> L <pad> | | | <pad> <pad> <pad> <pad> S H H <pad> <pad> <pad> <pad> A R R <pad> <pad> P <pad> <pad> | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> B <pad> <pad> L L <pad> <pad> <pad> <pad> <pad> O W W <pad> <pad> | | | <pad> <pad> <pad> <pad> <pad> <pad> <pad> H <pad> <pad> <pad> <pad> <pad> <pad> <pad> I G H H | | <pad> <pad> O N <pad> | | H <pad> I S S | | <pad> <pad> C H H <pad> <pad> <pad> E <pad> S S <pad> T T <pad> <pad> | | | <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>"
            .Split(' ');
        var result = decoder.DecodeChain(tokens);

        var expected = new[] { "H", "I", "S", " ", "I", "N", "S", "T", "A", "N", "C", "T", " ", "P", "A", "N", "I", "C", " ", "W", "A", "S", " ", "F", "O", "L", "L", "O", "W", "E", "D", " ", "B", "Y", " ", "A", " ", "S", "M", "A", "L", "L", " ", "S", "H", "A", "R", "P", " ", "B", "L", "O", "W", " ", "H", "I", "G", "H", " ", "O", "N", " ", "H", "I", "S", " ", "C", "H", "E", "S", "T", " " };
        Assert.AreEqual(expected.Length, result.Count);
        for (int i = 0; i < expected.Length; i++)
            Assert.AreEqual(expected[i], result[i], $"Mismatch at index {i}");
    }

    // ─────────────────────────────────────────────
    //  WordPiece 解码器 cleanup 行为
    // ─────────────────────────────────────────────

    [TestMethod]
    public void WordPiece_Cleanup_PerToken()
    {
        // 验证 cleanup 逐 token 执行（与 Rust 一致）
        var decoder = new WordPieceDecoder(prefix: "##", cleanup: true);

        // " hello" → cleanup: " ." → "." 逻辑不适用，因为 " ." 不在单个 token 内
        // 标点作为独立 token 时：token "." 不以 prefix 开头 → " ." → cleanup → "."
        var tokens = new[] { "hello", "." };
        var chain = decoder.DecodeChain(tokens);

        // 与 Rust 一致：首 token 不加空格
        // "hello" → CleanupToken: 不变
        // " ." → CleanupToken: 移除空格 → "."
        Assert.AreEqual("hello", chain[0]);
        Assert.AreEqual(".", chain[1]);
    }

    [TestMethod]
    public void WordPiece_Cleanup_ContractionPerToken()
    {
        // 验证缩写处理在 token 内部执行
        var decoder = new WordPieceDecoder(prefix: "##", cleanup: true);

        // 与 Rust 一致：首 token 不加空格
        // "I" → 首 token，不加空格 → cleanup → "I"
        // "n't" → 非首且无 prefix → " n't" → cleanup → "n't"
        var tokens = new[] { "I", "n't" };
        var chain = decoder.DecodeChain(tokens);
        Assert.AreEqual("I", chain[0]);
        Assert.AreEqual("n't", chain[1]); // " n't" → "n't"

        Assert.AreEqual("In't", decoder.Decode(tokens));
    }

    // ─────────────────────────────────────────────
    //  EncodeFast / EncodeBatchFast 一致性
    // ─────────────────────────────────────────────

    [TestMethod]
    public void EncodeFast_MatchesEncodeIds()
    {
        // 验证 EncodeFast 返回的 ID 与 Encode 一致
        var tokenizer = CreateTestTokenizer();

        var text = "Hello World!";
        var encoding = tokenizer.Encode(text);
        var fastIds = tokenizer.EncodeFast(text);

        Assert.AreEqual(encoding.GetIds().Length, fastIds.Length);
        for (int i = 0; i < encoding.GetIds().Length; i++)
            Assert.AreEqual(encoding.GetIds()[i], fastIds[i], $"ID mismatch at index {i}");
    }

    [TestMethod]
    public void EncodeBatchFast_MatchesEncodeBatch()
    {
        var tokenizer = CreateTestTokenizer();

        var texts = new[] { "Hello", "World", "!" };
        var batch = tokenizer.EncodeBatch(texts);
        var fastBatch = tokenizer.EncodeBatchFast(texts);

        Assert.AreEqual(batch.Length, fastBatch.Length);
        for (int i = 0; i < batch.Length; i++)
        {
            var ids = batch[i].GetIds();
            var fastIds = fastBatch[i];
            Assert.AreEqual(ids.Length, fastIds.Length, $"Length mismatch at index {i}");
            for (int j = 0; j < ids.Length; j++)
                Assert.AreEqual(ids[j], fastIds[j], $"ID mismatch at batch {i}, token {j}");
        }
    }

    [TestMethod]
    public void EncodeFast_WithSpecialTokens()
    {
        var tokenizer = CreateTestTokenizerWithSpecialTokens();

        var text = "hello world";
        var encoding = tokenizer.Encode(text);
        var fastIds = tokenizer.EncodeFast(text);

        Assert.AreEqual(encoding.GetIds().Length, fastIds.Length);
        for (int i = 0; i < encoding.GetIds().Length; i++)
            Assert.AreEqual(encoding.GetIds()[i], fastIds[i]);
    }

    [TestMethod]
    public void EncodeFast_EmptyString()
    {
        var tokenizer = CreateTestTokenizer();

        var fastIds = tokenizer.EncodeFast("");
        Assert.IsNotNull(fastIds);
        // 空字符串可能产生空结果或仅含特殊 token
    }

    // ─────────────────────────────────────────────
    //  TemplateProcessing pair 编码
    // ─────────────────────────────────────────────

    [TestMethod]
    public void TemplateProcessing_PairEncoding()
    {
        // 对齐 Rust template_processing 测试
        var clsId = 1u;
        var sepId = 0u;

        var processor = new TemplateProcessing(
            singleTemplate:
            [
                Template.Special(clsId, "[CLS]", 0),
                Template.A(0),
                Template.Special(sepId, "[SEP]", 0)
            ],
            pairTemplate:
            [
                Template.Special(clsId, "[CLS]", 0),
                Template.A(0),
                Template.Special(sepId, "[SEP]", 0),
                Template.B(1),
                Template.Special(sepId, "[SEP]", 1)
            ]);

        Assert.AreEqual(2, processor.AddedTokens(false));
        Assert.AreEqual(3, processor.AddedTokens(true));

        // 创建测试 encoding
        var encoding = new Encoding(
            ids: [12, 14],
            typeIds: [0, 0],
            tokens: ["Hello", "there"],
            words: [null, null],
            offsets: [(0, 5), (6, 11)],
            specialTokensMask: [0, 0],
            attentionMask: [1, 1]);

        var pair = new Encoding(
            ids: [15],
            typeIds: [0],
            tokens: ["pair"],
            words: [null],
            offsets: [(0, 4)],
            specialTokensMask: [0],
            attentionMask: [1]);

        // 单条编码
        var singleResult = processor.Process(encoding, null, true);
        Assert.AreEqual(4, singleResult.Length); // [CLS] Hello there [SEP]
        Assert.AreEqual(clsId, singleResult.GetIds()[0]);
        Assert.AreEqual(12u, singleResult.GetIds()[1]);
        Assert.AreEqual(14u, singleResult.GetIds()[2]);
        Assert.AreEqual(sepId, singleResult.GetIds()[3]);

        // 配对编码
        var pairResult = processor.Process(encoding, pair, true);
        Assert.AreEqual(6, pairResult.Length); // [CLS] Hello there [SEP] pair [SEP]
        Assert.AreEqual(clsId, pairResult.GetIds()[0]);
        Assert.AreEqual(12u, pairResult.GetIds()[1]);
        Assert.AreEqual(14u, pairResult.GetIds()[2]);
        Assert.AreEqual(sepId, pairResult.GetIds()[3]);
        Assert.AreEqual(15u, pairResult.GetIds()[4]);
        Assert.AreEqual(sepId, pairResult.GetIds()[5]);

        // 验证 typeIds
        Assert.AreEqual(0u, pairResult.GetTypeIds()[0]); // [CLS]
        Assert.AreEqual(0u, pairResult.GetTypeIds()[1]); // Hello
        Assert.AreEqual(0u, pairResult.GetTypeIds()[2]); // there
        Assert.AreEqual(0u, pairResult.GetTypeIds()[3]); // [SEP]
        Assert.AreEqual(1u, pairResult.GetTypeIds()[4]); // pair
        Assert.AreEqual(1u, pairResult.GetTypeIds()[5]); // [SEP]
    }

    [TestMethod]
    public void TemplateProcessing_PairNoSpecialTokens()
    {
        var processor = new TemplateProcessing(
            singleTemplate: [Template.A(0)],
            pairTemplate: [Template.A(0), Template.B(1)]);

        var encoding = new Encoding(
            ids: [12], typeIds: [0], tokens: ["Hello"],
            words: [null], offsets: [(0, 5)],
            specialTokensMask: [0], attentionMask: [1]);

        var pair = new Encoding(
            ids: [15], typeIds: [0], tokens: ["pair"],
            words: [null], offsets: [(0, 4)],
            specialTokensMask: [0], attentionMask: [1]);

        var result = processor.Process(encoding, pair, false);
        Assert.AreEqual(2, result.Length); // Hello pair（无特殊 token）
        Assert.AreEqual(12u, result.GetIds()[0]);
        Assert.AreEqual(15u, result.GetIds()[1]);
    }

    // ─────────────────────────────────────────────
    //  RobertaProcessing pair 编码
    // ─────────────────────────────────────────────

    [TestMethod]
    public void RobertaProcessing_PairEncoding()
    {
        // 对齐 Rust roberta_processing 测试
        var processor = new RobertaProcessing();

        var encoding = new Encoding(
            ids: [12, 14],
            typeIds: [0, 0],
            tokens: ["Hello", "there"],
            words: [null, null],
            offsets: [(0, 5), (6, 11)],
            specialTokensMask: [0, 0],
            attentionMask: [1, 1]);

        var pair = new Encoding(
            ids: [15],
            typeIds: [0],
            tokens: ["pair"],
            words: [null],
            offsets: [(0, 4)],
            specialTokensMask: [0],
            attentionMask: [1]);

        // 单条：<s> Hello there </s>
        var singleResult = processor.Process(encoding, null, true);
        Assert.AreEqual(4, singleResult.Length);
        Assert.AreEqual(0u, singleResult.GetIds()[0]);  // <s>
        Assert.AreEqual(12u, singleResult.GetIds()[1]); // Hello
        Assert.AreEqual(14u, singleResult.GetIds()[2]); // there
        Assert.AreEqual(2u, singleResult.GetIds()[3]);  // </s>

        // 验证所有 typeIds 为 0（RoBERTa 不区分段落）
        foreach (var typeId in singleResult.GetTypeIds())
            Assert.AreEqual(0u, typeId);

        // 配对：<s> Hello there </s> </s> pair </s>
        var pairResult = processor.Process(encoding, pair, true);
        Assert.AreEqual(7, pairResult.Length);
        Assert.AreEqual(0u, pairResult.GetIds()[0]);  // <s>
        Assert.AreEqual(12u, pairResult.GetIds()[1]); // Hello
        Assert.AreEqual(14u, pairResult.GetIds()[2]); // there
        Assert.AreEqual(2u, pairResult.GetIds()[3]);  // </s>
        Assert.AreEqual(2u, pairResult.GetIds()[4]);  // </s>
        Assert.AreEqual(15u, pairResult.GetIds()[5]); // pair
        Assert.AreEqual(2u, pairResult.GetIds()[6]);  // </s>

        // 配对时所有 typeIds 也为 0
        foreach (var typeId in pairResult.GetTypeIds())
            Assert.AreEqual(0u, typeId);
    }

    [TestMethod]
    public void RobertaProcessing_NoSpecialTokens()
    {
        var processor = new RobertaProcessing();

        var encoding = new Encoding(
            ids: [12, 14],
            typeIds: [0, 0],
            tokens: ["Hello", "there"],
            words: [null, null],
            offsets: [(0, 5), (6, 11)],
            specialTokensMask: [0, 0],
            attentionMask: [1, 1]);

        var pair = new Encoding(
            ids: [15],
            typeIds: [0],
            tokens: ["pair"],
            words: [null],
            offsets: [(0, 4)],
            specialTokensMask: [0],
            attentionMask: [1]);

        // 不添加特殊 token 时，直接合并
        var result = processor.Process(encoding, pair, false);
        Assert.AreEqual(3, result.Length); // Hello there pair
        Assert.AreEqual(12u, result.GetIds()[0]);
        Assert.AreEqual(14u, result.GetIds()[1]);
        Assert.AreEqual(15u, result.GetIds()[2]);
    }

    // ─────────────────────────────────────────────
    //  辅助方法
    // ─────────────────────────────────────────────

    private static Tokenizer CreateTestTokenizer()
    {
        var vocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0, ["hello"] = 1, ["world"] = 2, ["!"] = 3
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges([])
            .SetUnkToken("<unk>")
            .Build();

        return new TokenizerBuilder()
            .WithModel(model)
            .WithPreTokenizer(new WhitespacePreTokenizer())
            .Build();
    }

    private static Tokenizer CreateTestTokenizerWithSpecialTokens()
    {
        var vocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0, ["hello"] = 1, ["world"] = 2, ["!"] = 3,
            ["[CLS]"] = 4, ["[SEP]"] = 5
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges([])
            .SetUnkToken("<unk>")
            .Build();

        var tokenizer = new TokenizerBuilder()
            .WithModel(model)
            .WithPreTokenizer(new WhitespacePreTokenizer())
            .Build();

        tokenizer.AddToken(new AddedToken("[CLS]", isSpecial: true));
        tokenizer.AddToken(new AddedToken("[SEP]", isSpecial: true));

        return tokenizer;
    }
}
