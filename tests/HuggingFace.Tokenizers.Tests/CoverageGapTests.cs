using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Decoders;
using HuggingFace.Tokenizers.Models;
using HuggingFace.Tokenizers.Models.BPE;
using HuggingFace.Tokenizers.Models.WordPiece;
using HuggingFace.Tokenizers.Normalizers;
using HuggingFace.Tokenizers.PreTokenizers;
using HuggingFace.Tokenizers.Processors;
using HuggingFace.Tokenizers.Serialization;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// 覆盖 Review 中发现的 9 个零测试公共组件。
/// 每个组件至少覆盖核心逻辑路径。
/// </summary>
[TestClass]
public class CoverageGapTests
{
    // ══════════════════════════════════════════════
    //  FuseDecoder
    // ══════════════════════════════════════════════

    [TestMethod]
    public void FuseDecoder_DecodeChain_ReturnsSingleToken()
    {
        var decoder = new FuseDecoder();
        var result = decoder.DecodeChain(["hel", "lo", " world"]);
        Assert.AreEqual(1, result.Count);
        Assert.AreEqual("hello world", result[0]);
    }

    [TestMethod]
    public void FuseDecoder_Decode_ConcatenatesAllTokens()
    {
        var decoder = new FuseDecoder();
        var result = decoder.Decode(["a", "b", "c"]);
        Assert.AreEqual("abc", result);
    }

    [TestMethod]
    public void FuseDecoder_EmptyTokens_ReturnsEmpty()
    {
        var decoder = new FuseDecoder();
        var result = decoder.Decode([]);
        Assert.AreEqual("", result);
    }

    // ══════════════════════════════════════════════
    //  SequenceDecoder
    // ══════════════════════════════════════════════

    [TestMethod]
    public void SequenceDecoder_ChainsDecoders()
    {
        var wp = new WordPieceDecoder();
        var fuse = new FuseDecoder();
        var seq = new SequenceDecoder([wp, fuse]);

        // WP 解码：非 ## token 前加空格，## 去掉前缀
        // ["hel", "##lo", "world"] → WP → ["hel", "lo", " world"] → fuse → "hel lo world"
        var result = seq.Decode(["hel", "##lo", "world"]);
        // WordPieceDecoder 在非首 token 前加空格（非 ## token）
        Assert.IsTrue(result.Contains("hel"), $"应包含 'hel': {result}");
        Assert.IsTrue(result.Contains("lo"), $"应包含 'lo': {result}");
        Assert.IsTrue(result.Contains("world"), $"应包含 'world': {result}");
    }

    [TestMethod]
    public void SequenceDecoder_DecodeChain_PassesThroughAll()
    {
        var wp = new WordPieceDecoder();
        var seq = new SequenceDecoder([wp]);

        var result = seq.DecodeChain(["un", "##affect", "##ed"]);
        Assert.AreEqual(3, result.Count);
        Assert.AreEqual("un", result[0]);
        Assert.AreEqual("affect", result[1]);
        Assert.AreEqual("ed", result[2]);
    }

    [TestMethod]
    public void SequenceDecoder_EmptyDecoders_ReturnsOriginal()
    {
        var seq = new SequenceDecoder([]);
        var result = seq.Decode(["hello"]);
        Assert.AreEqual("hello", result);
    }

    [TestMethod]
    public void SequenceDecoder_NullDecoders_Throws()
    {
        Assert.ThrowsExactly<ArgumentNullException>(() => _ = new SequenceDecoder(null!));
    }

    // ══════════════════════════════════════════════
    //  SequenceProcessor
    // ══════════════════════════════════════════════

    [TestMethod]
    public void SequenceProcessor_ChainsProcessors()
    {
        var p1 = new BertProcessing(("[CLS]", 1), ("[SEP]", 2));
        var p2 = new BertProcessing(("[BOS]", 3), ("[EOS]", 4));
        var seq = new SequenceProcessor(p1, p2);

        var vocab = new Dictionary<string, uint>
        {
            ["[CLS]"] = 1, ["[SEP]"] = 2, ["[BOS]"] = 3, ["[EOS]"] = 4,
            ["hello"] = 5, ["world"] = 6, ["<unk>"] = 0
        };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab).SetMerges([]).SetUnkToken("<unk>").Build();
        var tokenizer = new Tokenizer(model);
        var encoding = tokenizer.Encode("hello world");

        var result = seq.Process(encoding, null, true);
        Assert.IsNotNull(result);
        Assert.IsTrue(result.GetIds().Length > 2, "应包含特殊 token");
    }

    [TestMethod]
    public void SequenceProcessor_AddedTokens_SumsAll()
    {
        var p1 = new BertProcessing(("[CLS]", 1), ("[SEP]", 2));
        var p2 = new BertProcessing(("[BOS]", 3), ("[EOS]", 4));
        var seq = new SequenceProcessor(p1, p2);

        // BertProcessing.AddedTokens(isPair=false) = 2, isPair=true = 3
        // 两个处理器链式：2+2=4 (single), 3+3=6 (pair)
        Assert.AreEqual(4, seq.AddedTokens(false), "single: 2+2=4");
        Assert.AreEqual(6, seq.AddedTokens(true), "pair: 3+3=6");
    }

    [TestMethod]
    public void SequenceProcessor_EmptyProcessors_Throws()
    {
        Assert.ThrowsExactly<ArgumentException>(() => _ = new SequenceProcessor());
    }

    // ══════════════════════════════════════════════
    //  PrecompiledNormalizer
    // ══════════════════════════════════════════════

    [TestMethod]
    public void PrecompiledNormalizer_EmptyMap_DoesNotModify()
    {
        var normalizer = new PrecompiledNormalizer([]);
        var ns = new NormalizedString("Hello World");
        normalizer.Normalize(ns);
        Assert.AreEqual("Hello World", ns.Get());
    }

    [TestMethod]
    public void PrecompiledNormalizer_DefaultOptions_AreCorrect()
    {
        var normalizer = new PrecompiledNormalizer([]);
        Assert.IsTrue(normalizer.AddDummyPrefix);
        Assert.IsTrue(normalizer.EscapeWhiteSpaces);
        Assert.IsTrue(normalizer.RemoveExtraWhiteSpaces);
        Assert.IsFalse(normalizer.TreatWhitespaceAsSuffix);
    }

    // ══════════════════════════════════════════════
    //  ReplaceNormalizer
    // ══════════════════════════════════════════════

    [TestMethod]
    public void ReplaceNormalizer_StringPattern_ReplacesAll()
    {
        var normalizer = new ReplaceNormalizer("foo", "bar");
        var ns = new NormalizedString("foo baz foo");
        normalizer.Normalize(ns);
        Assert.AreEqual("bar baz bar", ns.Get());
    }

    [TestMethod]
    public void ReplaceNormalizer_RegexPattern_ReplacesAll()
    {
        var normalizer = new ReplaceNormalizer(new System.Text.RegularExpressions.Regex(@"\s+"), "_");
        var ns = new NormalizedString("hello   world");
        normalizer.Normalize(ns);
        Assert.AreEqual("hello_world", ns.Get());
    }

    [TestMethod]
    public void ReplaceNormalizer_Properties_AreCorrect()
    {
        var normalizer = new ReplaceNormalizer("pat", "rep");
        Assert.AreEqual("pat", normalizer.PatternString);
        Assert.AreEqual("rep", normalizer.Replacement);
        Assert.IsFalse(normalizer.IsRegexPattern);
    }

    [TestMethod]
    public void ReplaceNormalizer_NullPattern_Throws()
    {
        Assert.ThrowsExactly<ArgumentNullException>(() => _ = new ReplaceNormalizer((string)null!, "rep"));
    }

    // ══════════════════════════════════════════════
    //  ByteLevelNormalizer
    // ══════════════════════════════════════════════

    [TestMethod]
    public void ByteLevelNormalizer_AsciiText_MapsToGPT2Chars()
    {
        var normalizer = new ByteLevelNormalizer();
        var ns = new NormalizedString("Hello");
        normalizer.Normalize(ns);
        var result = ns.Get();
        Assert.AreEqual("Hello", result);
    }

    [TestMethod]
    public void ByteLevelNormalizer_EmptyText_DoesNothing()
    {
        var normalizer = new ByteLevelNormalizer();
        var ns = new NormalizedString("");
        normalizer.Normalize(ns);
        Assert.AreEqual("", ns.Get());
    }

    [TestMethod]
    public void ByteLevelNormalizer_Unicode_MapsUtf8Bytes()
    {
        var normalizer = new ByteLevelNormalizer();
        var ns = new NormalizedString("你好");
        normalizer.Normalize(ns);
        var result = ns.Get();
        Assert.AreNotEqual("你好", result, "中文字符应被重新映射");
        Assert.IsTrue(result.Length > 0);
    }

    // ══════════════════════════════════════════════
    //  StripAccentsNormalizer
    // ══════════════════════════════════════════════

    [TestMethod]
    public void StripAccentsNormalizer_RemovesAccents()
    {
        var normalizer = new StripAccentsNormalizer();
        var ns = new NormalizedString("café résumé");
        normalizer.Normalize(ns);
        Assert.AreEqual("cafe resume", ns.Get());
    }

    [TestMethod]
    public void StripAccentsNormalizer_NoAccents_Unchanged()
    {
        var normalizer = new StripAccentsNormalizer();
        var ns = new NormalizedString("hello world");
        normalizer.Normalize(ns);
        Assert.AreEqual("hello world", ns.Get());
    }

    [TestMethod]
    public void StripAccentsNormalizer_EmptyText_Unchanged()
    {
        var normalizer = new StripAccentsNormalizer();
        var ns = new NormalizedString("");
        normalizer.Normalize(ns);
        Assert.AreEqual("", ns.Get());
    }

    // ══════════════════════════════════════════════
    //  ModelWrapper
    // ══════════════════════════════════════════════

    [TestMethod]
    public void ModelWrapper_DelegatesTokenize()
    {
        var vocab = new Dictionary<string, uint> { ["<unk>"] = 0, ["hello"] = 1, ["world"] = 2 };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab).SetMerges([]).SetUnkToken("<unk>").Build();

        var wrapper = ModelWrapper.CreateBPE(model);
        var tokens = wrapper.Tokenize("hello world");
        Assert.IsTrue(tokens.Count > 0);
    }

    [TestMethod]
    public void ModelWrapper_DelegatesTokenToId()
    {
        var vocab = new Dictionary<string, uint> { ["<unk>"] = 0, ["hello"] = 1 };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab).SetMerges([]).SetUnkToken("<unk>").Build();

        var wrapper = ModelWrapper.CreateBPE(model);
        Assert.AreEqual((uint)1, wrapper.TokenToId("hello"));
        Assert.IsNull(wrapper.TokenToId("nonexistent"));
    }

    [TestMethod]
    public void ModelWrapper_DelegatesIdToToken()
    {
        var vocab = new Dictionary<string, uint> { ["<unk>"] = 0, ["hello"] = 1 };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab).SetMerges([]).SetUnkToken("<unk>").Build();

        var wrapper = ModelWrapper.CreateBPE(model);
        Assert.AreEqual("hello", wrapper.IdToToken(1));
        Assert.IsNull(wrapper.IdToToken(999));
    }

    [TestMethod]
    public void ModelWrapper_DelegatesGetVocabSize()
    {
        var vocab = new Dictionary<string, uint> { ["<unk>"] = 0, ["hello"] = 1 };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab).SetMerges([]).SetUnkToken("<unk>").Build();

        var wrapper = ModelWrapper.CreateBPE(model);
        Assert.AreEqual(2, wrapper.GetVocabSize());
    }

    [TestMethod]
    public void ModelWrapper_AutoDetect_DetectsBPE()
    {
        var vocab = new Dictionary<string, uint> { ["<unk>"] = 0 };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab).SetMerges([]).SetUnkToken("<unk>").Build();

        var wrapper = ModelWrapper.AutoDetect(model);
        Assert.AreEqual(ModelType.BPE, wrapper.Type);
    }

    [TestMethod]
    public void ModelWrapper_AutoDetect_DetectsWordPiece()
    {
        var vocab = new Dictionary<string, uint> { ["<unk>"] = 0 };
        var model = new WordPieceModel.WordPieceBuilder()
            .SetVocab(vocab).Build();

        var wrapper = ModelWrapper.AutoDetect(model);
        Assert.AreEqual(ModelType.WordPiece, wrapper.Type);
    }

    [TestMethod]
    public void ModelWrapper_ToString_ContainsTypeName()
    {
        var vocab = new Dictionary<string, uint> { ["<unk>"] = 0 };
        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab).SetMerges([]).SetUnkToken("<unk>").Build();

        var wrapper = ModelWrapper.CreateBPE(model);
        var str = wrapper.ToString();
        Assert.IsTrue(str.Contains("BPE"), $"应包含模型类型: {str}");
        Assert.IsTrue(str.Contains("ModelWrapper"), $"应包含类名: {str}");
    }

    [TestMethod]
    public void ModelWrapper_NullModel_Throws()
    {
        Assert.ThrowsExactly<ArgumentNullException>(() => _ = new ModelWrapper(null!, ModelType.BPE));
    }

    // ══════════════════════════════════════════════
    //  TokenizerDownloader — 仅测试参数验证，不测试网络
    // ══════════════════════════════════════════════

    [TestMethod]
    public void TokenizerDownloader_DefaultCacheDir_IsSet()
    {
        Assert.IsFalse(string.IsNullOrEmpty(TokenizerDownloader.DefaultCacheDir));
        Assert.IsTrue(TokenizerDownloader.DefaultCacheDir.Contains("huggingface"));
    }

    [TestMethod]
    public async Task TokenizerDownloader_EmptyModelId_Throws()
    {
        try
        {
            await TokenizerDownloader.DownloadAsync("");
            Assert.Fail("应抛出 ArgumentException");
        }
        catch (ArgumentException) { /* 预期 */ }
    }

    [TestMethod]
    public async Task TokenizerDownloader_NullModelId_Throws()
    {
        try
        {
            await TokenizerDownloader.DownloadAsync(null!);
            Assert.Fail("应抛出 ArgumentException");
        }
        catch (ArgumentException) { /* 预期 */ }
    }
}
