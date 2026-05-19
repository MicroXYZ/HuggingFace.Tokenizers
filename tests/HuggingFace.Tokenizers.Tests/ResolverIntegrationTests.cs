using System.Text.Json;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Decoders;
using HuggingFace.Tokenizers.Normalizers;
using HuggingFace.Tokenizers.PreTokenizers;
using HuggingFace.Tokenizers.Serialization;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Resolver 集成测试。
/// 覆盖 NormalizerResolver、PreTokenizerResolver、DecoderResolver、PostProcessorResolver、ModelResolver。
/// </summary>
[TestClass]
public class ResolverIntegrationTests
{
    // ── 辅助方法 ──

    private static NormalizerJsonModel CreateNormalizerModel(string type, Dictionary<string, JsonElement>? extra = null)
    {
        return new NormalizerJsonModel { Type = type, AdditionalData = extra };
    }

    private static PreTokenizerJsonModel CreatePreTokenizerModel(string type, Dictionary<string, JsonElement>? extra = null)
    {
        return new PreTokenizerJsonModel { Type = type, AdditionalData = extra };
    }

    private static DecoderJsonModel CreateDecoderModel(string type, Dictionary<string, JsonElement>? extra = null)
    {
        return new DecoderJsonModel { Type = type, AdditionalData = extra };
    }

    private static PostProcessorJsonModel CreatePostProcessorModel(string type, Dictionary<string, JsonElement>? extra = null)
    {
        return new PostProcessorJsonModel { Type = type, AdditionalData = extra };
    }

    private static JsonElement StringElement(string value) => JsonDocument.Parse($"\"{value}\"").RootElement;
    private static JsonElement BoolElement(bool value) => JsonDocument.Parse(value ? "true" : "false").RootElement;
    private static JsonElement IntElement(int value) => JsonDocument.Parse(value.ToString()).RootElement;
    private static JsonElement ParseJson(string json) => JsonDocument.Parse(json).RootElement;

    // ══════════════════════════════════════════════
    // NormalizerResolver
    // ══════════════════════════════════════════════

    [TestMethod]
    public void NormalizerResolver_BertNormalizer_ReturnsInstance()
    {
        var model = CreateNormalizerModel("BertNormalizer");
        var normalizer = NormalizerResolver.Resolve(model);

        Assert.IsInstanceOfType(normalizer, typeof(BertNormalizer));
    }

    [TestMethod]
    public void NormalizerResolver_NFC_ReturnsNfcNormalizer()
    {
        var model = CreateNormalizerModel("NFC");
        var normalizer = NormalizerResolver.Resolve(model);

        Assert.IsInstanceOfType(normalizer, typeof(NfcNormalizer));
    }

    [TestMethod]
    public void NormalizerResolver_NFD_ReturnsNfdNormalizer()
    {
        var model = CreateNormalizerModel("NFD");
        var normalizer = NormalizerResolver.Resolve(model);

        Assert.IsInstanceOfType(normalizer, typeof(NfdNormalizer));
    }

    [TestMethod]
    public void NormalizerResolver_NFKC_ReturnsNfkcNormalizer()
    {
        var model = CreateNormalizerModel("NFKC");
        var normalizer = NormalizerResolver.Resolve(model);

        Assert.IsInstanceOfType(normalizer, typeof(NfkcNormalizer));
    }

    [TestMethod]
    public void NormalizerResolver_NFKD_ReturnsNfkdNormalizer()
    {
        var model = CreateNormalizerModel("NFKD");
        var normalizer = NormalizerResolver.Resolve(model);

        Assert.IsInstanceOfType(normalizer, typeof(NfkdNormalizer));
    }

    [TestMethod]
    public void NormalizerResolver_Lowercase_ReturnsLowercaseNormalizer()
    {
        var model = CreateNormalizerModel("Lowercase");
        var normalizer = NormalizerResolver.Resolve(model);

        Assert.IsInstanceOfType(normalizer, typeof(LowercaseNormalizer));
    }

    [TestMethod]
    public void NormalizerResolver_Strip_ReturnsStripNormalizer()
    {
        var model = CreateNormalizerModel("Strip");
        var normalizer = NormalizerResolver.Resolve(model);

        Assert.IsInstanceOfType(normalizer, typeof(StripNormalizer));
    }

    [TestMethod]
    public void NormalizerResolver_StripAccents_ReturnsStripAccentsNormalizer()
    {
        var model = CreateNormalizerModel("StripAccents");
        var normalizer = NormalizerResolver.Resolve(model);

        Assert.IsInstanceOfType(normalizer, typeof(StripAccentsNormalizer));
    }

    [TestMethod]
    public void NormalizerResolver_Nmt_ReturnsNmtNormalizer()
    {
        var model = CreateNormalizerModel("Nmt");
        var normalizer = NormalizerResolver.Resolve(model);

        Assert.IsInstanceOfType(normalizer, typeof(NmtNormalizer));
    }

    [TestMethod]
    public void NormalizerResolver_ByteLevel_ReturnsByteLevelNormalizer()
    {
        var model = CreateNormalizerModel("ByteLevel");
        var normalizer = NormalizerResolver.Resolve(model);

        Assert.IsInstanceOfType(normalizer, typeof(ByteLevelNormalizer));
    }

    [TestMethod]
    public void NormalizerResolver_Sequence_ReturnsSequenceNormalizer()
    {
        var sequenceArray = ParseJson(@"[{""type"":""NFC""},{""type"":""Lowercase""}]");
        var model = CreateNormalizerModel("Sequence", new Dictionary<string, JsonElement>
        {
            ["normalizers"] = sequenceArray
        });
        var normalizer = NormalizerResolver.Resolve(model);

        Assert.IsInstanceOfType(normalizer, typeof(SequenceNormalizer));
    }

    [TestMethod]
    public void NormalizerResolver_Prepend_ReturnsPrependNormalizer()
    {
        var model = CreateNormalizerModel("Prepend", new Dictionary<string, JsonElement>
        {
            ["prepend"] = StringElement("▁")
        });
        var normalizer = NormalizerResolver.Resolve(model);

        Assert.IsInstanceOfType(normalizer, typeof(PrependNormalizer));
    }

    [TestMethod]
    public void NormalizerResolver_UnsupportedType_ThrowsNotSupported()
    {
        var model = CreateNormalizerModel("UnknownNormalizer");

        Assert.ThrowsExactly<NotSupportedException>(() => NormalizerResolver.Resolve(model));
    }

    [TestMethod]
    public void NormalizerResolver_NullModel_ThrowsArgumentNull()
    {
        Assert.ThrowsExactly<ArgumentNullException>(() => NormalizerResolver.Resolve(null!));
    }

    // ══════════════════════════════════════════════
    // PreTokenizerResolver
    // ══════════════════════════════════════════════

    [TestMethod]
    public void PreTokenizerResolver_BertPreTokenizer_ReturnsInstance()
    {
        var model = CreatePreTokenizerModel("BertPreTokenizer");
        var preTokenizer = PreTokenizerResolver.Resolve(model);

        Assert.IsInstanceOfType(preTokenizer, typeof(BertPreTokenizer));
    }

    [TestMethod]
    public void PreTokenizerResolver_ByteLevel_ReturnsByteLevelPreTokenizer()
    {
        var model = CreatePreTokenizerModel("ByteLevel");
        var preTokenizer = PreTokenizerResolver.Resolve(model);

        Assert.IsInstanceOfType(preTokenizer, typeof(ByteLevelPreTokenizer));
    }

    [TestMethod]
    public void PreTokenizerResolver_Whitespace_ReturnsWhitespacePreTokenizer()
    {
        var model = CreatePreTokenizerModel("Whitespace");
        var preTokenizer = PreTokenizerResolver.Resolve(model);

        Assert.IsInstanceOfType(preTokenizer, typeof(WhitespacePreTokenizer));
    }

    [TestMethod]
    public void PreTokenizerResolver_Metaspace_ReturnsMetaspacePreTokenizer()
    {
        var model = CreatePreTokenizerModel("Metaspace");
        var preTokenizer = PreTokenizerResolver.Resolve(model);

        Assert.IsInstanceOfType(preTokenizer, typeof(MetaspacePreTokenizer));
    }

    [TestMethod]
    public void PreTokenizerResolver_Digits_ReturnsDigitsPreTokenizer()
    {
        var model = CreatePreTokenizerModel("Digits");
        var preTokenizer = PreTokenizerResolver.Resolve(model);

        Assert.IsInstanceOfType(preTokenizer, typeof(DigitsPreTokenizer));
    }

    [TestMethod]
    public void PreTokenizerResolver_Punctuation_ReturnsPunctuationPreTokenizer()
    {
        var model = CreatePreTokenizerModel("Punctuation");
        var preTokenizer = PreTokenizerResolver.Resolve(model);

        Assert.IsInstanceOfType(preTokenizer, typeof(PunctuationPreTokenizer));
    }

    [TestMethod]
    public void PreTokenizerResolver_CharDelimiterSplit_ReturnsDelimiterSplitPreTokenizer()
    {
        var model = CreatePreTokenizerModel("CharDelimiterSplit", new Dictionary<string, JsonElement>
        {
            ["delimiter"] = StringElement("|")
        });
        var preTokenizer = PreTokenizerResolver.Resolve(model);

        Assert.IsInstanceOfType(preTokenizer, typeof(DelimiterSplitPreTokenizer));
    }

    [TestMethod]
    public void PreTokenizerResolver_UnicodeScripts_ReturnsInstance()
    {
        var model = CreatePreTokenizerModel("UnicodeScripts");
        var preTokenizer = PreTokenizerResolver.Resolve(model);

        Assert.IsInstanceOfType(preTokenizer, typeof(UnicodeScriptsPreTokenizer));
    }

    [TestMethod]
    public void PreTokenizerResolver_Sequence_ReturnsSequencePreTokenizer()
    {
        var sequenceArray = ParseJson(@"[{""type"":""Whitespace""},{""type"":""BertPreTokenizer""}]");
        var model = CreatePreTokenizerModel("Sequence", new Dictionary<string, JsonElement>
        {
            ["pretokenizers"] = sequenceArray
        });
        var preTokenizer = PreTokenizerResolver.Resolve(model);

        Assert.IsInstanceOfType(preTokenizer, typeof(SequencePreTokenizer));
    }

    [TestMethod]
    public void PreTokenizerResolver_UnsupportedType_ThrowsNotSupported()
    {
        var model = CreatePreTokenizerModel("UnknownPreTokenizer");

        Assert.ThrowsExactly<NotSupportedException>(() => PreTokenizerResolver.Resolve(model));
    }

    [TestMethod]
    public void PreTokenizerResolver_NullModel_ThrowsArgumentNull()
    {
        Assert.ThrowsExactly<ArgumentNullException>(() => PreTokenizerResolver.Resolve(null!));
    }

    // ══════════════════════════════════════════════
    // DecoderResolver
    // ══════════════════════════════════════════════

    [TestMethod]
    public void DecoderResolver_ByteLevel_ReturnsByteLevelDecoder()
    {
        var model = CreateDecoderModel("ByteLevel");
        var decoder = DecoderResolver.Resolve(model);

        Assert.IsInstanceOfType(decoder, typeof(ByteLevelDecoder));
    }

    [TestMethod]
    public void DecoderResolver_WordPiece_ReturnsWordPieceDecoder()
    {
        var model = CreateDecoderModel("WordPiece");
        var decoder = DecoderResolver.Resolve(model);

        Assert.IsInstanceOfType(decoder, typeof(WordPieceDecoder));
    }

    [TestMethod]
    public void DecoderResolver_Metaspace_ReturnsMetaspaceDecoder()
    {
        var model = CreateDecoderModel("Metaspace");
        var decoder = DecoderResolver.Resolve(model);

        Assert.IsInstanceOfType(decoder, typeof(MetaspaceDecoder));
    }

    [TestMethod]
    public void DecoderResolver_BPEDecoder_ReturnsBpeDecoder()
    {
        var model = CreateDecoderModel("BPEDecoder");
        var decoder = DecoderResolver.Resolve(model);

        Assert.IsInstanceOfType(decoder, typeof(BpeDecoder));
    }

    [TestMethod]
    public void DecoderResolver_CTC_ReturnsCtcDecoder()
    {
        var model = CreateDecoderModel("CTC");
        var decoder = DecoderResolver.Resolve(model);

        Assert.IsInstanceOfType(decoder, typeof(CtcDecoder));
    }

    [TestMethod]
    public void DecoderResolver_ByteFallback_ReturnsByteFallbackDecoder()
    {
        var model = CreateDecoderModel("ByteFallback");
        var decoder = DecoderResolver.Resolve(model);

        Assert.IsInstanceOfType(decoder, typeof(ByteFallbackDecoder));
    }

    [TestMethod]
    public void DecoderResolver_Fuse_ReturnsFuseDecoder()
    {
        var model = CreateDecoderModel("Fuse");
        var decoder = DecoderResolver.Resolve(model);

        Assert.IsInstanceOfType(decoder, typeof(FuseDecoder));
    }

    [TestMethod]
    public void DecoderResolver_Strip_ReturnsStripDecoder()
    {
        var model = CreateDecoderModel("Strip");
        var decoder = DecoderResolver.Resolve(model);

        Assert.IsInstanceOfType(decoder, typeof(StripDecoder));
    }

    [TestMethod]
    public void DecoderResolver_Replace_ReturnsReplaceDecoder()
    {
        var model = CreateDecoderModel("Replace", new Dictionary<string, JsonElement>
        {
            ["pattern"] = StringElement("##"),
            ["replacement"] = StringElement("")
        });
        var decoder = DecoderResolver.Resolve(model);

        Assert.IsInstanceOfType(decoder, typeof(ReplaceDecoder));
    }

    [TestMethod]
    public void DecoderResolver_Sequence_ReturnsSequenceDecoder()
    {
        var sequenceArray = ParseJson(@"[{""type"":""ByteFallback""},{""type"":""Fuse""}]");
        var model = CreateDecoderModel("Sequence", new Dictionary<string, JsonElement>
        {
            ["decoders"] = sequenceArray
        });
        var decoder = DecoderResolver.Resolve(model);

        Assert.IsInstanceOfType(decoder, typeof(SequenceDecoder));
    }

    [TestMethod]
    public void DecoderResolver_UnsupportedType_ThrowsNotSupported()
    {
        var model = CreateDecoderModel("UnknownDecoder");

        Assert.ThrowsExactly<NotSupportedException>(() => DecoderResolver.Resolve(model));
    }

    [TestMethod]
    public void DecoderResolver_NullModel_ThrowsArgumentNull()
    {
        Assert.ThrowsExactly<ArgumentNullException>(() => DecoderResolver.Resolve(null!));
    }

    // ══════════════════════════════════════════════
    // PostProcessorResolver
    // ══════════════════════════════════════════════

    [TestMethod]
    public void PostProcessorResolver_TemplateProcessing_WithValidData_ReturnsInstance()
    {
        var single = ParseJson(@"[{""SpecialToken"":{""id"":1,""type_id"":0}},{""Sequence"":{""id"":""A"",""type_id"":0}},{""SpecialToken"":{""id"":2,""type_id"":0}}]");
        var pair = ParseJson(@"[{""SpecialToken"":{""id"":1,""type_id"":0}},{""Sequence"":{""id"":""A"",""type_id"":0}},{""SpecialToken"":{""id"":2,""type_id"":0}},{""Sequence"":{""id"":""B"",""type_id"":1}},{""SpecialToken"":{""id"":2,""type_id"":1}}]");
        var model = CreatePostProcessorModel("TemplateProcessing", new Dictionary<string, JsonElement>
        {
            ["single"] = single,
            ["pair"] = pair
        });
        var processor = PostProcessorResolver.Resolve(model);

        Assert.IsNotNull(processor);
    }

    [TestMethod]
    public void PostProcessorResolver_UnsupportedType_ThrowsNotSupported()
    {
        var model = CreatePostProcessorModel("UnknownProcessor");

        Assert.ThrowsExactly<NotSupportedException>(() => PostProcessorResolver.Resolve(model));
    }

    [TestMethod]
    public void PostProcessorResolver_NullModel_ThrowsArgumentNull()
    {
        Assert.ThrowsExactly<ArgumentNullException>(() => PostProcessorResolver.Resolve(null!));
    }

    // ══════════════════════════════════════════════
    // ModelResolver
    // ══════════════════════════════════════════════

    [TestMethod]
    public void ModelResolver_BPE_WithVocab_ReturnsInstance()
    {
        var vocab = ParseJson(@"{""a"":0,""b"":1,""ab"":2}");
        var model = new ModelJsonModel
        {
            Type = "BPE",
            AdditionalData = new Dictionary<string, JsonElement>
            {
                ["vocab"] = vocab
            }
        };
        var result = ModelResolver.Resolve(model);

        Assert.IsNotNull(result);
    }

    [TestMethod]
    public void ModelResolver_WordPiece_WithVocab_ReturnsInstance()
    {
        var vocab = ParseJson(@"{""[UNK]"":0,""hello"":1,""world"":2}");
        var model = new ModelJsonModel
        {
            Type = "WordPiece",
            AdditionalData = new Dictionary<string, JsonElement>
            {
                ["vocab"] = vocab
            }
        };
        var result = ModelResolver.Resolve(model);

        Assert.IsNotNull(result);
    }

    [TestMethod]
    public void ModelResolver_WordLevel_WithVocab_ReturnsInstance()
    {
        var vocab = ParseJson(@"{""<unk>"":0,""hello"":1,""world"":2}");
        var model = new ModelJsonModel
        {
            Type = "WordLevel",
            AdditionalData = new Dictionary<string, JsonElement>
            {
                ["vocab"] = vocab
            }
        };
        var result = ModelResolver.Resolve(model);

        Assert.IsNotNull(result);
    }

    [TestMethod]
    public void ModelResolver_Unigram_WithVocab_ReturnsInstance()
    {
        var vocab = ParseJson(@"[[""hello"",0.5],[""world"",0.3]]");
        var model = new ModelJsonModel
        {
            Type = "Unigram",
            AdditionalData = new Dictionary<string, JsonElement>
            {
                ["vocab"] = vocab
            }
        };
        var result = ModelResolver.Resolve(model);

        Assert.IsNotNull(result);
    }

    [TestMethod]
    public void ModelResolver_UnsupportedType_ThrowsNotSupported()
    {
        var model = new ModelJsonModel { Type = "UnknownModel" };

        Assert.ThrowsExactly<NotSupportedException>(() => ModelResolver.Resolve(model));
    }
}
