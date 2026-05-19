using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Models.BPE;
using HuggingFace.Tokenizers.PreTokenizers;
using HuggingFace.Tokenizers.Processors;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// 偏移 Trimming 和双序列偏移测试。
/// 对齐 Rust tokenizers 的 byte_level pre-tokenizer 和 PostProcessor 行为。
/// </summary>
[TestClass]
public class EncodingOffsetTrimmingTests
{
    /// <summary>
    /// 辅助方法：创建带有 ByteLevel 组件的 BPE 分词器。
    /// 词表包含 GPT-2 风格的字节级 token，以及常见英文词和 BPE 合并规则，
    /// 确保 BPE 能将单字符合并为整词 token。
    /// </summary>
    private static Tokenizer CreateByteLevelTokenizer(bool trimOffsets, bool addPrefixSpace = false)
    {
        var vocab = new Dictionary<string, uint>();
        uint id = 0;

        // 添加所有单字节 token（GPT-2 风格的字节编码）
        for (int b = 0; b < 256; b++)
        {
            var byteStr = ByteLevelPreTokenizer.Encode(((char)b).ToString());
            if (!vocab.ContainsKey(byteStr))
                vocab[byteStr] = id++;
        }

        // 添加常见词的字节级组合（整词条目）
        var words = new[] { "Hello", "there", "how", "are", "you", "My", "name", "is", "Anthony" };
        foreach (var word in words)
        {
            var encoded = ByteLevelPreTokenizer.Encode(word);
            if (!vocab.ContainsKey(encoded))
                vocab[encoded] = id++;
        }

        // 添加带前导空格的词（Ġ 前缀）
        var spaceWords = new[] { " there", " how", " are", " you", " name", " is", " Anthony" };
        foreach (var word in spaceWords)
        {
            var encoded = ByteLevelPreTokenizer.Encode(word);
            if (!vocab.ContainsKey(encoded))
                vocab[encoded] = id++;
        }

        // 添加 BPE 合并所需的中间 token
        foreach (var s in new[] {
            "He", "Hel", "Hell",                       // Hello
            "th", "the", "ther",                       // there
            "ho", "ar", "yo", "na", "nam",             // how/are/you/name
            "An", "Ant", "Anth", "Antho", "Anthon",    // Anthony
            "Ġt", "Ġth", "Ġthe", "Ġther",             // Ġthere
            "Ġh", "Ġho",                               // Ġhow
            "Ġa", "Ġar",                               // Ġare
            "Ġy", "Ġyo",                               // Ġyou
            "Ġn", "Ġna", "Ġnam",                       // Ġname
            "ĠA", "ĠAn", "ĠAnt", "ĠAnth", "ĠAntho", "ĠAnthon" // ĠAnthony
        })
            if (!vocab.ContainsKey(s))
                vocab[s] = id++;

        // BPE 合并规则。
        // 关键：space-prefix 合并（Ġ+x）必须排在字符合并（x+y）之前，
        // 否则 x+y 先消耗掉 x，Ġ+x 就无法执行。
        // 这与 Rust GPT-2 merges.txt 中 Ġ 前缀 token 排在前面的惯例一致。
        var merges = new List<(string, string)>
        {
            // ── Ġthere（space-prefix 优先） ──
            ("Ġ", "t"), ("Ġt", "h"), ("Ġth", "e"), ("Ġthe", "r"), ("Ġther", "e"),
            // ── Ġhow ──
            ("Ġ", "h"), ("Ġh", "o"), ("Ġho", "w"),
            // ── Ġare ──
            ("Ġ", "a"), ("Ġa", "r"), ("Ġar", "e"),
            // ── Ġyou ──
            ("Ġ", "y"), ("Ġy", "o"), ("Ġyo", "u"),
            // ── Ġname ──
            ("Ġ", "n"), ("Ġn", "a"), ("Ġna", "m"), ("Ġnam", "e"),
            // ── ĠAnthony ──
            ("Ġ", "A"), ("ĠA", "n"), ("ĠAn", "t"), ("ĠAnt", "h"),
            ("ĠAnth", "o"), ("ĠAntho", "n"), ("ĠAnthon", "y"),
            // ── Hello ──
            ("H", "e"), ("He", "l"), ("Hel", "l"), ("Hell", "o"),
            // ── there ──
            ("t", "h"), ("th", "e"), ("the", "r"), ("ther", "e"),
            // ── how ──
            ("h", "o"), ("ho", "w"),
            // ── are ──
            ("a", "r"), ("ar", "e"),
            // ── you ──
            ("y", "o"), ("yo", "u"),
            // ── name ──
            ("n", "a"), ("na", "m"), ("nam", "e"),
            // ── Anthony ──
            ("A", "n"), ("An", "t"), ("Ant", "h"), ("Anth", "o"), ("Antho", "n"), ("Anthon", "y"),
        };

        var model = new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges(merges)
            .Build();

        return new Tokenizer(model)
        {
            PreTokenizer = new ByteLevelPreTokenizer(
                addPrefixSpace: addPrefixSpace,
                useRegex: true,
                trimOffsets: trimOffsets),
            PostProcessor = new ByteLevelPostProcessor(
                trimOffsets: trimOffsets,
                addPrefixSpace: addPrefixSpace)
        };
    }

    // ── 偏移 Trimming ─────────────────────────────

    [TestMethod]
    public void Encode_NoTrim_OffsetsIncludeLeadingSpace()
    {
        // Arrange — addPrefixSpace=false, so "Hello there" splits into ["Hello", " there"]
        var tokenizer = CreateByteLevelTokenizer(trimOffsets: false, addPrefixSpace: false);
        var input = "Hello there";

        // Act
        var encoding = tokenizer.Encode(input);

        // Assert — BPE produces: ["Hello", "Ġthere"]
        // No trim: offsets include leading space in second token
        var offsets = encoding.GetOffsets();
        Assert.AreEqual(2, encoding.Length);
        Assert.AreEqual("Hello", encoding.GetTokens()[0]);
        Assert.AreEqual("Ġthere", encoding.GetTokens()[1]);
        Assert.AreEqual((0, 5), offsets[0]);   // "Hello"
        Assert.AreEqual((5, 11), offsets[1]);   // " there" (including space)

        // 验证偏移能正确映射回原文
        Assert.AreEqual("Hello", input[offsets[0].Start..offsets[0].End]);
        Assert.AreEqual(" there", input[offsets[1].Start..offsets[1].End]);
    }

    [TestMethod]
    public void Encode_Trim_OffsetsExcludeLeadingSpace()
    {
        // Arrange
        var tokenizer = CreateByteLevelTokenizer(trimOffsets: true, addPrefixSpace: false);
        var input = "Hello there";

        // Act
        var encoding = tokenizer.Encode(input);

        // Assert — trim removes leading Ġ (space) from "Ġthere"
        var offsets = encoding.GetOffsets();
        Assert.AreEqual(2, encoding.Length);
        Assert.AreEqual("Hello", encoding.GetTokens()[0]);
        Assert.AreEqual("Ġthere", encoding.GetTokens()[1]);
        Assert.AreEqual((0, 5), offsets[0]);   // "Hello" unchanged
        Assert.AreEqual((6, 11), offsets[1]);   // "there" (space trimmed)

        // 验证偏移映射回原文时不包含前导空格
        Assert.AreEqual("Hello", input[offsets[0].Start..offsets[0].End]);
        Assert.AreEqual("there", input[offsets[1].Start..offsets[1].End]);
    }

    [TestMethod]
    public void Encode_Trim_MultipleSpaces_AllTokensTrimmed()
    {
        // Arrange
        var tokenizer = CreateByteLevelTokenizer(trimOffsets: true, addPrefixSpace: false);
        var input = "Hello there, how are you?";

        // Act
        var encoding = tokenizer.Encode(input);

        // Assert — 所有非首 token 的前导空格应被去除
        var offsets = encoding.GetOffsets();
        for (int i = 1; i < offsets.Length; i++)
        {
            var (start, end) = offsets[i];
            var slice = input[start..end];
            Assert.IsFalse(slice.StartsWith(" "), $"Token {i} offset starts with space: '{slice}'");
        }
    }

    // ── 双序列偏移 ────────────────────────────────

    [TestMethod]
    public void EncodePair_OffsetsSpanBothSequences()
    {
        // Arrange
        var tokenizer = CreateByteLevelTokenizer(trimOffsets: false, addPrefixSpace: false);
        var inputA = "My name";
        var inputB = "is Anthony";

        // Act
        var encoding = tokenizer.EncodePair(inputA, inputB);

        // Assert — 偏移应覆盖两个序列的全部范围
        var offsets = encoding.GetOffsets();
        Assert.IsTrue(offsets.Length > 0);

        // 第一序列的 token 偏移应在 inputA 范围内
        var typeIds = encoding.GetTypeIds();
        for (int i = 0; i < offsets.Length; i++)
        {
            if (typeIds[i] == 0)
            {
                Assert.IsTrue(offsets[i].End <= inputA.Length,
                    $"Seq A token {i} offset ({offsets[i].End}) exceeds inputA length ({inputA.Length})");
            }
        }
    }

    [TestMethod]
    public void EncodePair_TypeIdsDistinguishSequences()
    {
        // Arrange
        var tokenizer = CreateByteLevelTokenizer(trimOffsets: false, addPrefixSpace: false);

        // Act
        var encoding = tokenizer.EncodePair("My name", "is Anthony");

        // Assert — typeId 应区分两个序列
        var typeIds = encoding.GetTypeIds();
        Assert.IsTrue(typeIds.Any(t => t == 0), "Should have tokens from sequence 0");
        Assert.IsTrue(typeIds.Any(t => t == 1), "Should have tokens from sequence 1");
    }

    [TestMethod]
    public void EncodePair_WordIdsDistinguishSequences()
    {
        // Arrange
        var tokenizer = CreateByteLevelTokenizer(trimOffsets: false, addPrefixSpace: false);

        // Act
        var encoding = tokenizer.EncodePair("My name", "is Anthony");

        // Assert — word IDs 在第二序列中应重新从 0 开始
        var wordIds = encoding.GetWordIds();
        var typeIds = encoding.GetTypeIds();

        // 找到第二序列第一个 token 的索引
        var firstSeqBIndex = Array.FindIndex(typeIds, t => t == 1);
        Assert.IsTrue(firstSeqBIndex >= 0, "Should have sequence B tokens");

        // 第二序列的 word ID 应从 0 或较小值开始（独立计数）
        var seqBWordIds = wordIds[firstSeqBIndex..]
            .Where(w => w.HasValue)
            .Select(w => w!.Value)
            .ToList();
        Assert.IsTrue(seqBWordIds.Min() == 0,
            "Word IDs for sequence B should start from 0");
    }

    // ── TokenToChars 在 trimming 后的映射 ─────────

    [TestMethod]
    public void TokenToChars_AfterTrim_MapsCorrectlyToOriginalInput()
    {
        // Arrange
        var tokenizer = CreateByteLevelTokenizer(trimOffsets: true, addPrefixSpace: false);
        var input = "Hello there";

        // Act
        var encoding = tokenizer.Encode(input);

        // Assert — 每个 token 的 TokenToChars 映射回原文都应是有效子串
        for (int i = 0; i < encoding.Length; i++)
        {
            var charOffsets = encoding.TokenToChars(i);
            Assert.IsNotNull(charOffsets, $"Token {i} should have char offsets");
            var (start, end) = charOffsets!.Value;
            Assert.IsTrue(start >= 0 && end <= input.Length && start <= end,
                $"Token {i} offsets ({start}, {end}) out of range for input length {input.Length}");
            var slice = input[start..end];
            Assert.IsTrue(slice.Length > 0, $"Token {i} maps to empty slice");
        }
    }
}
