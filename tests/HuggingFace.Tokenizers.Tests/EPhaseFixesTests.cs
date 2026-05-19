using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Decoders;
using HuggingFace.Tokenizers.Processors;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// 阶段 E 逻辑错误修复的测试覆盖。
/// </summary>
[TestClass]
public class EPhaseFixesTests
{
    // ── E.1: SpecialTokenPiece 空集合检查 ──────────────────────────

    [TestMethod]
    public void SpecialTokenPiece_EmptyTokenIds_ThrowsArgumentException()
    {
        Assert.ThrowsExactly<ArgumentException>(() =>
            Template.Special(Array.Empty<uint>(), Array.Empty<string>()));
    }

    [TestMethod]
    public void SpecialTokenPiece_MismatchedLengths_ThrowsArgumentException()
    {
        Assert.ThrowsExactly<ArgumentException>(() =>
            Template.Special(new uint[] { 1, 2 }, new string[] { "[CLS]" }));
    }

    [TestMethod]
    public void SpecialTokenPiece_ValidSingle_CreatesSuccessfully()
    {
        var piece = Template.Special(101, "[CLS]");
        Assert.AreEqual(1, piece.TokenIds.Count);
        Assert.AreEqual(101u, piece.TokenIds[0]);
    }

    [TestMethod]
    public void SpecialTokenPiece_ValidMulti_CreatesSuccessfully()
    {
        var piece = Template.Special(new uint[] { 101, 102 }, new string[] { "[CLS]", "[PAD]" });
        Assert.AreEqual(2, piece.TokenIds.Count);
    }

    // ── E.2: ByteLevelDecoder Decode 与 DecodeChain 多字节一致性 ──

    [TestMethod]
    public void ByteLevelDecoder_MultiByteConsistency()
    {
        var decoder = new ByteLevelDecoder();
        // 包含多字节 UTF-8 字符的 token
        var tokens = new List<string> { "Hello", "world" };
        var chainResult = decoder.DecodeChain(tokens);
        var decodeResult = decoder.Decode(tokens);
        // Decode 应等于 DecodeChain 拼接
        Assert.AreEqual(string.Join("", chainResult), decodeResult);
    }

    [TestMethod]
    public void ByteLevelDecoder_EmptyTokens_Consistency()
    {
        var decoder = new ByteLevelDecoder();
        var tokens = new List<string> { "", "a", "" };
        var chainResult = decoder.DecodeChain(tokens);
        var decodeResult = decoder.Decode(tokens);
        Assert.AreEqual(string.Join("", chainResult), decodeResult);
    }

    // ── E.7: NormalizedString.Prepend 空字符串对齐 ─────────────────

    [TestMethod]
    public void NormalizedString_PrependEmpty_DoesNotThrow()
    {
        var ns = new NormalizedString("hello");
        // Prepend 空字符串不应改变内容
        ns.Prepend("");
        Assert.AreEqual("hello", ns.Get());
    }

    [TestMethod]
    public void NormalizedString_PrependToEmpty_SetsContent()
    {
        var ns = new NormalizedString("");
        ns.Prepend("prefix");
        Assert.AreEqual("prefix", ns.Get());
    }

    // ── E.8: Encoding.MergeWith 溢出防护 ──────────────────────────

    [TestMethod]
    public void Encoding_MergeWith_EmptyOverflowing_MergesCorrectly()
    {
        var enc1 = new Encoding(
            new uint[] { 1, 2 }, new uint[] { 0, 0 },
            new string[] { "a", "b" }, new uint?[] { 0, 0 },
            new (int, int)[] { (0, 1), (1, 2) },
            new uint[] { 0, 0 }, new uint[] { 1, 1 });

        var enc2 = new Encoding(
            new uint[] { 3 }, new uint[] { 0 },
            new string[] { "c" }, new uint?[] { 0 },
            new (int, int)[] { (0, 1) },
            new uint[] { 0 }, new uint[] { 1 });

        enc1.MergeWith(enc2, false);

        Assert.AreEqual(3, enc1.Length);
        Assert.AreEqual(1u, enc1.GetIds()[0]);
        Assert.AreEqual(3u, enc1.GetIds()[2]);
    }

    [TestMethod]
    public void Encoding_MergeWith_WithOverflowing_CrossProduct()
    {
        // 测试溢出编码的叉积合并
        var inner1 = new Encoding(
            new uint[] { 10 }, new uint[] { 0 },
            new string[] { "x" }, new uint?[] { 0 },
            new (int, int)[] { (0, 1) },
            new uint[] { 0 }, new uint[] { 1 });

        var inner2 = new Encoding(
            new uint[] { 20 }, new uint[] { 0 },
            new string[] { "y" }, new uint?[] { 0 },
            new (int, int)[] { (0, 1) },
            new uint[] { 0 }, new uint[] { 1 });

        var enc1 = new Encoding(
            new uint[] { 1 }, new uint[] { 0 },
            new string[] { "a" }, new uint?[] { 0 },
            new (int, int)[] { (0, 1) },
            new uint[] { 0 }, new uint[] { 1 },
            new List<Encoding> { inner1 });

        var enc2 = new Encoding(
            new uint[] { 2 }, new uint[] { 0 },
            new string[] { "b" }, new uint?[] { 0 },
            new (int, int)[] { (0, 1) },
            new uint[] { 0 }, new uint[] { 1 },
            new List<Encoding> { inner2 });

        enc1.MergeWith(enc2, false);

        // 叉积：self.overflow + pair, self.overflow + pair.overflow, self + pair.overflow
        Assert.AreEqual(3, enc1.GetOverflowing().Count);
    }
}
