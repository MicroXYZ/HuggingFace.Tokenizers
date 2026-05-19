using HuggingFace.Tokenizers.Abstractions;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Utf8Helpers 工具类的单元测试。
/// 验证 UTF-8 编码、indexMapping/byteEndMapping 构建、偏移转换的正确性。
/// </summary>
[TestClass]
public class Utf8HelpersTests
{
    // ─────────────────────────────────────────────
    //  EncodeToUtf8 — ASCII
    // ─────────────────────────────────────────────

    [TestMethod]
    public void EncodeToUtf8_Ascii_EachByteMapsToOwnChar()
    {
        Utf8Helpers.EncodeToUtf8("Hello", out byte[] utf8, out int[] mapping, out int[] endMapping, out int utf8Length);

        Assert.AreEqual(5, utf8Length);
        CollectionAssert.AreEqual(new byte[] { 0x48, 0x65, 0x6C, 0x6C, 0x6F }, utf8.AsSpan(0, utf8Length).ToArray());
        for (int i = 0; i < 5; i++)
        {
            Assert.AreEqual(i, mapping[i], $"mapping[{i}]");
            Assert.AreEqual(i + 1, endMapping[i], $"endMapping[{i}]");
        }

        ReturnArrays(utf8, mapping, endMapping);
    }

    // ─────────────────────────────────────────────
    //  EncodeToUtf8 — CJK (3 字节 UTF-8)
    // ─────────────────────────────────────────────

    [TestMethod]
    public void EncodeToUtf8_Cjk_ThreeBytesPerChar()
    {
        Utf8Helpers.EncodeToUtf8("你好", out byte[] utf8, out int[] mapping, out int[] endMapping, out int utf8Length);

        Assert.AreEqual(6, utf8Length);
        // 你: bytes 0,1,2 → char 0, end = 1
        for (int b = 0; b < 3; b++)
        {
            Assert.AreEqual(0, mapping[b]);
            Assert.AreEqual(1, endMapping[b]);
        }
        // 好: bytes 3,4,5 → char 1, end = 2
        for (int b = 3; b < 6; b++)
        {
            Assert.AreEqual(1, mapping[b]);
            Assert.AreEqual(2, endMapping[b]);
        }

        ReturnArrays(utf8, mapping, endMapping);
    }

    // ─────────────────────────────────────────────
    //  EncodeToUtf8 — Emoji (4 字节 UTF-8，代理对)
    // ─────────────────────────────────────────────

    [TestMethod]
    public void EncodeToUtf8_Emoji_FourBytesFromSurrogatePair()
    {
        Utf8Helpers.EncodeToUtf8("😀", out byte[] utf8, out int[] mapping, out int[] endMapping, out int utf8Length);

        Assert.AreEqual(4, utf8Length);
        // 所有 4 字节映射到 char 0（高代理），end = 2（高代理 + 低代理）
        for (int b = 0; b < 4; b++)
        {
            Assert.AreEqual(0, mapping[b], $"mapping[{b}]");
            Assert.AreEqual(2, endMapping[b], $"endMapping[{b}]");
        }

        ReturnArrays(utf8, mapping, endMapping);
    }

    // ─────────────────────────────────────────────
    //  EncodeToUtf8 — 混合文本
    // ─────────────────────────────────────────────

    [TestMethod]
    public void EncodeToUtf8_MixedText_CorrectMapping()
    {
        // "Hello你好😀"
        // char:  H(0) e(1) l(2) l(3) o(4) 你(5) 好(6) 😀高(7) 😀低(8)
        Utf8Helpers.EncodeToUtf8("Hello你好😀", out byte[] utf8, out int[] mapping, out int[] endMapping, out int utf8Length);

        Assert.AreEqual(5 + 6 + 4, utf8Length);

        // H: byte 0 → char 0, end 1
        Assert.AreEqual(0, mapping[0]); Assert.AreEqual(1, endMapping[0]);
        // e: byte 1 → char 1, end 2
        Assert.AreEqual(1, mapping[1]); Assert.AreEqual(2, endMapping[1]);
        // l: byte 2 → char 2, end 3
        Assert.AreEqual(2, mapping[2]); Assert.AreEqual(3, endMapping[2]);
        // l: byte 3 → char 3, end 4
        Assert.AreEqual(3, mapping[3]); Assert.AreEqual(4, endMapping[3]);
        // o: byte 4 → char 4, end 5
        Assert.AreEqual(4, mapping[4]); Assert.AreEqual(5, endMapping[4]);
        // 你: bytes 5,6,7 → char 5, end 6
        for (int b = 5; b < 8; b++) { Assert.AreEqual(5, mapping[b]); Assert.AreEqual(6, endMapping[b]); }
        // 好: bytes 8,9,10 → char 6, end 7
        for (int b = 8; b < 11; b++) { Assert.AreEqual(6, mapping[b]); Assert.AreEqual(7, endMapping[b]); }
        // 😀: bytes 11,12,13,14 → char 7, end 9
        for (int b = 11; b < 15; b++) { Assert.AreEqual(7, mapping[b]); Assert.AreEqual(9, endMapping[b]); }

        ReturnArrays(utf8, mapping, endMapping);
    }

    // ─────────────────────────────────────────────
    //  TryDecodeRune
    // ─────────────────────────────────────────────

    [TestMethod]
    public void TryDecodeRune_Ascii()
    {
        var utf8 = new byte[] { 0x48 };
        bool ok = Utf8Helpers.TryDecodeRune(utf8, 0, out var rune, out int consumed);
        Assert.IsTrue(ok);
        Assert.AreEqual('H', rune.Value);
        Assert.AreEqual(1, consumed);
    }

    [TestMethod]
    public void TryDecodeRune_Cjk()
    {
        var utf8 = new byte[] { 0xE4, 0xBD, 0xA0 };
        bool ok = Utf8Helpers.TryDecodeRune(utf8, 0, out var rune, out int consumed);
        Assert.IsTrue(ok);
        Assert.AreEqual(0x4F60, rune.Value);
        Assert.AreEqual(3, consumed);
    }

    [TestMethod]
    public void TryDecodeRune_Emoji()
    {
        var utf8 = new byte[] { 0xF0, 0x9F, 0x98, 0x80 };
        bool ok = Utf8Helpers.TryDecodeRune(utf8, 0, out var rune, out int consumed);
        Assert.IsTrue(ok);
        Assert.AreEqual(0x1F600, rune.Value);
        Assert.AreEqual(4, consumed);
    }

    // ─────────────────────────────────────────────
    //  边界情况：空字符串
    // ─────────────────────────────────────────────

    [TestMethod]
    public void EncodeToUtf8_EmptyString()
    {
        Utf8Helpers.EncodeToUtf8("", out byte[] utf8, out int[] mapping, out int[] endMapping, out int utf8Length);
        Assert.AreEqual(0, utf8Length);
    }

    // ─────────────────────────────────────────────
    //  与 ML.NET EncodeToUtf8 输出一致性
    // ─────────────────────────────────────────────

    [TestMethod]
    public void EncodeToUtf8_ConsistencyWithMlNet()
    {
        var testCases = new[]
        {
            "Hello",
            "你好",
            "😀",
            "Hello你好😀",
            "café",
            "👨‍👩‍👧‍👦",
            "𝌆",
        };

        foreach (var text in testCases)
        {
            Utf8Helpers.EncodeToUtf8(text, out byte[] ourUtf8, out int[] ourMapping, out int[] ourEndMapping, out int ourLen);

            int mlNetUtf8Length = System.Text.Encoding.UTF8.GetMaxByteCount(text.Length);
            byte[] mlNetUtf8 = new byte[mlNetUtf8Length];
            int[] mlNetMapping = new int[mlNetUtf8Length];
            int mlNetLen = MlNetEncodeToUtf8(text.AsSpan(), mlNetUtf8, mlNetMapping);

            Assert.AreEqual(mlNetLen, ourLen, $"UTF-8 length mismatch for '{text}'");
            for (int i = 0; i < ourLen; i++)
            {
                Assert.AreEqual(mlNetUtf8[i], ourUtf8[i], $"byte[{i}] mismatch for '{text}'");
                Assert.AreEqual(mlNetMapping[i], ourMapping[i], $"mapping[{i}] mismatch for '{text}'");
            }

            ReturnArrays(ourUtf8, ourMapping, ourEndMapping);
        }
    }

    // ─────────────────────────────────────────────
    //  辅助方法
    // ─────────────────────────────────────────────

    private static void ReturnArrays(byte[] utf8, int[] mapping, int[] endMapping)
    {
        // 数组已改为普通 new 分配（非 ArrayPool），无需归还。
    }

    /// <summary>
    /// ML.NET 的 EncodeToUtf8 参考实现（直接复制，用于一致性验证）。
    /// </summary>
    private static int MlNetEncodeToUtf8(ReadOnlySpan<char> text, Span<byte> destination, Span<int> indexMapping)
    {
        int targetIndex = 0;
        for (int i = 0; i < text.Length; i++)
        {
            uint c = (uint)text[i];
            if (c <= 0x7Fu)
            {
                destination[targetIndex] = (byte)c;
                indexMapping[targetIndex] = i;
                targetIndex++;
                continue;
            }
            if (c <= 0x7FFu)
            {
                destination[targetIndex] = (byte)((c + (0b110u << 11)) >> 6);
                destination[targetIndex + 1] = (byte)((c & 0x3Fu) + 0x80u);
                indexMapping[targetIndex] = indexMapping[targetIndex + 1] = i;
                targetIndex += 2;
                continue;
            }
            if (i < text.Length - 1 && char.IsSurrogatePair((char)c, text[i + 1]))
            {
                uint value = (uint)char.ConvertToUtf32((char)c, text[i + 1]);
                destination[targetIndex] = (byte)((value + (0b11110u << 21)) >> 18);
                destination[targetIndex + 1] = (byte)(((value & (0x3Fu << 12)) >> 12) + 0x80u);
                destination[targetIndex + 2] = (byte)(((value & (0x3Fu << 6)) >> 6) + 0x80u);
                destination[targetIndex + 3] = (byte)((value & 0x3Fu) + 0x80u);
                indexMapping[targetIndex] = indexMapping[targetIndex + 1] =
                    indexMapping[targetIndex + 2] = indexMapping[targetIndex + 3] = i;
                i++;
                targetIndex += 4;
                continue;
            }
            destination[targetIndex] = (byte)((c + (0b1110u << 16)) >> 12);
            destination[targetIndex + 1] = (byte)(((c & (0x3Fu << 6)) >> 6) + 0x80u);
            destination[targetIndex + 2] = (byte)((c & 0x3Fu) + 0x80u);
            indexMapping[targetIndex] = indexMapping[targetIndex + 1] = indexMapping[targetIndex + 2] = i;
            targetIndex += 3;
        }
        return targetIndex;
    }
}
