using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// RuneHelpers 直接单元测试。
/// </summary>
[TestClass]
public class RuneHelpersTests
{
    [TestMethod]
    public void FormatByteFallbackToken_ValidByte_ReturnsHexFormat()
    {
        // 0x41 = 'A'
        var result = RuneHelpers.FormatByteFallbackToken(0x41);
        Assert.AreEqual("<0x41>", result);
    }

    [TestMethod]
    public void FormatByteFallbackToken_Zero_ReturnsHexFormat()
    {
        var result = RuneHelpers.FormatByteFallbackToken(0x00);
        Assert.AreEqual("<0x00>", result);
    }

    [TestMethod]
    public void FormatByteFallbackToken_MaxByte_ReturnsHexFormat()
    {
        var result = RuneHelpers.FormatByteFallbackToken(0xFF);
        Assert.AreEqual("<0xFF>", result);
    }

    [TestMethod]
    public void TryParseByteFallbackToken_ValidFormat_ReturnsTrue()
    {
        bool result = RuneHelpers.TryParseByteFallbackToken("<0x41>", out byte b);
        Assert.IsTrue(result);
        Assert.AreEqual(0x41, b);
    }

    [TestMethod]
    public void TryParseByteFallbackToken_InvalidFormat_ReturnsFalse()
    {
        bool result = RuneHelpers.TryParseByteFallbackToken("hello", out byte b);
        Assert.IsFalse(result);
    }

    [TestMethod]
    public void TryParseByteFallbackToken_EmptyString_ReturnsFalse()
    {
        bool result = RuneHelpers.TryParseByteFallbackToken("", out byte b);
        Assert.IsFalse(result);
    }

    [TestMethod]
    public void LogSumExp_TwoValues_ReturnsCorrectResult()
    {
        // log(exp(1) + exp(2)) ≈ 2.3133
        double result = RuneHelpers.LogSumExp(1.0, 2.0);
        Assert.AreEqual(2.3133, result, 0.001);
    }

    [TestMethod]
    public void LogSumExp_SameValues_ReturnsSamePlusLog2()
    {
        // log(exp(x) + exp(x)) = x + log(2)
        double x = 5.0;
        double result = RuneHelpers.LogSumExp(x, x);
        Assert.AreEqual(x + Math.Log(2), result, 1e-10);
    }

    [TestMethod]
    public void LogSumExp_OneVerySmall_ReturnsLarger()
    {
        // log(exp(-1000) + exp(1)) ≈ 1
        double result = RuneHelpers.LogSumExp(-1000.0, 1.0);
        Assert.AreEqual(1.0, result, 0.001);
    }
}
