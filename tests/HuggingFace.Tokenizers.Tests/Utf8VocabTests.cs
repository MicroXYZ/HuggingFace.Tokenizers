using HuggingFace.Tokenizers.Internal;
using System.Text;

namespace HuggingFace.Tokenizers.Tests;

[TestClass]
public class Utf8VocabTests
{
    [TestMethod]
    public void TryGetId_ByString_FindsToken()
    {
        var vocab = new Utf8Vocab(new Dictionary<string, uint>
        {
            ["hello"] = 1, ["world"] = 2, ["<unk>"] = 0
        });

        Assert.IsTrue(vocab.TryGetId("hello".AsSpan(), out var id));
        Assert.AreEqual(1u, id);
    }

    [TestMethod]
    public void TryGetId_ByUtf8Bytes_FindsToken()
    {
        var vocab = new Utf8Vocab(new Dictionary<string, uint>
        {
            ["hello"] = 1, ["world"] = 2
        });

        var bytes = Encoding.UTF8.GetBytes("world");
        Assert.IsTrue(vocab.TryGetId(bytes, out var id));
        Assert.AreEqual(2u, id);
    }

    [TestMethod]
    public void TryGetId_NonExistent_ReturnsFalse()
    {
        var vocab = new Utf8Vocab(new Dictionary<string, uint>
        {
            ["hello"] = 1
        });

        Assert.IsFalse(vocab.TryGetId("missing".AsSpan(), out _));
    }

    [TestMethod]
    public void TryGetId_UnicodeToken_Works()
    {
        var vocab = new Utf8Vocab(new Dictionary<string, uint>
        {
            ["你好"] = 10, ["世界"] = 20
        });

        Assert.IsTrue(vocab.TryGetId("你好".AsSpan(), out var id));
        Assert.AreEqual(10u, id);

        var bytes = Encoding.UTF8.GetBytes("世界");
        Assert.IsTrue(vocab.TryGetId(bytes, out id));
        Assert.AreEqual(20u, id);
    }

    [TestMethod]
    public void GetTokenString_ValidId_ReturnsToken()
    {
        var vocab = new Utf8Vocab(new Dictionary<string, uint>
        {
            ["hello"] = 1
        });

        Assert.AreEqual("hello", vocab.GetTokenString(1));
    }

    [TestMethod]
    public void GetTokenString_UnknownId_ReturnsPlaceholder()
    {
        var vocab = new Utf8Vocab(new Dictionary<string, uint>
        {
            ["hello"] = 1
        });

        Assert.AreEqual("<99>", vocab.GetTokenString(99));
    }

    [TestMethod]
    public void Count_ReturnsCorrectCount()
    {
        var vocab = new Utf8Vocab(new Dictionary<string, uint>
        {
            ["a"] = 0, ["b"] = 1, ["c"] = 2
        });

        Assert.AreEqual(3, vocab.Count);
    }

    [TestMethod]
    public void LargeVocab_AllTokensFindable()
    {
        var dict = new Dictionary<string, uint>();
        for (int i = 0; i < 1000; i++)
            dict[$"token_{i}"] = (uint)i;

        var vocab = new Utf8Vocab(dict);

        for (int i = 0; i < 1000; i++)
        {
            Assert.IsTrue(vocab.TryGetId($"token_{i}".AsSpan(), out var id), $"token_{i} not found");
            Assert.AreEqual((uint)i, id);
        }
    }

    [TestMethod]
    public void EmptyVocab_NoTokens()
    {
        var vocab = new Utf8Vocab(new Dictionary<string, uint>());
        Assert.IsFalse(vocab.TryGetId("anything".AsSpan(), out _));
        Assert.AreEqual(0, vocab.Count);
    }
}
