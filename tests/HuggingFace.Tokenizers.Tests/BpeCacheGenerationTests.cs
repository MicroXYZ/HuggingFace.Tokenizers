using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Models.BPE;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// BPE 缓存 generation-based O(1) 清除测试。
/// 验证 ClearCache() 通过 generation 递增使缓存失效，而非遍历清除。
/// </summary>
    [TestClass]
public class BpeCacheGenerationTests
{
    /// <summary>
    /// 构建带缓存的 BPE 模型。
    /// </summary>
    private static BpeModel CreateCachedModel(int cacheCapacity = 1000)
    {
        var vocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0,
            ["hello"] = 1,
            ["world"] = 2,
            [" "] = 3,
            ["!"] = 4
        };
        var merges = new List<(string, string)>();

        return new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges(merges)
            .SetUnkToken("<unk>")
            .SetCacheCapacity(cacheCapacity)
            .Build();
    }

    [TestMethod]
    public void ClearCache_不抛异常()
    {
        var model = CreateCachedModel();
        model.Tokenize("hello"); // 填充缓存
        model.ClearCache();      // O(1) generation 递增
    }

    [TestMethod]
    public void ClearCache_后重新计算结果()
    {
        var model = CreateCachedModel();
        var input = "hello world";

        var before = model.Tokenize(input);
        model.ClearCache();
        var after = model.Tokenize(input);

        // ClearCache 后结果应一致（确定性算法）
        Assert.AreEqual(before.Count, after.Count);
        for (int i = 0; i < before.Count; i++)
        {
            Assert.AreEqual(before[i].Id, after[i].Id);
            Assert.AreEqual(before[i].Value, after[i].Value);
        }
    }

    [TestMethod]
    public void 多次ClearCache_不影响正确性()
    {
        var model = CreateCachedModel();
        var input = "hello world!";

        for (int i = 0; i < 10; i++)
        {
            model.ClearCache();
            var tokens = model.Tokenize(input);
            Assert.IsTrue(tokens.Count() > 0);
        }
    }

    [TestMethod]
    public void 缓存命中_结果一致()
    {
        var model = CreateCachedModel();
        var input = "hello";

        var first = model.Tokenize(input);
        var second = model.Tokenize(input); // 应命中缓存

        Assert.AreEqual(first.Count, second.Count);
        for (int i = 0; i < first.Count; i++)
        {
            Assert.AreEqual(first[i].Id, second[i].Id);
        }
    }

    [TestMethod]
    public void ClearCache_后缓存未命中_重新计算()
    {
        var model = CreateCachedModel();
        var input = "hello";

        var first = model.Tokenize(input);
        model.ClearCache();
        var afterClear = model.Tokenize(input);

        // 结果应一致
        Assert.AreEqual(first.Count, afterClear.Count);
        for (int i = 0; i < first.Count; i++)
        {
            Assert.AreEqual(first[i].Id, afterClear[i].Id);
        }
    }

    [TestMethod]
    public void Rebuild_清除缓存()
    {
        var model = CreateCachedModel();
        var oldTokens = model.Tokenize("hello");
        Assert.IsTrue(oldTokens.Count() > 0);

        // Rebuild 应自动清除缓存，使用新的词表
        // 原模型有 unkToken="<unk>"，Rebuild 保持配置不变
        var newVocab = new Dictionary<string, uint>
        {
            ["<unk>"] = 0,
            ["hello"] = 1,
            ["world"] = 2,
            [" "] = 3,
            ["!"] = 4
        };
        model.Rebuild(newVocab, new List<(string, string)>());

        // Rebuild 后 "hello" 在词表中，但 BPE 无合并规则时
        // 会按字符拆分。"h" 不在新词表中 → 走 unk 路径
        // 验证 Rebuild 后缓存被清除（不会返回旧缓存结果）
        var tokens = model.Tokenize("hello");
        Assert.IsTrue(tokens.Count() > 0);
        // 第一个 token 应该是 unk（因为 "h" 不在词表中）
        Assert.AreEqual(0u, tokens[0].Id);
    }

    [TestMethod]
    public void CacheCapacity_正确报告()
    {
        var model = CreateCachedModel(cacheCapacity: 500);
        Assert.AreEqual(500, model.CacheCapacity);
    }
}
