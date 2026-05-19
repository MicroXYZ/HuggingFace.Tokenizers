using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Models.BPE;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// BPE Dropout 功能测试。
/// 验证 dropout 参数在分词过程中正确生效。
/// </summary>
    [TestClass]
public class BpeDropoutTests
{
    /// <summary>
    /// 构建一个可执行合并的 BPE 模型。
    /// 词表包含字符 "a","b","c" 及其合并结果 "ab","abc"。
    /// 合并规则：a+b→ab, ab+c→abc。
    /// </summary>
    private static BpeModel CreateMergableModel(float dropout = 0f)
    {
        var vocab = new Dictionary<string, uint>
        {
            ["a"] = 0,
            ["b"] = 1,
            ["c"] = 2,
            ["ab"] = 3,
            ["abc"] = 4
        };
        var merges = new List<(string, string)>
        {
            ("a", "b"),   // rank 0: a + b → ab
            ("ab", "c")   // rank 1: ab + c → abc
        };

        return new BpeModel.BpeBuilder()
            .SetVocab(vocab)
            .SetMerges(merges)
            .SetDropout(dropout)
            .Build();
    }

    [TestMethod]
    public void Tokenize_DropoutZero_结果确定性()
    {
        // dropout=0 时，相同输入应始终产生相同的 token 序列（确定性行为）
        var model = CreateMergableModel(dropout: 0f);
        var input = "abc";

        // 多次运行，结果应完全一致
        var firstRun = model.Tokenize(input);
        for (int i = 0; i < 50; i++)
        {
            var tokens = model.Tokenize(input);
            Assert.AreEqual(firstRun.Count, tokens.Count);
            for (int j = 0; j < firstRun.Count; j++)
            {
                Assert.AreEqual(firstRun[j].Id, tokens[j].Id);
                Assert.AreEqual(firstRun[j].Value, tokens[j].Value);
            }
        }

        // dropout=0 应执行所有合并，最终结果为单个 "abc" token
        Assert.AreEqual(1, firstRun.Count());
        Assert.AreEqual(4u, firstRun[0].Id); // "abc" 的 ID
    }

    [TestMethod]
    public void Tokenize_DropoutOne_不做任何合并()
    {
        // dropout=1.0 时，每个合并对都有 100% 概率被跳过，
        // 因此不应产生任何合并，结果为单字符 token 序列。
        var model = CreateMergableModel(dropout: 1.0f);
        var input = "abc";

        // 多次运行验证：dropout=1.0 时结果应始终为三个单独字符
        // 注意：由于 Random 的随机性，理论上 dropout=1.0 时所有合并被跳过，
        // 但为了鲁棒性，多次运行均应得到三字符结果。
        for (int attempt = 0; attempt < 20; attempt++)
        {
            var tokens = model.Tokenize(input);
            Assert.AreEqual(3, tokens.Count);
            Assert.AreEqual("a", tokens[0].Value);
            Assert.AreEqual("b", tokens[1].Value);
            Assert.AreEqual("c", tokens[2].Value);
        }
    }

    [TestMethod]
    public void Tokenize_DropoutMidRange_结果可能不同()
    {
        // dropout 在 0-1 之间时，合并操作会随机跳过，
        // 多次运行应至少产生一次不同的结果。
        var model = CreateMergableModel(dropout: 0.5f);
        var input = "abc";

        var allSame = true;
        var firstResult = model.Tokenize(input);

        for (int i = 0; i < 100; i++)
        {
            var tokens = model.Tokenize(input);
            if (tokens.Count != firstResult.Count ||
                !tokens.Zip(firstResult).All(pair => pair.First.Id == pair.Second.Id))
            {
                allSame = false;
                break;
            }
        }

        // 100 次运行中应至少有一次结果不同（概率极低全部相同）
        Assert.IsFalse(allSame, "dropout=0.5 时，100 次运行应至少产生一次不同的分词结果");
    }
}
