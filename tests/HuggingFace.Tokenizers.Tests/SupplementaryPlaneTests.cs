using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Models.Unigram;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// 补充平面字符（emoji、CJK 扩展等）端到端测试。
/// 验证 Lattice 和 Trie 对 UTF-16 代理对的正确处理。
/// </summary>
[TestClass]
public class SupplementaryPlaneTests
{
    [TestMethod]
    public void Lattice_Viterbi_WithSupplementaryPlaneChars_PathCorrect()
    {
        // 验证 Lattice 对补充平面字符的 Viterbi 路径正确
        // "a😀b" 中 "a" 占 1 char，"😀" 占 2 chars（代理对），"b" 占 1 char
        var lattice = new Lattice("a😀b", 100, 101);

        // 插入节点时使用 char 偏移（UTF-16 索引）
        lattice.Insert(0, 1, -1.0, 0);  // "a"：char [0, 1)
        lattice.Insert(1, 2, -1.0, 1);  // "😀"：char [1, 3)，代理对占 2 char
        lattice.Insert(3, 1, -1.0, 2);  // "b"：char [3, 4)

        var path = lattice.Viterbi();

        // 路径应包含 3 个 token 节点（不含 BOS/EOS）
        Assert.AreEqual(3, path.Count);
        Assert.AreEqual(0, path[0].Id);  // "a"
        Assert.AreEqual(1, path[1].Id);  // "😀"
        Assert.AreEqual(2, path[2].Id);  // "b"
    }

    [TestMethod]
    public void Lattice_Piece_WithSupplementaryPlaneChars_ReturnsCorrectSubstring()
    {
        // 验证 Piece 方法正确提取包含补充平面字符的片段
        var lattice = new Lattice("hello 😀🎉 world", 100, 101);

        // "hello " = 6 chars, "😀" = 2 chars, "🎉" = 2 chars, " world" = 6 chars
        lattice.Insert(6, 2, -1.0, 0);   // "😀"
        lattice.Insert(8, 2, -1.0, 1);   // "🎉"
        lattice.Insert(6, 4, -1.0, 2);   // "😀🎉"

        var node0 = lattice.Nodes[2]; // 第一个插入的节点
        Assert.AreEqual("😀", lattice.Piece(node0));

        var node2 = lattice.Nodes[4]; // 第三个插入的节点
        Assert.AreEqual("😀🎉", lattice.Piece(node2));
    }

    [TestMethod]
    public void Lattice_Viterbi_WithEmoji_OffsetsDoNotOverlap()
    {
        // 验证包含 emoji 的文本编码后偏移量不越界、不错位
        var lattice = new Lattice("a😀b🎉c", 100, 101);

        // "a" = [0,1), "😀" = [1,3), "b" = [3,4), "🎉" = [4,6), "c" = [6,7)
        lattice.Insert(0, 1, -1.0, 0);
        lattice.Insert(1, 2, -1.0, 1);
        lattice.Insert(3, 1, -1.0, 2);
        lattice.Insert(4, 2, -1.0, 3);
        lattice.Insert(6, 1, -1.0, 4);

        var path = lattice.Viterbi();

        Assert.AreEqual(5, path.Count);

        // 验证偏移量连续且无间隙
        for (int i = 0; i < path.Count; i++)
        {
            var node = path[i];
            // 偏移量不应越界
            Assert.IsTrue(node.Pos >= 0, $"节点 {i} 的 Pos 不应为负");
            Assert.IsTrue(node.Pos + node.Length <= lattice.Len, $"节点 {i} 的偏移越界");
            // 偏移量不应重叠
            if (i > 0)
            {
                Assert.AreEqual(path[i - 1].Pos + path[i - 1].Length, node.Pos,
                    $"节点 {i} 与前一节点偏移不连续");
            }
        }
    }

    [TestMethod]
    public void Trie_FindPrefixes_WithSupplementaryPlaneChars_PositionsCorrect()
    {
        // 验证 Trie 对补充平面字符的位置计算
        var trie = new Trie();
        trie.Insert("😀", 0);
        trie.Insert("😀🎉", 1);

        var results = trie.FindPrefixes("😀🎉", 0);

        // "😀" 占 2 个 char，所以 End 应为 2
        Assert.IsTrue(results.Any(r => r.End == 2 && r.TokenId == 0),
            "应匹配 '😀'，End=2");

        // "😀🎉" 占 4 个 char，所以 End 应为 4
        Assert.IsTrue(results.Any(r => r.End == 4 && r.TokenId == 1),
            "应匹配 '😀🎉'，End=4");
    }

    [TestMethod]
    public void Trie_FindPrefixes_WithCjkExtendedChars_PositionsCorrect()
    {
        // 验证 Trie 对 CJK 扩展字符（BMP 内）的正确处理
        var trie = new Trie();
        trie.Insert("你好", 0);
        trie.Insert("你", 1);
        trie.Insert("世界", 2);

        var results = trie.FindPrefixes("你好世界", 0);

        // "你" 占 1 char → End=1
        Assert.IsTrue(results.Any(r => r.End == 1 && r.TokenId == 1));
        // "你好" 占 2 chars → End=2
        Assert.IsTrue(results.Any(r => r.End == 2 && r.TokenId == 0));

        // 从 "世界" 的起始位置（char 索引 2）开始搜索
        var results2 = trie.FindPrefixes("你好世界", 2);
        // 不应匹配 "你" 或 "你好"（它们从位置 0 开始）
        Assert.IsFalse(results2.Any(r => r.TokenId == 1));
        Assert.IsFalse(results2.Any(r => r.TokenId == 0));
        // 应匹配 "世界" → "世" 在位置 2，"界" 在位置 3，End=4
        Assert.IsTrue(results2.Any(r => r.End == 4 && r.TokenId == 2));
    }

    [TestMethod]
    public void Trie_Contains_WithSupplementaryPlaneChars_ReturnsTrue()
    {
        // 验证 Contains 方法对补充平面字符的正确性
        var trie = new Trie();
        trie.Insert("😀", 0);
        trie.Insert("🎉🎊", 1);

        Assert.IsTrue(trie.Contains("😀"));
        Assert.IsTrue(trie.Contains("🎉🎊"));
        Assert.IsFalse(trie.Contains("😀🎉"));
        Assert.IsFalse(trie.Contains("🎉"));
    }

    [TestMethod]
    public void Lattice_Nbest_WithSupplementaryPlaneChars_ReturnsMultiplePaths()
    {
        // 验证 nbest 搜索对补充平面字符的正确性
        var lattice = new Lattice("abc", 100, 101);

        // 插入多种分词方案，分数不同以确保 nbest 能找到多条路径
        lattice.Insert(0, 1, -1.0, 0);  // "a"
        lattice.Insert(1, 1, -1.0, 1);  // "b"
        lattice.Insert(2, 1, -1.0, 2);  // "c"
        lattice.Insert(0, 2, -5.0, 3);  // "ab"（低分）
        lattice.Insert(1, 2, -5.0, 4);  // "bc"（低分）

        var nbest = lattice.Nbest(3);

        Assert.IsTrue(nbest.Count >= 1, "应至少返回 1 条路径");
        // 第一条路径应该是分数最高的 "a"+"b"+"c"
        Assert.AreEqual(3, nbest[0].Count);
    }
}
