using Microsoft.VisualStudio.TestTools.UnitTesting;
using HuggingFace.Tokenizers.Models.Unigram;

namespace HuggingFace.Tokenizers.Tests;

/// <summary>
/// Tests for Lattice — matching Rust lattice.rs test cases.
/// </summary>
    [TestClass]
public class LatticeTests
{
    [TestMethod]
    public void SetSentence_EmptyString()
    {
        var lattice = new Lattice("", 1, 2);
        Assert.AreEqual(0, lattice.Len);
        Assert.AreEqual("", lattice.Sentence);
        Assert.AreEqual("", lattice.Surface(0));
    }

    [TestMethod]
    public void SetSentence_SimpleString()
    {
        var lattice = new Lattice("test", 1, 2);
        Assert.AreEqual(4, lattice.Len);
        Assert.AreEqual("test", lattice.Sentence);
        Assert.AreEqual("test", lattice.Surface(0));
        Assert.AreEqual("est", lattice.Surface(1));
        Assert.AreEqual("st", lattice.Surface(2));
        Assert.AreEqual("t", lattice.Surface(3));

        Assert.AreEqual(1, lattice.BosNode.Id);
        Assert.AreEqual(2, lattice.EosNode.Id);
    }

    [TestMethod]
    public void SetSentence_UnicodeString()
    {
        var lattice = new Lattice("テストab", 1, 2);
        Assert.AreEqual(5, lattice.Len); // テ=1 ス=1 ト=1 a=1 b=1 → 5 UTF-16 chars
        Assert.AreEqual("テストab", lattice.Sentence);
        Assert.AreEqual("テストab", lattice.Surface(0));
    }

    [TestMethod]
    public void Insert_BasicNodes()
    {
        var lattice = new Lattice("ABあい", 1, 2);

        lattice.Insert(0, 1, 0.0, 3);
        lattice.Insert(1, 1, 0.0, 4);
        lattice.Insert(2, 1, 0.0, 5); // あ (1 UTF-16 char)
        lattice.Insert(3, 1, 0.0, 6); // い (1 UTF-16 char)
        lattice.Insert(0, 2, 0.0, 7); // AB
        lattice.Insert(1, 3, 0.0, 8); // Bあい (wrong length but tests structure)

        // BOS + EOS + 6 inserted = 8 nodes
        Assert.AreEqual(8, lattice.Nodes.Count);

        // Check BOS/EOS
        Assert.AreEqual(1, lattice.BosNode.Id);
        Assert.AreEqual(2, lattice.EosNode.Id);
    }

    [TestMethod]
    public void Viterbi_BasicPath()
    {
        var lattice = new Lattice("ABC", 1, 2);

        // Insert individual character nodes
        lattice.Insert(0, 1, 0.0, 3); // A
        lattice.Insert(1, 1, 0.0, 4); // B
        lattice.Insert(2, 1, 0.0, 5); // C

        var path = lattice.Viterbi();
        Assert.AreEqual(3, path.Count);
        Assert.AreEqual("A", lattice.Piece(path[0]));
        Assert.AreEqual("B", lattice.Piece(path[1]));
        Assert.AreEqual("C", lattice.Piece(path[2]));
    }

    [TestMethod]
    public void Viterbi_PrefersHigherScore()
    {
        var lattice = new Lattice("ABC", 1, 2);

        // Individual chars with score 0
        lattice.Insert(0, 1, 0.0, 3); // A
        lattice.Insert(1, 1, 0.0, 4); // B
        lattice.Insert(2, 1, 0.0, 5); // C

        // AB with higher score
        lattice.Insert(0, 2, 2.0, 6); // AB
        CollectionAssert.AreEqual(new[] { "AB", "C" }, lattice.Tokens().ToArray());

        // BC with even higher score
        lattice.Insert(1, 2, 5.0, 7); // BC
        CollectionAssert.AreEqual(new[] { "A", "BC" }, lattice.Tokens().ToArray());

        // ABC with highest score
        lattice.Insert(0, 3, 10.0, 8); // ABC
        CollectionAssert.AreEqual(new[] { "ABC" }, lattice.Tokens().ToArray());
    }

    [TestMethod]
    public void Viterbi_IncompleteLattice_ReturnsEmpty()
    {
        var lattice = new Lattice("ABC", 1, 2);
        // No nodes inserted — can't form a path
        Assert.AreEqual(0, lattice.Viterbi().Count());
    }

    [TestMethod]
    public void Nbest_ReturnsMultiplePaths()
    {
        var lattice = new Lattice("ABC", 1, 2);

        lattice.Insert(0, 1, 0.0, 3); // A
        lattice.Insert(1, 1, 0.0, 4); // B
        lattice.Insert(2, 1, 0.0, 5); // C
        lattice.Insert(0, 2, 2.0, 6); // AB
        lattice.Insert(1, 2, 5.0, 7); // BC
        lattice.Insert(0, 3, 10.0, 8); // ABC

        var nbests = lattice.NbestTokens(10);
        Assert.AreEqual(4, nbests.Count);
        CollectionAssert.AreEqual(new[] { "ABC" }, nbests[0].ToArray());
        CollectionAssert.AreEqual(new[] { "A", "BC" }, nbests[1].ToArray());
        CollectionAssert.AreEqual(new[] { "AB", "C" }, nbests[2].ToArray());
        CollectionAssert.AreEqual(new[] { "A", "B", "C" }, nbests[3].ToArray());

        Assert.AreEqual(0, lattice.NbestTokens(0).Count());
        Assert.AreEqual(1, lattice.NbestTokens(1).Count());
    }

    [TestMethod]
    public void Sample_ReturnsValidPath()
    {
        var lattice = new Lattice("ABC", 1, 2);

        lattice.Insert(0, 1, 0.0, 3);
        lattice.Insert(1, 1, 0.0, 4);
        lattice.Insert(2, 1, 0.0, 5);
        lattice.Insert(0, 2, 2.0, 6);
        lattice.Insert(1, 2, 5.0, 7);

        // Sample with low temperature (should prefer high-score paths)
        var sample = lattice.Sample(0.1);
        Assert.IsTrue(sample.Count() > 0);

        // All sampled nodes should be valid
        foreach (var node in sample)
        {
            Assert.IsTrue(node.Pos >= 0);
            Assert.IsTrue(node.Pos + node.Length <= lattice.Len);
        }
    }

    [TestMethod]
    public void SampleToken_ReturnsStrings()
    {
        var lattice = new Lattice("ABC", 1, 2);

        lattice.Insert(0, 1, 0.0, 3);
        lattice.Insert(1, 1, 0.0, 4);
        lattice.Insert(2, 1, 0.0, 5);

        var tokens = lattice.SampleToken(1.0);
        Assert.IsTrue(tokens.Count() > 0);
        // Should reconstruct the original string
        Assert.AreEqual("ABC", string.Join("", tokens));
    }

    [TestMethod]
    public void PopulateMarginal_ComputesExpectedCounts()
    {
        var lattice = new Lattice("ABC", 1, 2);

        lattice.Insert(0, 1, 1.0, 3);  // A
        lattice.Insert(1, 1, 1.2, 4);  // B
        lattice.Insert(2, 1, 2.5, 5);  // C
        lattice.Insert(0, 2, 3.0, 6);  // AB
        lattice.Insert(1, 2, 4.0, 7);  // BC
        lattice.Insert(0, 3, 2.0, 8);  // ABC

        // 9 nodes: BOS, EOS, A, B, C, AB, BC, ABC + one more from Insert
        // Actually: 2 sentinels + 6 inserted = 8
        var expected = new double[9]; // IDs go up to 8
        double logZ = lattice.PopulateMarginal(1.0, expected);

        // logZ should be > 0
        Assert.IsTrue(logZ > 0);

        // IDs 0,1 are BOS/EOS — their expected counts should be 0
        Assert.AreEqual(0.0, expected[0]);
        Assert.AreEqual(0.0, expected[1]);
        Assert.AreEqual(0.0, expected[2]);

        // Token IDs 3-8 should have positive expected counts
        Assert.IsTrue(expected[3] > 0); // A
        Assert.IsTrue(expected[4] > 0); // B
        Assert.IsTrue(expected[5] > 0); // C
        Assert.IsTrue(expected[6] > 0); // AB
        Assert.IsTrue(expected[7] > 0); // BC
        Assert.IsTrue(expected[8] > 0); // ABC
    }

    [TestMethod]
    public void Clear_ResetsLattice()
    {
        var lattice = new Lattice("test", 1, 2);
        lattice.Insert(0, 1, 0.0, 3);
        Assert.AreEqual(3, lattice.Nodes.Count); // BOS + EOS + 1

        lattice.Clear();
        Assert.AreEqual(2, lattice.Nodes.Count); // BOS + EOS only
        Assert.AreEqual(1, lattice.BosNode.Id);
        Assert.AreEqual(2, lattice.EosNode.Id);
    }
}
