// Based on Microsoft.ML.Tokenizers DoubleArrayTrie (MIT License)
// Original: https://github.com/dotnet/machinelearning

namespace HuggingFace.Tokenizers.Normalizers;

/// <summary>
/// 双数组字典树公共前缀搜索的结果对。
/// </summary>
internal readonly struct DoubleArrayResultPair
{
    public int Length { get; }
    public int Value { get; }

    public DoubleArrayResultPair(int length, int value)
    {
        Length = length;
        Value = value;
    }
}

/// <summary>
/// 双数组字典树的单元。每个单元包含 base 和 check 值。
/// </summary>
internal readonly struct DoubleArrayUnit
{
    private readonly uint _unit;

    public DoubleArrayUnit(uint unit) => _unit = unit;

    /// <summary>Base value (offset for child nodes).</summary>
    public uint Base => _unit;

    /// <summary>Check value (parent node verification).</summary>
    public uint Check => _unit;

    /// <summary>Whether this unit is a leaf (has a value).</summary>
    public bool HasLeaf => ((_unit >> 8) & 1) == 1;

    /// <summary>Value stored in a leaf node.</summary>
    public int Value => (int)(_unit >> 10);

    /// <summary>Label of this node.</summary>
    public byte Label => (byte)(_unit & 0xFF);
}

/// <summary>
/// 只读双数组字典树，用于高效公共前缀搜索。
/// 被 <see cref="PrecompiledNormalizer"/> 用于查找预编译标准化规则。
/// </summary>
internal sealed class DoubleArrayTrie
{
    private readonly DoubleArrayUnit[] _units;

    public DoubleArrayTrie(DoubleArrayUnit[] units)
    {
        _units = units;
    }

    /// <summary>
    /// 查找字典树中作为给定输入前缀的所有键。
    /// 返回找到的匹配数。
    /// </summary>
    /// <param name="input">UTF-8 encoded input bytes.</param>
    /// <param name="results">Buffer to store results (length, value pairs).</param>
    /// <returns>Number of matches found.</returns>
    public int CommonPrefixSearch(ReadOnlySpan<byte> input, Span<DoubleArrayResultPair> results)
    {
        int numResults = 0;
        uint nodeId = 0;
        var unit = _units[nodeId];
        uint idx;

        // Follow the root's child
        idx = unit.Base;
        if (idx >= _units.Length)
        {
            return 0;
        }

        var unit2 = _units[idx];
        nodeId = idx;

        for (int i = 0; i < input.Length; i++)
        {
            byte c = input[i];
            idx = nodeId + c;

            if (idx >= _units.Length || _units[idx].Check != nodeId)
            {
                break;
            }

            nodeId = idx;
            unit = _units[nodeId];

            if (unit.HasLeaf)
            {
                // This node has a leaf child (null byte terminator), meaning it's a complete key
                if (numResults < results.Length)
                {
                    results[numResults] = new DoubleArrayResultPair(i + 1, unit.Value);
                    numResults++;
                }
            }

            // Move to child
            idx = unit.Base;
            if (idx >= _units.Length)
            {
                break;
            }
            nodeId = idx;
        }

        return numResults;
    }
}
