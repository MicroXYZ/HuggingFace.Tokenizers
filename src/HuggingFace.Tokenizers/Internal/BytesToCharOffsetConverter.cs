namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// 使用预计算的映射将字节偏移转换为字符偏移。
/// 使用 int[] 数组替代 Dictionary&lt;int,int&gt;，减少每条目 ~32 字节开销。
/// </summary>
internal sealed class BytesToCharOffsetConverter
{
    private readonly int[] _map;

    /// <summary>
    /// 构建字节→字符偏移映射。
    /// 对于 UTF-8 编码中的每个字节位置，存储对应的字符索引。
    /// </summary>
    public BytesToCharOffsetConverter(string sequence)
    {
        // 估算 UTF-8 字节数：每个 char 最多 3 字节（BMP），surrogate pair 最多 4 字节
        int utf8Len = System.Text.Encoding.UTF8.GetByteCount(sequence);
        _map = new int[utf8Len + 1];
        int bytePos = 0;
        int charIdx = 0;
        foreach (var rune in sequence.EnumerateRunes())
        {
            int runeByteLen = rune.Utf8SequenceLength;
            for (int n = 0; n < runeByteLen; n++)
            {
                _map[bytePos + n] = charIdx;
            }
            bytePos += runeByteLen;
            charIdx += rune.Utf16SequenceLength;
        }
        // 末尾哨兵：用于 Convert 中 End 超出范围的情况
        _map[bytePos] = charIdx;
    }

    /// <summary>
    /// 将字节偏移对转换为字符偏移对。
    /// </summary>
    public (int Start, int End) Convert((int Start, int End) offsets)
    {
        int start = offsets.Start < _map.Length ? _map[offsets.Start] : offsets.Start;
        int end;
        if (offsets.End < _map.Length)
        {
            end = _map[offsets.End];
        }
        else
        {
            // End 超出映射范围（exclusive bound）
            end = offsets.End - 1 < _map.Length ? _map[offsets.End - 1] + 1 : start + 1;
        }
        return (start, end);
    }
}
