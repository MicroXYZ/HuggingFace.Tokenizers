using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 管理已拆分为子部分的字符串，每个子部分可以独立进行标准化和分词。
/// 追踪偏移回到原始字符串。
/// </summary>
public sealed class PreTokenizedString
{
    private readonly string _original;
    private readonly List<Split> _splits;

    public PreTokenizedString(string original)
    {
        _original = original;
        _splits = [new Split(new NormalizedString(original))];
    }

    /// <summary>
    /// 接受 ReadOnlySpan&lt;char&gt;，内部物化为 string。
    /// </summary>
    public PreTokenizedString(ReadOnlySpan<char> original) : this(original.ToString()) { }

    internal PreTokenizedString(string original, List<Split> splits)
    {
        _original = original;
        _splits = splits;
    }

    /// <summary>获取原始字符串。</summary>
    public string GetOriginal() => _original;

    /// <summary>获取当前拆分列表。</summary>
    public IReadOnlyList<Split> GetSplits() => _splits;

    /// <summary>
    /// 使用提供的函数拆分每个子部分。
    /// 拆分函数接收索引和 NormalizedString，必须产生
    /// 拼接后能还原原始输入的拆分结果。
    /// </summary>
    public void Split(Func<int, NormalizedString, IEnumerable<NormalizedString>> splitFn)
    {
        var newSplits = new List<Split>(_splits.Count);
        for (int i = 0; i < _splits.Count; i++)
        {
            var originalSplit = _splits[i];
            if (originalSplit.Tokens is not null)
            {
                newSplits.Add(originalSplit);
                continue;
            }

            foreach (var part in splitFn(i, originalSplit.Normalized))
            {
                if (!part.IsEmpty)
                    newSplits.Add(new Split(part));
            }
        }
        _splits.Clear();
        _splits.AddRange(newSplits);
    }

    /// <summary>
    /// 标准化所有未附加 token 的拆分。
    /// </summary>
    public void Normalize(Action<NormalizedString> normalize)
    {
        foreach (var split in _splits)
        {
            if (split.Tokens is null)
                normalize(split.Normalized);
        }
    }

    /// <summary>
    /// 分词所有未附加 token 的拆分。
    /// </summary>
    public void Tokenize(Func<NormalizedString, List<Token>> tokenize)
    {
        foreach (var split in _splits)
        {
            if (split.Tokens is null)
                split.Tokens = tokenize(split.Normalized);
        }
    }

    /// <summary>
    /// 使用轻量级 TokenRef 分词所有未附加 token 的拆分。
    /// 不分配 string，适用于内部编码路径。
    /// </summary>
    public void TokenizeRef(Func<NormalizedString, List<TokenRef>> tokenizeRef)
    {
        foreach (var split in _splits)
        {
            if (split.Tokens is null && split.TokenRefs is null)
                split.TokenRefs = tokenizeRef(split.Normalized);
        }
    }

    /// <summary>
    /// 带 token 限制的分词（用于截断）。
    /// 分词到足够的 token 数量后停止，避免对全部文本完整编码后再截断。
    /// </summary>
    /// <summary>
    /// 带 token 限制的分词（用于截断）。
    /// 分词到足够的 token 数量后停止，避免对全部文本完整编码后再截断。
    /// 与 Rust tokenize_with_limit 对齐：不 trim 最后一个 split 的多余 tokens，
    /// 由调用方的 Truncate 负责精确截断 + overflow 创建。
    /// </summary>
    public void TokenizeWithLimit(
        Func<NormalizedString, List<Token>> tokenize,
        int maxTokens,
        TruncationDirection direction)
    {
        int totalTokens = 0;

        if (direction == TruncationDirection.Right)
        {
            int lastTokenizedIdx = 0;
            for (int i = 0; i < _splits.Count; i++)
            {
                var split = _splits[i];
                if (split.Tokens is not null)
                {
                    totalTokens += split.Tokens.Count;
                    lastTokenizedIdx = i + 1;
                    if (totalTokens >= maxTokens) break;
                    continue;
                }

                var tokens = tokenize(split.Normalized);
                split.Tokens = tokens;
                totalTokens += tokens.Count;
                lastTokenizedIdx = i + 1;

                if (totalTokens >= maxTokens) break;
            }

            // 与 Rust 对齐：不 trim 最后一个 split 的多余 tokens
            // 由 Truncate 负责精确截断
            if (lastTokenizedIdx < _splits.Count)
                _splits.RemoveRange(lastTokenizedIdx, _splits.Count - lastTokenizedIdx);
        }
        else // Left
        {
            int firstTokenizedIdx = _splits.Count;
            for (int i = _splits.Count - 1; i >= 0; i--)
            {
                var split = _splits[i];
                if (split.Tokens is not null)
                {
                    totalTokens += split.Tokens.Count;
                    firstTokenizedIdx = i;
                    if (totalTokens >= maxTokens) break;
                    continue;
                }

                var tokens = tokenize(split.Normalized);
                split.Tokens = tokens;
                totalTokens += tokens.Count;
                firstTokenizedIdx = i;

                if (totalTokens >= maxTokens) break;
            }

            // 与 Rust 对齐：不 trim 第一个 split 的多余 tokens
            // 由 Truncate 负责精确截断
            if (firstTokenizedIdx > 0)
                _splits.RemoveRange(0, firstTokenizedIdx);
        }
    }

    /// <summary>
    /// 仅从所有拆分中提取 token ID，跳过偏移追踪、
    /// token 字符串、类型 ID、word ID、特殊 token 掩码和注意力掩码。
    /// 当仅需要 ID 时，比 <see cref="ToEncoding"/> 显著更快。
    /// 优先使用 TokenRefs（无 string 分配），回退到 Tokens。
    /// </summary>
    public uint[] ToIds()
    {
        int totalTokens = 0;
        for (int i = 0; i < _splits.Count; i++)
        {
            var split = _splits[i];
            if (split.TokenRefs is not null)
            {
                totalTokens += split.TokenRefs.Count;
            }
            else if (split.Tokens is not null)
            {
                totalTokens += split.Tokens.Count;
            }
            else
            {
                throw new InvalidOperationException($"Split {i} has not been tokenized yet.");
            }
        }

        var ids = new uint[totalTokens];
        int offset = 0;
        for (int i = 0; i < _splits.Count; i++)
        {
            var split = _splits[i];
            if (split.TokenRefs is not null)
            {
                var refs = split.TokenRefs;
                for (int j = 0; j < refs.Count; j++)
                    ids[offset++] = refs[j].Id;
            }
            else if (split.Tokens is not null)
            {
                var tokens = split.Tokens;
                for (int j = 0; j < tokens.Count; j++)
                    ids[offset++] = tokens[j].Id;
            }
        }
        return ids;
    }

    /// <summary>
    /// 转换为 Encoding，所有偏移相对于原始字符串。
    /// 当 offsetType 为 Char 时，模型（如 BPE）的字节偏移会使用原始字符串转换为字符偏移。
    /// 当为 Byte 时，偏移直接传递（与 Rust 默认 encode() 行为一致）。
    /// </summary>
    public Encoding ToEncoding(uint typeId, OffsetType offsetType = OffsetType.Byte)
    {
        int totalCount = ValidateAndCountTokens();

        var (ids, tokens, offsets, typeIds, words, specialMask, attentionMask) = AllocateEncodingArrays(totalCount, typeId);

        BytesToCharOffsetConverter? converter = offsetType == OffsetType.Char
            ? new BytesToCharOffsetConverter(_original)
            : null;

        int idx = 0;
        for (int wordIdx = 0; wordIdx < _splits.Count; wordIdx++)
        {
            var split = _splits[wordIdx];
            foreach (var token in split.Tokens!)
            {
                ids[idx] = token.Id;
                tokens[idx] = token.Value;

                var tokenOffsets = token.Offsets;
                var converted = split.Normalized.ConvertOffsets(
                    OffsetReferential.Normalized,
                    tokenOffsets.Start..tokenOffsets.End);
                if (converted is not null)
                    tokenOffsets = (converted.Value.Start.Value, converted.Value.End.Value);

                offsets[idx] = converter is not null
                    ? converter.Convert(tokenOffsets)
                    : tokenOffsets;
                words[idx] = (uint)wordIdx;
                idx++;
            }
        }

        return new Encoding(ids, typeIds, tokens, words, offsets, specialMask, attentionMask);
    }

    /// <summary>
    /// 使用固定 word 索引转换为 Encoding（用于预分词输入）。
    /// </summary>
    public Encoding ToEncoding(uint typeId, uint? wordIdx, OffsetType offsetType = OffsetType.Byte)
    {
        int totalCount = ValidateAndCountTokens();

        var (ids, tokens, offsets, typeIds, words, specialMask, attentionMask) = AllocateEncodingArrays(totalCount, typeId);

        int idx = 0;
        foreach (var split in _splits)
        {
            foreach (var token in split.Tokens!)
            {
                ids[idx] = token.Id;
                tokens[idx] = token.Value;
                offsets[idx] = token.Offsets;
                words[idx] = wordIdx;
                idx++;
            }
        }

        return new Encoding(ids, typeIds, tokens, words, offsets, specialMask, attentionMask);
    }

    /// <summary>
    /// 验证所有拆分已分词，返回总 token 数。
    /// </summary>
    private int ValidateAndCountTokens()
    {
        int totalCount = 0;
        for (int w = 0; w < _splits.Count; w++)
        {
            if (_splits[w].Tokens is null)
                throw new InvalidOperationException($"Split {w} has not been tokenized yet.");
            totalCount += _splits[w].Tokens!.Count;
        }
        return totalCount;
    }

    /// <summary>
    /// 分配 Encoding 所需的数组并填充常量值。
    /// </summary>
    private static (uint[] ids, string[] tokens, (int Start, int End)[] offsets,
        uint[] typeIds, uint?[] words, uint[] specialMask, uint[] attentionMask)
        AllocateEncodingArrays(int totalCount, uint typeId)
    {
        var ids = new uint[totalCount];
        var tokens = new string[totalCount];
        var offsets = new (int Start, int End)[totalCount];
        var typeIds = new uint[totalCount];
        var words = new uint?[totalCount];
        var specialMask = new uint[totalCount];
        var attentionMask = new uint[totalCount];

        Array.Fill(typeIds, typeId);
        Array.Fill(attentionMask, 1u);

        return (ids, tokens, offsets, typeIds, words, specialMask, attentionMask);
    }
}
