using System.Diagnostics.CodeAnalysis;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Processors;

/// <summary>
/// 字节级后处理器 used by GPT-2 and similar models.
/// 修剪偏移以排除首尾空白（字节级空格字符）。
/// 与 Rust ByteLevel PostProcessor 实现一致。
/// </summary>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
[TokenizerComponent("ByteLevel")]
public sealed class ByteLevelPostProcessor : IPostProcessor
{
    private readonly bool _trimOffsets;
    private readonly bool _addPrefixSpace;

    /// <summary>
    /// 默认选项的单例实例（trimOffsets=true, addPrefixSpace=true）。
    /// </summary>
    public static readonly ByteLevelPostProcessor Instance = new();

    /// <summary>
    /// 创建新的 ByteLevelPostProcessor.
    /// </summary>
    /// <param name="trimOffsets">Whether to trim whitespace from offsets.</param>
    /// <param name="addPrefixSpace">Whether prefix space was added during pre-tokenization.</param>
    public ByteLevelPostProcessor(bool trimOffsets = true, bool addPrefixSpace = true)
    {
        _trimOffsets = trimOffsets;
        _addPrefixSpace = addPrefixSpace;
    }

    /// <inheritdoc />
    public int AddedTokens(bool isPair) => 0;

    /// <inheritdoc />
    public Encoding Process(Encoding encoding, Encoding? pairEncoding, bool addSpecialTokens)
    {
        if (_trimOffsets)
        {
            ProcessOffsets(encoding, _addPrefixSpace);
            foreach (var overflow in encoding.GetOverflowing())
                ProcessOffsets(overflow, _addPrefixSpace);
        }

        // 设置序列 ID（与 Rust PostProcessor::process 一致）
        encoding.SetSequenceId(0);

        if (pairEncoding is not null)
        {
            if (_trimOffsets)
            {
                ProcessOffsets(pairEncoding, _addPrefixSpace);
                foreach (var overflow in pairEncoding.GetOverflowing())
                    ProcessOffsets(overflow, _addPrefixSpace);
            }

            // 设置 pair 的序列 ID 和 typeIds（与 Rust 一致）
            pairEncoding.SetSequenceId(1);
            encoding.SetTypeIds(new uint[encoding.Length]); // all 0
            var pairTypeIds = new uint[pairEncoding.Length];
            Array.Fill(pairTypeIds, 1u);
            pairEncoding.SetTypeIds(pairTypeIds);

            return Encoding.Merge([encoding, pairEncoding], growingOffsets: false);
        }

        return encoding;
    }

    /// <inheritdoc />
    public IReadOnlyList<Encoding> ProcessEncodings(IReadOnlyList<Encoding> encodings, bool addSpecialTokens)
    {
        if (_trimOffsets)
        {
            for (int i = 0; i < encodings.Count; i++)
            {
                ProcessOffsets(encodings[i], _addPrefixSpace);
                foreach (var overflow in encodings[i].GetOverflowing())
                    ProcessOffsets(overflow, _addPrefixSpace);
            }
        }

        // Set sequence IDs (matches Rust)
        for (int i = 0; i < encodings.Count; i++)
            encodings[i].SetSequenceId(i);

        return encodings;
    }

    /// <summary>
    /// 从 token 偏移中修剪首尾空白。
    /// 与 Rust process_offsets 函数完全一致。
    /// 在字节级编码中，空格变为 Ġ (U+0120)，需要特殊处理。
    /// </summary>
    private static void ProcessOffsets(Encoding encoding, bool addPrefixSpace)
    {
        var offsets = encoding.GetOffsets();
        var tokens = encoding.GetTokens();
        var specialMask = encoding.GetSpecialTokensMask();

        for (int i = 0; i < offsets.Length; i++)
        {
            // Skip special tokens
            if (specialMask[i] == 1)
                continue;

            var (start, end) = offsets[i];
            var token = tokens[i];

            if (string.IsNullOrEmpty(token))
                continue;

            // 计算前导空白字符数。
            // Ġ (U+0120) 是 ByteLevel 编码中的空格等价物，
            // 与 Rust BYTES_CHAR[&b' '] 检查一致。
            int leadingSpaces = 0;
            foreach (var c in token)
            {
                if (c == ' ' || c == '\u0120' || char.IsWhiteSpace(c))
                    leadingSpaces++;
                else
                    break;
            }

            // 计算尾随空白字符数
            int trailingSpaces = 0;
            for (int j = token.Length - 1; j >= 0; j--)
            {
                if (token[j] == ' ' || token[j] == '\u0120' || char.IsWhiteSpace(token[j]))
                    trailingSpaces++;
                else
                    break;
            }

            if (leadingSpaces > 0)
            {
                // First token 或 offset starts at 0: preserve prefix space if addPrefixSpace
                bool isFirst = i == 0 || start == 0;
                if (isFirst && addPrefixSpace && leadingSpaces == 1)
                {
                    // Don't remove the single leading space we added
                    leadingSpaces = 0;
                }
                start = Math.Min(start + leadingSpaces, end);
            }

            if (trailingSpaces > 0 && end >= trailingSpaces)
            {
                end = Math.Max(end - trailingSpaces, start);
            }

            if (start != offsets[i].Start || end != offsets[i].End)
                encoding.SetOffset(i, (start, end));
        }
    }
}
