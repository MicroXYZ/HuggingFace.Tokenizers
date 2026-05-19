// 基于 Microsoft.ML.Tokenizers SentencePieceNormalizer（MIT 许可证）
// Original: https://github.com/dotnet/machinelearning
// Adapted for HuggingFace.Tokenizers NormalizedString interface.

using System.Diagnostics;
using System.Text;
using HuggingFace.Tokenizers.Abstractions;

// Alias to avoid conflict with HuggingFace.Tokenizers.Abstractions.Encoding
using SysEncoding = System.Text.Encoding;

namespace HuggingFace.Tokenizers.Normalizers;

/// <summary>
/// 使用 SentencePiece 预编译标准化规则的标准化器。
/// 加载二进制预编译字符映射（来自 SentencePiece 的 normalizer.model），
/// 并应用其定义的字符变换。
/// 用于 ALBERT、XLNet 和其他基于 SentencePiece 的模型。
/// </summary>
[TokenizerComponent("Precompiled")]
public sealed class PrecompiledNormalizer : INormalizer
{
    private const int MaxTrieResultsSize = 32;

    private readonly DoubleArrayTrie? _trie;
    private readonly byte[]? _normalizedMap;
    private readonly byte[]? _rawCharsMap;

    /// <summary>
    /// 是否在开头添加虚拟前缀字符 (▁)。
    /// </summary>
    public bool AddDummyPrefix { get; }

    /// <summary>
    /// 是否将空格替换为 ▁ 字符。
    /// </summary>
    public bool EscapeWhiteSpaces { get; }

    /// <summary>
    /// 是否移除多余（连续）空白。
    /// </summary>
    public bool RemoveExtraWhiteSpaces { get; }

    /// <summary>
    /// 是否将空白视为后缀而非前缀。
    /// </summary>
    public bool TreatWhitespaceAsSuffix { get; }

    /// <summary>
    /// 原始预编译字符映射字节，用于序列化。
    /// </summary>
    public byte[]? RawCharsMap => _rawCharsMap;

    /// <summary>
    /// 从预编译字符映射创建 PrecompiledNormalizer。
    /// </summary>
    /// <param name="precompiledCharsMap">
    /// 来自 SentencePiece normalizer.model 的二进制预编译字符映射。
    /// 格式：[uint32 trie_size][trie_data...][normalized_strings...]
    /// </param>
    /// <param name="addDummyPrefix">是否在开头添加 ▁ 前缀。</param>
    /// <param name="escapeWhiteSpaces">是否将空格替换为 ▁。</param>
    /// <param name="removeExtraWhiteSpaces">是否移除连续空白。</param>
    /// <param name="treatWhitespaceAsSuffix">是否将前缀改为后缀添加。</param>
    public PrecompiledNormalizer(
        ReadOnlySpan<byte> precompiledCharsMap,
        bool addDummyPrefix = true,
        bool escapeWhiteSpaces = true,
        bool removeExtraWhiteSpaces = true,
        bool treatWhitespaceAsSuffix = false)
    {
        AddDummyPrefix = addDummyPrefix;
        EscapeWhiteSpaces = escapeWhiteSpaces;
        RemoveExtraWhiteSpaces = removeExtraWhiteSpaces;
        TreatWhitespaceAsSuffix = treatWhitespaceAsSuffix;

        if (!precompiledCharsMap.IsEmpty)
        {
            _rawCharsMap = precompiledCharsMap.ToArray();
            DecodePrecompiledCharsMap(precompiledCharsMap, out var trieBlob, out _normalizedMap);
            Debug.Assert(trieBlob is not null);
            _trie = new DoubleArrayTrie(trieBlob!);
        }
    }

    /// <summary>
    /// 从预编译字符映射字节数组创建 PrecompiledNormalizer。
    /// </summary>
    public PrecompiledNormalizer(
        byte[] precompiledCharsMap,
        bool addDummyPrefix = true,
        bool escapeWhiteSpaces = true,
        bool removeExtraWhiteSpaces = true,
        bool treatWhitespaceAsSuffix = false)
        : this(precompiledCharsMap.AsSpan(), addDummyPrefix, escapeWhiteSpaces, removeExtraWhiteSpaces, treatWhitespaceAsSuffix)
    {
    }

    /// <inheritdoc />
    public void Normalize(NormalizedString normalized)
    {
        var text = normalized.GetSpan();
        if (text.IsEmpty)
            return;

        // 基于字素的标准化，与 Rust 实现一致。
        // 遍历字素，对每个字素尝试字典树匹配，构建 (char, change) 对。
        var transformations = new List<(char, int)>(text.Length);
        bool modified = false;

        // 使用 Span 版本的字素遍历，避免 GetTextElement() 的 string 分配
        int pos = 0;
        Span<char> singleChar = stackalloc char[1];
        while (pos < text.Length)
        {
            int graphemeLen = System.Globalization.StringInfo.GetNextTextElementLength(text.Slice(pos));
            var grapheme = text.Slice(pos, graphemeLen);
            pos += graphemeLen;

            // 先尝试匹配整个字素（如果足够短）。
            // 使用字节长度比较（与 Rust 一致：grapheme.len() < 6，其中 len() 为字节数）。
            int graphemeByteLen = SysEncoding.UTF8.GetByteCount(grapheme);
            if (graphemeByteLen < 6)
            {
                var norm = TransformUtf8(grapheme);
                if (norm is not null)
                {
                    modified = true;
                    ReplaceTransform(transformations, grapheme, norm);
                    continue;
                }
            }

            // 回退到字素内的单个字符
            foreach (var c in grapheme)
            {
                singleChar[0] = c;
                var norm = TransformUtf8(singleChar);
                if (norm is not null)
                {
                    modified = true;
                    ReplaceTransform(transformations, singleChar, norm);
                }
                else
                {
                    transformations.Add((c, 0));
                }
            }
        }

        if (modified)
        {
            normalized.Transform(transformations, 0);
        }
    }

    /// <summary>
    /// 在预编译字典树中查找字符串并返回标准化形式。
    /// 如果此字符串不存在标准化规则则返回 null。
    /// 已接受 ReadOnlySpan<char>，内部全程 Span 操作。
    /// </summary>
    private string? TransformUtf8(ReadOnlySpan<char> input)
    {
        if (_trie is null || _normalizedMap is null)
            return null;

        // 使用 stackalloc 避免 Encoding.UTF8.GetBytes 堆分配
        Span<byte> inputBytes = stackalloc byte[SysEncoding.UTF8.GetMaxByteCount(input.Length)];
        System.Text.Unicode.Utf8.FromUtf16(input, inputBytes, out _, out int bytesWritten);
        inputBytes = inputBytes.Slice(0, bytesWritten);

        Span<DoubleArrayResultPair> trieResults = stackalloc DoubleArrayResultPair[MaxTrieResultsSize];
        int numNodes = _trie.CommonPrefixSearch(inputBytes, trieResults);

        int longestLength = 0;
        int longestValue = 0;
        for (int k = 0; k < numNodes; k++)
        {
            if (trieResults[k].Length > longestLength)
            {
                longestLength = trieResults[k].Length;
                longestValue = trieResults[k].Value;
            }
        }

        if (longestLength == 0 || longestLength != inputBytes.Length)
            return null; // No exact match

        // 在映射中查找以 null 结尾的标准化字符串
        int end = longestValue;
        while (end < _normalizedMap.Length && _normalizedMap[end] != 0)
            end++;

        return SysEncoding.UTF8.GetString(_normalizedMap, longestValue, end - longestValue);
    }

    /// <summary>
    /// 构建替换的 (char, change) 变换，与 Rust 的 replace() 函数一致。
    /// oldPart 为原始文本，newPart 为标准化后的文本。
    /// 使用 Rune 计算 Unicode 码位数量，避免补充平面字符的计数错误。
    /// </summary>
    private static void ReplaceTransform(List<(char, int)> transformations, ReadOnlySpan<char> oldPart, string newPart)
    {
        // 使用 Rune 计算 Unicode code point 数量（而非 UTF-16 char 数量）
        // 匹配 Rust 的 str.chars().count() 语义
        int oldCount = 0;
        foreach (var rune in oldPart.EnumerateRunes()) oldCount++;
        int newCount = newPart.EnumerateRunes().Count();
        int diff = newCount - oldCount;

        // All new chars start with change=0
        foreach (var c in newPart)
            transformations.Add((c, 0));

        if (diff > 0)
        {
            // Adding chars: mark the last `diff` chars as change=1 (insertions)
            for (int i = transformations.Count - diff; i < transformations.Count; i++)
            {
                var entry = transformations[i];
                entry.Item2 = 1;
                transformations[i] = entry;
            }
        }
        else if (diff < 0)
        {
            // Removing chars: add `diff` to the last char's change
            if (transformations.Count > 0)
            {
                int last = transformations.Count - 1;
                var entry = transformations[last];
                entry.Item2 += diff;
                transformations[last] = entry;
            }
        }
    }

    /// <summary>
    /// 解码 SentencePiece 预编译字符映射二进制格式。
    /// </summary>
    private static void DecodePrecompiledCharsMap(
        ReadOnlySpan<byte> blob,
        out DoubleArrayUnit[] trieBlob,
        out byte[] normalized)
    {
        if (blob.Length <= sizeof(uint))
            throw new ArgumentException("Precompiled chars map blob is too small.");

        // 前 4 个字节：字典树数据大小（字节）
        uint trieBlobSize = BitConverter.ToUInt32(blob);
        blob = blob.Slice(sizeof(uint));

        if (trieBlobSize >= blob.Length)
            throw new ArgumentException("Trie data size exceeds blob size.");

        // 解析字典树单元
        int numUnits = (int)(trieBlobSize / sizeof(uint));
        trieBlob = new DoubleArrayUnit[numUnits];
        for (int i = 0; i < numUnits; i++)
        {
            trieBlob[i] = new DoubleArrayUnit(BitConverter.ToUInt32(blob.Slice(i * sizeof(uint))));
        }

        // 剩余字节：以 null 结尾的标准化字符串
        blob = blob.Slice((int)trieBlobSize);
        normalized = blob.ToArray();
    }
}
