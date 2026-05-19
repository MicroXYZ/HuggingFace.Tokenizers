using System.Diagnostics.CodeAnalysis;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Processors;

/// <summary>
/// RoBERTa 风格分词的后处理器。
/// 在序列开头添加 &lt;s&gt; ，末尾添加 &lt;/s&gt; 。
/// 对于配对输入，还在两个序列之间插入 &lt;/s&gt; 。
/// 所有 encoding 的 typeIds 均重置为 0（不区分段落）。
/// </summary>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
[TokenizerComponent("RobertaProcessing")]
public sealed class RobertaProcessing : IPostProcessor
{
    private readonly (string Token, uint Id) _sep;
    private readonly (string Token, uint Id) _cls;
    private readonly bool _trimOffsets;
    private readonly bool _addPrefixSpace;

    /// <summary>SEP token 对。</summary>
    public (string Token, uint Id) Sep => _sep;

    /// <summary>CLS token 对。</summary>
    public (string Token, uint Id) Cls => _cls;

    /// <summary>是否从偏移量中修剪空白。</summary>
    public bool TrimOffsets => _trimOffsets;

    /// <summary>是否期望 token 中有前缀空格。</summary>
    public bool AddPrefixSpace => _addPrefixSpace;

    /// <summary>
    /// 创建新的 RoBERTa 后处理器。
    /// </summary>
    public RobertaProcessing(
        (string Token, uint Id) sep = default,
        (string Token, uint Id) cls = default,
        bool trimOffsets = true,
        bool addPrefixSpace = true)
    {
        _sep = sep.Token is null ? ("</s>", 2u) : sep;
        _cls = cls.Token is null ? ("<s>", 0u) : cls;
        _trimOffsets = trimOffsets;
        _addPrefixSpace = addPrefixSpace;
    }

    /// <inheritdoc />
    public int AddedTokens(bool isPair) => isPair ? 4 : 2;

    /// <inheritdoc />
    public Encoding Process(Encoding encoding, Encoding? pairEncoding, bool addSpecialTokens)
    {
        if (!addSpecialTokens)
            return pairEncoding is null ? encoding : Encoding.Merge([encoding, pairEncoding], growingOffsets: false);

        // RoBERTa 将所有 encoding 的 typeIds 设为 0（不区分段落）
        ResetTypeIds(encoding);

        var mainResult = WrapEncoding(encoding, pairEncoding);

        if (_trimOffsets)
            ApplyTrimOffsets(mainResult);

        return mainResult;
    }

    /// <inheritdoc />
    public IReadOnlyList<Encoding> ProcessEncodings(IReadOnlyList<Encoding> encodings, bool addSpecialTokens)
    {
        if (encodings.Count == 0)
            return [];

        if (encodings.Count == 1)
            return [Process(encodings[0], null, addSpecialTokens)];

        var results = new List<Encoding>(encodings.Count);
        foreach (var enc in encodings)
            results.Add(Process(enc, null, addSpecialTokens));
        return results;
    }

    /// <summary>
    /// 用 CLS/SEP 包装编码，并递归包装其溢出。
    /// 直接附加 overflowings，不提取数组重建 Encoding。
    /// </summary>
    private Encoding WrapEncoding(Encoding encoding, Encoding? pairEncoding)
    {
        var overflowings = encoding.GetOverflowing();

        // 创建不含溢出的 encoding 用于 Merge（COW 浅拷贝，共享底层数组）
        var encodingNoOverflow = encoding.CloneWithoutOverflowing();

        var parts = new List<Encoding>(6)
        {
            MakeSpecialToken(_cls, typeId: 0),
            encodingNoOverflow,
            MakeSpecialToken(_sep, typeId: 0)
        };

        if (pairEncoding is not null)
        {
            ResetTypeIds(pairEncoding);
            var pairOverflowings = pairEncoding.GetOverflowing();

            var pairNoOverflow = pairEncoding.CloneWithoutOverflowing();

            parts.Add(MakeSpecialToken(_sep, typeId: 0));
            parts.Add(pairNoOverflow);
            parts.Add(MakeSpecialToken(_sep, typeId: 0));

            var allPairEncodings = new List<Encoding> { pairEncoding };
            allPairEncodings.AddRange(pairOverflowings);

            var mergedOverflowings = new List<Encoding>();
            foreach (var mainOverflow in overflowings)
                foreach (var pairEnc in allPairEncodings)
                    mergedOverflowings.Add(WrapEncoding(mainOverflow, pairEnc));
            foreach (var pairOverflow in pairOverflowings)
                mergedOverflowings.Add(WrapEncoding(encoding, pairOverflow));

            var merged = Encoding.Merge(parts, growingOffsets: false);
            if (mergedOverflowings.Count > 0)
                merged.GetOverflowing().AddRange(mergedOverflowings);
            return merged;
        }

        if (overflowings.Count > 0)
        {
            var mergedOverflowings = new List<Encoding>();
            foreach (var overflow in overflowings)
                mergedOverflowings.Add(WrapEncoding(overflow, null));

            var merged = Encoding.Merge(parts, growingOffsets: false);
            merged.GetOverflowing().AddRange(mergedOverflowings);
            return merged;
        }

        return Encoding.Merge(parts, growingOffsets: false);
    }

    /// <summary>
    /// 将 encoding 的所有 typeIds 重置为 0。
    /// </summary>
    private static void ResetTypeIds(Encoding encoding)
    {
        var typeIds = new uint[encoding.Length];
        encoding.SetTypeIds(typeIds);
    }

    private static Encoding MakeSpecialToken((string Token, uint Id) token, uint typeId)
        => PostProcessorHelper.MakeSpecialToken(token, typeId);

    /// <summary>
    /// 从 token 偏移中修剪首尾空白。
    /// </summary>
    private static void ApplyTrimOffsets(Encoding encoding)
    {
        var offsets = encoding.GetOffsets();
        var tokens = encoding.GetTokens();
        var specialMask = encoding.GetSpecialTokensMask();

        for (int i = 0; i < offsets.Length; i++)
        {
            var (start, end) = offsets[i];
            var token = tokens[i];

            if (specialMask[i] == 1)
                continue;

            while (start < end && start < token.Length && char.IsWhiteSpace(token[start]))
                start++;

            while (end > start && end - 1 < token.Length && char.IsWhiteSpace(token[end - 1]))
                end--;

            if (start != offsets[i].Start || end != offsets[i].End)
                encoding.SetOffset(i, (start, end));
        }
    }
}
