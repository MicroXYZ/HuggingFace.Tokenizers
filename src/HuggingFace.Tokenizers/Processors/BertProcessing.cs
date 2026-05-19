using System.Diagnostics.CodeAnalysis;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Processors;

/// <summary>
/// BERT 风格分词的后处理器。
/// 在序列开头添加 [CLS]，末尾添加 [SEP]。
/// 对于配对输入，还在两个序列之间插入 [SEP]。
/// 处理溢出 token（递归地用 CLS/SEP 包装每个溢出）。
/// </summary>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
[TokenizerComponent("BertProcessing")]
public sealed class BertProcessing : IPostProcessor
{
    private readonly (string Token, uint Id) _sep;
    private readonly (string Token, uint Id) _cls;

    /// <summary>SEP token 对。</summary>
    public (string Token, uint Id) Sep => _sep;

    /// <summary>CLS token 对。</summary>
    public (string Token, uint Id) Cls => _cls;

    /// <summary>
    /// 创建新的 BERT post-processor with the specified SEP and CLS tokens.
    /// </summary>
    public BertProcessing(
        (string Token, uint Id) sep = default,
        (string Token, uint Id) cls = default)
    {
        _sep = sep.Token is null ? ("[SEP]", 102u) : sep;
        _cls = cls.Token is null ? ("[CLS]", 101u) : cls;
    }

    /// <inheritdoc />
    public int AddedTokens(bool isPair) => isPair ? 3 : 2;

    /// <inheritdoc />
    public Encoding Process(Encoding encoding, Encoding? pairEncoding, bool addSpecialTokens)
    {
        if (!addSpecialTokens)
            return pairEncoding is null ? encoding : Encoding.Merge([encoding, pairEncoding], growingOffsets: false);

        return WrapEncoding(encoding, pairEncoding);
    }

    /// <summary>
    /// 用 CLS/SEP token 包装编码，并递归包装其溢出。
    /// MakeSpecialToken 命中缓存时零分配。
    /// 直接附加 overflowings，不提取数组重建 Encoding。
    /// </summary>
    private Encoding WrapEncoding(Encoding encoding, Encoding? pairEncoding)
    {
        // 提取溢出，由本方法自行递归处理
        var overflowings = encoding.GetOverflowing();

        // 创建一个不含溢出的 encoding 用于 Merge（COW 浅拷贝，共享底层数组）
        var encodingNoOverflow = encoding.CloneWithoutOverflowing();

        // Build: [CLS] + encoding + [SEP]（MakeSpecialToken 缓存命中，零分配）
        var parts = new List<Encoding>(3 + (pairEncoding is not null ? 2 : 0))
        {
            MakeSpecialToken(_cls, typeId: 0),
            encodingNoOverflow,
            MakeSpecialToken(_sep, typeId: 0)
        };

        if (pairEncoding is not null)
        {
            var pairOverflowings = pairEncoding.GetOverflowing();

            var pairNoOverflow = pairEncoding.CloneWithoutOverflowing();

            parts.Add(WithTypeId(pairNoOverflow, 1));
            parts.Add(MakeSpecialToken(_sep, typeId: 1));

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

        // 无配对 — 仅包装溢出
        if (overflowings.Count > 0)
        {
            var mergedOverflowings = new List<Encoding>();
            foreach (var overflow in overflowings)
                mergedOverflowings.Add(WrapEncoding(overflow, null));

            var merged = Encoding.Merge(parts, growingOffsets: false);
            if (mergedOverflowings.Count > 0)
                merged.GetOverflowing().AddRange(mergedOverflowings);
            return merged;
        }

        return Encoding.Merge(parts, growingOffsets: false);
    }

    /// <inheritdoc />
    public IReadOnlyList<Encoding> ProcessEncodings(IReadOnlyList<Encoding> encodings, bool addSpecialTokens)
    {
        if (encodings.Count == 0)
            return [];

        if (encodings.Count == 1)
            return [Process(encodings[0], null, addSpecialTokens)];

        var results = new List<Encoding>(encodings.Count);
        results.Add(Process(encodings[0], null, addSpecialTokens));
        for (int i = 1; i < encodings.Count; i++)
            results.Add(Process(encodings[0], encodings[i], addSpecialTokens));
        return results;
    }

    private static Encoding MakeSpecialToken((string Token, uint Id) token, uint typeId)
        => PostProcessorHelper.MakeSpecialToken(token, typeId);

    private static Encoding WithTypeId(Encoding encoding, uint typeId)
        => PostProcessorHelper.WithTypeId(encoding, typeId);
}
