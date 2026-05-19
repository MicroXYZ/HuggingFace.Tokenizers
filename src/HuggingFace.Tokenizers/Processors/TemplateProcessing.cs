using System.Diagnostics.CodeAnalysis;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Processors;

/// <summary>
/// 基于模板的后处理器，按照片段序列插入 token。
/// 支持引用输入序列（A、B）和具有多个 ID 的特殊 token。
/// </summary>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
[TokenizerComponent("TemplateProcessing")]
public sealed class TemplateProcessing : IPostProcessor
{
    private readonly IReadOnlyList<TemplatePiece> _singleTemplate;
    private readonly IReadOnlyList<TemplatePiece>? _pairTemplate;

    /// <summary>单条输入的模板。</summary>
    public IReadOnlyList<TemplatePiece> SingleTemplate => _singleTemplate;

    /// <summary>配对输入的模板，未配置时为 null。</summary>
    public IReadOnlyList<TemplatePiece>? PairTemplate => _pairTemplate;

    /// <summary>
    /// 创建基于模板的后处理器。
    /// </summary>
    public TemplateProcessing(
        IReadOnlyList<TemplatePiece> singleTemplate,
        IReadOnlyList<TemplatePiece>? pairTemplate = null)
    {
        ArgumentNullException.ThrowIfNull(singleTemplate);

        if (singleTemplate.Count == 0)
            throw new ArgumentException("Single template must contain at least one piece.", nameof(singleTemplate));

        _singleTemplate = singleTemplate;
        _pairTemplate = pairTemplate;
    }

    /// <inheritdoc />
    public int AddedTokens(bool isPair)
    {
        var template = GetTemplate(isPair);
        int count = 0;
        foreach (var piece in template)
        {
            if (piece is SpecialTokenPiece special)
                count += special.TokenIds.Count;
        }
        return count;
    }

    /// <inheritdoc />
    public Encoding Process(Encoding encoding, Encoding? pairEncoding, bool addSpecialTokens)
    {
        if (!addSpecialTokens)
            return pairEncoding is null ? encoding : Encoding.Merge([encoding, pairEncoding], growingOffsets: false);

        bool isPair = pairEncoding is not null;
        var template = GetTemplate(isPair);
        var parts = new List<Encoding>(template.Count);

        foreach (var piece in template)
        {
            switch (piece)
            {
                case SequencePiece seq:
                    var source = seq.SequenceIndex == 0
                        ? encoding
                        : pairEncoding ?? throw new InvalidOperationException(
                            "Template references sequence 1 but no pair encoding was provided.");
                    parts.Add(WithTypeId(source, piece.TypeId));
                    break;

                case SpecialTokenPiece special:
                    parts.Add(MakeSpecialToken(special, piece.TypeId));
                    break;
            }
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
            results.Add(Process(encodings[i - 1], encodings[i], addSpecialTokens));
        return results;
    }

    private IReadOnlyList<TemplatePiece> GetTemplate(bool isPair)
    {
        if (isPair && _pairTemplate is not null)
            return _pairTemplate;
        return _singleTemplate;
    }

    /// <summary>
    /// 为特殊 token 创建 Encoding，支持多个 ID。
    /// 使用 actual token text (not placeholder).
    /// </summary>
    private static Encoding MakeSpecialToken(SpecialTokenPiece special, uint typeId)
    {
        int len = special.TokenIds.Count;
        var ids = new uint[len];
        var typeIds = new uint[len];
        var tokens = new string[len];
        var words = new uint?[len];
        var offsets = new (int, int)[len];
        var specialMask = new uint[len];
        var attention = new uint[len];

        for (int i = 0; i < len; i++)
        {
            ids[i] = special.TokenIds[i];
            typeIds[i] = typeId;
            tokens[i] = special.TokenTexts[i];
            words[i] = null;
            offsets[i] = (0, 0);
            specialMask[i] = 1;
            attention[i] = 1;
        }

        return new Encoding(ids, typeIds, tokens, words, offsets, specialMask, attention);
    }

    private static Encoding WithTypeId(Encoding encoding, uint typeId)
        => PostProcessorHelper.WithTypeId(encoding, typeId);
}
