namespace HuggingFace.Tokenizers.Processors;

/// <summary>
/// 插入一个或多个 ID 的特殊 token 的模板片段。
/// 支持多 token 特殊 token（与 Rust SpecialToken 一致）。
/// </summary>
public sealed class SpecialTokenPiece : TemplatePiece
{
    /// <summary>特殊 token 的 ID 列表（支持多个 ID）。</summary>
    public IReadOnlyList<uint> TokenIds { get; }

    /// <summary>特殊 token 的文本表示（与 TokenIds 等长）。</summary>
    public IReadOnlyList<string> TokenTexts { get; }

    /// <summary>是否为特殊 token（在掩码中标记）。</summary>
    public bool IsSpecial => true;

    internal SpecialTokenPiece(uint tokenId, uint typeId)
        : this([tokenId], [$"[special:{tokenId}]"], typeId)
    {
    }

    internal SpecialTokenPiece(IReadOnlyList<uint> tokenIds, IReadOnlyList<string> tokenTexts, uint typeId)
        : base(typeId)
    {
        if (tokenIds.Count == 0)
            throw new ArgumentException("TokenIds 不能为空。", nameof(tokenIds));
        if (tokenIds.Count != tokenTexts.Count)
            throw new ArgumentException("TokenIds 和 TokenTexts 长度必须相同。");
        TokenIds = tokenIds;
        TokenTexts = tokenTexts;
    }
}
