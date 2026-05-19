namespace HuggingFace.Tokenizers.Processors;

/// <summary>
/// 构建 <see cref="TemplatePiece"/> 实例的工厂方法。
/// </summary>
public static class Template
{
    /// <summary>创建对序列 A（第一个输入）的引用。</summary>
    public static SequencePiece A(uint typeId = 0) => new(0, typeId);

    /// <summary>创建对序列 B（第二个输入）的引用。</summary>
    public static SequencePiece B(uint typeId = 1) => new(1, typeId);

    /// <summary>创建单个 ID 的特殊 token 片段。</summary>
    public static SpecialTokenPiece Special(uint tokenId, uint typeId = 0) => new(tokenId, typeId);

    /// <summary>创建带有显式 token 文本的特殊 token 片段。</summary>
    public static SpecialTokenPiece Special(uint tokenId, string tokenText, uint typeId = 0)
        => new([tokenId], [tokenText], typeId);

    /// <summary>创建多个 ID 和文本的特殊 token 片段。</summary>
    public static SpecialTokenPiece Special(IReadOnlyList<uint> tokenIds, IReadOnlyList<string> tokenTexts, uint typeId = 0)
        => new(tokenIds, tokenTexts, typeId);
}
