namespace HuggingFace.Tokenizers.Processors;

/// <summary>
/// 表示模板中的一个片段 — 或对序列的引用，或要插入的特殊 token。
/// </summary>
public abstract class TemplatePiece
{
    /// <summary>此片段产生的 token 的类型 ID。</summary>
    public uint TypeId { get; }

    internal TemplatePiece(uint typeId) => TypeId = typeId;
}
