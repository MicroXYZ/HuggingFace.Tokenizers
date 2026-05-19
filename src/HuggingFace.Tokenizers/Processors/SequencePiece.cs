namespace HuggingFace.Tokenizers.Processors;

/// <summary>
/// 引用序列 A（第一个输入）或序列 B（第二个/配对输入）的模板片段。
/// </summary>
public sealed class SequencePiece : TemplatePiece
{
    /// <summary>此片段引用的序列索引（0 = A，1 = B）。</summary>
    public int SequenceIndex { get; }

    internal SequencePiece(int sequenceIndex, uint typeId) : base(typeId)
        => SequenceIndex = sequenceIndex;
}
