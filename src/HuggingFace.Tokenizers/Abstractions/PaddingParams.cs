namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 填充配置。
/// </summary>
public sealed class PaddingParams
{
    /// <summary>填充策略。</summary>
    public PaddingStrategy Strategy { get; set; } = PaddingStrategy.BatchLongest;

    /// <summary>固定填充时的最大长度。</summary>
    public int MaxLength { get; set; }

    /// <summary>填充 token 的 ID。</summary>
    public uint PadId { get; set; }

    /// <summary>填充 token 的类型 ID。</summary>
    public uint PadTypeId { get; set; }

    /// <summary>填充 token 的字符串表示。</summary>
    public string PadToken { get; set; } = "[PAD]";

    /// <summary>填充方向。</summary>
    public PaddingDirection Direction { get; set; } = PaddingDirection.Right;

    /// <summary>填充到此值的倍数。为 null 时不启用。</summary>
    public int? PadToMultipleOf { get; set; }
}
