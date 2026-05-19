namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 截断配置。
/// </summary>
public sealed class TruncationParams
{
    /// <summary>截断策略。</summary>
    public TruncationStrategy Strategy { get; set; } = TruncationStrategy.LongestFirst;

    /// <summary>最大长度。与 Rust 一致，默认 512。</summary>
    public int MaxLength { get; set; } = 512;

    /// <summary>滑动窗口步长。</summary>
    public int Stride { get; set; }

    /// <summary>截断方向。</summary>
    public TruncationDirection Direction { get; set; } = TruncationDirection.Right;
}
