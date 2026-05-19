namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>截断方向。</summary>
public enum TruncationDirection
{
    /// <summary>保留尾部（从头部截断）。</summary>
    Left,
    /// <summary>保留头部（从尾部截断）。</summary>
    Right
}
