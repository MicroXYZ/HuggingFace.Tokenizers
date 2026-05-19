namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>截断策略。</summary>
public enum TruncationStrategy
{
    /// <summary>优先截断较长的序列。</summary>
    LongestFirst,
    /// <summary>仅截断第一个序列。</summary>
    OnlyFirst,
    /// <summary>仅截断第二个序列。</summary>
    OnlySecond
}
