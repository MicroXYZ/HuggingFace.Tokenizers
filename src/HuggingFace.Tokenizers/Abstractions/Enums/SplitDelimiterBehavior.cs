namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 拆分分隔符的行为。
/// </summary>
public enum SplitDelimiterBehavior
{
    Removed,
    Isolated,
    MergedWithPrevious,
    MergedWithNext,
    Contiguous
}
