namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>填充策略。</summary>
public enum PaddingStrategy
{
    /// <summary>填充到批次中最长序列的长度。</summary>
    BatchLongest,
    /// <summary>填充到固定长度。</summary>
    Fixed
}
