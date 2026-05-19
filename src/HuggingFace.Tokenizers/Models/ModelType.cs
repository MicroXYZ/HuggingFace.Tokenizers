namespace HuggingFace.Tokenizers.Models;

/// <summary>
/// 分词模型类型枚举。
/// </summary>
public enum ModelType
{
    /// <summary>字节对编码模型。</summary>
    BPE,

    /// <summary>WordPiece 模型（BERT 风格）。</summary>
    WordPiece,

    /// <summary>词级模型。</summary>
    WordLevel,

    /// <summary>Unigram 语言模型。</summary>
    Unigram
}
