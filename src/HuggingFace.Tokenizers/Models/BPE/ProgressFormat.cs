namespace HuggingFace.Tokenizers.Models.BPE;

/// <summary>
/// 训练进度输出格式。
/// </summary>
public enum ProgressFormat
{
    /// <summary>不输出进度。</summary>
    Silent,
    /// <summary>标准进度报告（默认）。</summary>
    Standard,
    /// <summary>JsonLines 格式，适用于 CI/CD 集成。</summary>
    JsonLines
}
