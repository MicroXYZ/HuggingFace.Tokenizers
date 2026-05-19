using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.PreTokenizers;

/// <summary>
/// 使用 <see cref="IPattern"/> 和可配置的
/// <see cref="SplitDelimiterBehavior"/>.
/// </summary>
[TokenizerComponent("Split", EnumNaming = EnumNamingConvention.PascalCase)]
public sealed class SplitPreTokenizer : IPreTokenizer
{
    /// <summary>
    /// 用于拆分的模式。
    /// </summary>
    public IPattern Pattern { get; }

    /// <summary>
    /// 应用于匹配分隔符的行为。
    /// </summary>
    public SplitDelimiterBehavior Behavior { get; }

    /// <summary>
    /// 是否反转模式匹配。
    /// </summary>
    public bool Invert { get; }

    /// <summary>
    /// 初始化新的 <see cref="SplitPreTokenizer"/>.
    /// </summary>
    /// <param name="pattern">要拆分的模式。</param>
    /// <param name="behavior">匹配分隔符在输出中的处理方式。</param>
    /// <param name="invert">是否反转模式匹配。</param>
    public SplitPreTokenizer(IPattern pattern, SplitDelimiterBehavior behavior = SplitDelimiterBehavior.Removed, bool invert = false)
    {
        ArgumentNullException.ThrowIfNull(pattern);
        Pattern = pattern;
        Behavior = behavior;
        Invert = invert;
    }

    /// <inheritdoc />
    public void PreTokenize(PreTokenizedString pretokenized)
    {
        pretokenized.Split((_, normalized) =>
        {
            var splitResult = normalized.Split(Behavior, Pattern.FindMatches);
            return splitResult.Select(part => part.Part);
        });
    }
}
