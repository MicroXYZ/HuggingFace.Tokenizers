using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.PreTokenizers;

/// <summary>
/// 按标点字符拆分文本的预分词器，
/// 使用指定的 <see cref="SplitDelimiterBehavior"/>。
/// </summary>
[TokenizerComponent("Punctuation", EnumNaming = EnumNamingConvention.PascalCase)]
public sealed class PunctuationPreTokenizer : IPreTokenizer
{
    /// <summary>
    /// 拆分时应用于标点分隔符的行为。
    /// </summary>
    public SplitDelimiterBehavior Behavior { get; }

    /// <summary>
    /// 初始化新的 <see cref="PunctuationPreTokenizer"/>.
    /// </summary>
    /// <param name="behavior">拆分输出中处理标点分隔符的方式。</param>
    public PunctuationPreTokenizer(SplitDelimiterBehavior behavior = SplitDelimiterBehavior.Isolated)
    {
        Behavior = behavior;
    }

    /// <inheritdoc />
    public void PreTokenize(PreTokenizedString pretokenized)
    {
        var pattern = new PunctuationPattern();
        pretokenized.Split((_, normalized) =>
        {
            var splitResult = normalized.Split(Behavior, pattern.FindMatches);
            return splitResult.Select(part => part.Part);
        });
    }

    /// <summary>
    /// 匹配单个标点字符的内部模式。
    /// </summary>
    private sealed class PunctuationPattern : IPattern
    {
        public IReadOnlyList<(int Start, int End, bool IsMatch)> FindMatches(ReadOnlySpan<char> inside)
        {
            var result = new List<(int, int, bool)>();
            int last = 0;
            for (int i = 0; i < inside.Length; i++)
            {
                char c = inside[i];
                // ASCII 标点用位图快速判断，非 ASCII 回退到 char.IsPunctuation
                bool isPunct = c < 128 ? AsciiBitmap.IsPunctuation(c) : char.IsPunctuation(c);
                if (isPunct)
                {
                    if (last < i)
                        result.Add((last, i, false));
                    result.Add((i, i + 1, true));
                    last = i + 1;
                }
            }
            if (last < inside.Length)
                result.Add((last, inside.Length, false));
            return result;
        }
    }
}
