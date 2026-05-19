using HuggingFace.Tokenizers.Internal;
using System.Text.RegularExpressions;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.PreTokenizers;

/// <summary>
/// 按空白和标点边界拆分的预分词器。
/// 使用正则模式 <c>\w+|[^\w\s]+</c>，与 Rust regex crate 的 <c>\w</c> 语义对齐。
///
/// ⚠ 关键差异：Rust regex 的 <c>\w</c> 包含 Unicode Alphabetic（含 SpacingCombiningMark 等），
/// 而 .NET 的 <c>\w</c> 不包含 SpacingCombiningMark (Mc)。
/// 例如天城文 "मॉडल" 中的 उ+0949 (Mc) 在 Rust 中是 word char，.NET 中不是。
/// 因此这里使用 Unicode 宽松模式使行为一致。
/// </summary>
[TokenizerComponent("Whitespace")]
public sealed partial class WhitespacePreTokenizer : IPreTokenizer
{
    /// <summary>
    /// 单例实例，方便使用。
    /// </summary>
    public static readonly WhitespacePreTokenizer Instance = new();

    // ⚠ Rust regex \w 与 .NET \w 的 Unicode 语义不同，必须显式对齐。
    //
    // Rust regex \w = [\p{Alphabetic}\p{Mark}\p{Decimal_Number}\p{Connector_Punctuation}\p{Join_Control}]
    //   参考: https://docs.rs/regex/latest/regex/#unicode
    //         https://github.com/rust-lang/regex/blob/master/regex-syntax/src/unicode_tables/perl_word.rs
    //
    // .NET \w = [\p{L}\p{M}\p{Nd}\p{Pc}]
    //   参考: https://learn.microsoft.com/en-us/dotnet/standard/base-types/character-classes-in-regular-expressions
    //
    // 差异映射（Rust 有而 .NET 缺失）：
    //   \p{Alphabetic} → \p{L} + \p{Nl}（LetterNumber，如 Ⅰ Ⅱ Ⅲ）
    //   \p{Join_Control} → U+200C (ZWNJ) + U+200D (ZWJ)，精确枚举，不用 \p{Cf}（过宽，含 SOFT HYPHEN/ZWSP/BOM 等非 \w 字符）
    //   \p{Mark} → \p{M}（已包含，VS16 U+FE0F 属 Mn）
    //
    // 因此 .NET 等价写法: [\p{L}\p{Nl}\p{M}\p{Nd}\p{Pc}\u200C\u200D]
#if NET7_0_OR_GREATER
    [GeneratedRegex(@"[\p{L}\p{Nl}\p{M}\p{Nd}\p{Pc}\u200C\u200D]+|[^\p{L}\p{Nl}\p{M}\p{Nd}\p{Pc}\u200C\u200D\s]+")]
    private static partial Regex SplitRegex();
#else
    private static Regex SplitRegex() => s_regex;
    private static readonly Regex s_regex = new(@"[\p{L}\p{Nl}\p{M}\p{Nd}\p{Pc}\u200C\u200D]+|[^\p{L}\p{Nl}\p{M}\p{Nd}\p{Pc}\u200C\u200D\s]+", RegexOptions.Compiled);
#endif

    /// <inheritdoc />
    public void PreTokenize(PreTokenizedString pretokenized)
    {
        var pattern = new RegexPattern(SplitRegex());
        pretokenized.Split((_, normalized) =>
        {
            var matches = pattern.FindMatches(normalized.GetSpan());
            return matches
                .Where(m => m.IsMatch)
                .Select(m => normalized.Slice(m.Start, m.End - m.Start));
        });
    }
}
