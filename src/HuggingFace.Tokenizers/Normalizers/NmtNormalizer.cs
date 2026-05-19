using System.Text;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Normalizers;

/// <summary>
/// 神经机器翻译（NMT）标准化器。
/// 严格匹配 Rust do_nmt() 实现：
/// 1. Filter specific ASCII control characters
/// 2. Map specific whitespace characters to space
/// 不执行额外的标点标准化、空白折叠或去除。
/// </summary>
[TokenizerComponent("Nmt")]
public sealed class NmtNormalizer : INormalizer
{
    /// <summary>
    /// 标准化给定的 <see cref="NormalizedString"/> for NMT preprocessing.
    /// 与 Rust do_nmt 完全一致（使用 Rune 正确处理补充平面字符）。
    /// </summary>
    public void Normalize(NormalizedString normalized)
    {
        // Step 1: Filter out specific ASCII control characters
        // Rust: filter(|c| !matches!(c as u32, 0x0001..=0x0008 | 0x000B | 0x000E..=0x001F | 0x007F | 0x008F | 0x009F))
        normalized.Filter(rune =>
        {
            var cp = rune.Value;
            return !(cp is (>= 0x0001 and <= 0x0008)
                or 0x000B
                or (>= 0x000E and <= 0x001F)
                or 0x007F
                or 0x008F
                or 0x009F);
        });

        // 步骤 2：将特定空白/控制码位映射为空格
        // Rust: .map(|c| match c as u32 { ... => ' ', _ => c })
        normalized.Map(rune =>
        {
            var cp = rune.Value;
            return cp switch
            {
                0x0009 => new Rune(' '),  // TAB
                0x000A => new Rune(' '),  // LINE FEED
                0x000C => new Rune(' '),  // FORM FEED
                0x000D => new Rune(' '),  // CARRIAGE RETURN
                0x1680 => new Rune(' '),  // OGHAM SPACE MARK
                >= 0x200B and <= 0x200F => new Rune(' '),  // ZERO WIDTH SPACE..RIGHT-TO-LEFT MARK
                0x2028 => new Rune(' '),  // LINE SEPARATOR
                0x2029 => new Rune(' '),  // PARAGRAPH SEPARATOR
                0x2581 => new Rune(' '),  // LOWER ONE EIGHTH BLOCK (▁)
                0xFEFF => new Rune(' '),  // ZERO WIDTH NO-BREAK SPACE (BOM)
                0xFFFD => new Rune(' '),  // REPLACEMENT CHARACTER
                _ => rune,
            };
        });
    }
}
