using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.PreTokenizers;

/// <summary>
/// 按 Unicode 码点（Rune）将文本拆分为固定长度块的预分词器。
/// 最后一个块可能短于指定长度。
/// 使用 Rune 遍历确保补充平面字符（如 emoji）不会被切碎。
/// </summary>
[TokenizerComponent("FixedLength")]
public sealed class FixedLengthPreTokenizer : IPreTokenizer
{
    /// <summary>
    /// 每个块的固定长度（Unicode 码点，非 UTF-16 字符）。
    /// </summary>
    public int Length { get; }

    /// <summary>
    /// 初始化新的 <see cref="FixedLengthPreTokenizer"/>.
    /// </summary>
    /// <param name="length">The number of Unicode code points per chunk. Must be positive.</param>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="length"/> is not positive.</exception>
    public FixedLengthPreTokenizer(int length = 5)
    {
        if (length <= 0)
            throw new ArgumentOutOfRangeException(nameof(length), length, "Length must be positive.");
        Length = length;
    }

    /// <inheritdoc />
    public void PreTokenize(PreTokenizedString pretokenized)
    {
        pretokenized.Split((_, normalized) =>
        {
            var text = normalized.GetSpan();
            if (text.IsEmpty)
                return Enumerable.Empty<NormalizedString>();

            // 使用 Rune 遍历计算正确的 Unicode 字符位置
            // 收集每个 Rune 在 UTF-16 字符串中的起始位置和 Rune 数量
            var runePositions = new List<int>(); // 每个 Rune 在 text 中的 char 起始索引
            int charIdx = 0;
            foreach (var rune in text.EnumerateRunes())
            {
                runePositions.Add(charIdx);
                charIdx += rune.Utf16SequenceLength;
            }

            int totalRunes = runePositions.Count;
            var parts = new List<NormalizedString>((totalRunes + Length - 1) / Length);

            for (int runeStart = 0; runeStart < totalRunes; runeStart += Length)
            {
                int runeCount = Math.Min(Length, totalRunes - runeStart);
                int charStart = runePositions[runeStart];
                int charEnd = (runeStart + runeCount < totalRunes)
                    ? runePositions[runeStart + runeCount]
                    : text.Length;
                int charLength = charEnd - charStart;

                if (charLength > 0)
                    parts.Add(normalized.Slice(charStart, charLength));
            }

            return parts;
        });
    }
}
