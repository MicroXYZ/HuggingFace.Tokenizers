using System.Text;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;
using SysEncoding = System.Text.Encoding;

namespace HuggingFace.Tokenizers.Normalizers;

/// <summary>
/// 字节级标准化器，GPT-2 风格。
/// 通过 GPT-2 字节编码器映射每个 UTF-8 字节。
/// 使用变换对保留对齐信息。
/// </summary>
[TokenizerComponent("ByteLevel")]
public sealed class ByteLevelNormalizer : INormalizer
{

    /// <summary>
    /// 标准化给定的 <see cref="NormalizedString"/> 使用字节级编码。
    /// 与 Rust 一致：不前缀前导空格（前导空格由 PreTokenizer 添加）。
    /// 使用变换对保留对齐信息。
    /// </summary>
    public void Normalize(NormalizedString normalized)
    {
        var text = normalized.GetSpan();
        if (text.IsEmpty) return;

        // 计算 UTF-8 字节数（不分配数组）
        int utf8ByteCount = 0;
        foreach (var rune in text.EnumerateRunes())
            utf8ByteCount += rune.Utf8SequenceLength;
        if (utf8ByteCount == 0) return;

        // 为每个 UTF-8 字节创建变换条目
        // 每个 Rune 可能产生多个字节，第一个字节 change=0（替换旧 char），
        // 后续字节 change=1（替换 + 额外消耗 1 个旧 char）
        // 与 Rust byte_level.rs: isize::from(i > 0) 一致
        var transformations = new List<(char Char, int Change)>(utf8ByteCount);
        bool isFirstByteOfRune = true;
        Span<byte> runeBuf = stackalloc byte[4];

        foreach (var rune in text.EnumerateRunes())
        {
            int written = rune.EncodeToUtf8(runeBuf);

            for (int i = 0; i < written; i++)
            {
                // S2: 使用数组查表替代 Dictionary 哈希查找
                char mappedChar = ByteLevelMapping.ByteToChar[runeBuf[i]];
                transformations.Add((mappedChar, isFirstByteOfRune ? 0 : 1));
                isFirstByteOfRune = false;
            }
            isFirstByteOfRune = true;
        }

        normalized.Transform(transformations, 0);
    }
}
