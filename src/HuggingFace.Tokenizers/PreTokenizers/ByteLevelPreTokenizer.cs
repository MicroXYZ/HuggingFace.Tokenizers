using System.Text;
using System.Text.RegularExpressions;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.PreTokenizers;

/// <summary>
/// GPT-2 风格字节级预分词器。
/// 将文本拆分为字节级片段，每个字节用可见 Unicode 字符表示。
/// 空格替换为 <c>Ġ</c> (U+0120)。
/// </summary>
[TokenizerComponent("ByteLevel")]
public sealed partial class ByteLevelPreTokenizer : IPreTokenizer
{
    /// <summary>
    /// 默认选项的单例实例。
    /// </summary>
    public static readonly ByteLevelPreTokenizer Instance = new();

    private readonly bool _addPrefixSpace;
    private readonly bool _useRegex;
    private readonly bool _trimOffsets;

    /// <summary>是否在输入前添加前导空格。</summary>
    public bool AddPrefixSpace => _addPrefixSpace;

    /// <summary>是否在字节编码前应用 GPT-2 正则拆分。</summary>
    public bool UseRegex => _useRegex;

    /// <summary>是否修剪偏移。</summary>
    public bool TrimOffsets => _trimOffsets;

    /// <summary>
    /// 字节级编码前用于拆分文本的正则模式（GPT-2 风格）。
    /// </summary>
    private const string ByteLevelPattern =
        @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

#if NET7_0_OR_GREATER
    /// <summary>
    /// 编译时生成的正则实例（AOT 友好）。
    /// </summary>
    [GeneratedRegex(ByteLevelPattern)]
    private static partial Regex CachedRegex();
#else
    /// <summary>
    /// 缓存的编译正则实例（与 Rust LazyLock 模式一致）。
    /// </summary>
    private static readonly Regex CachedRegexInstance = new(ByteLevelPattern, RegexOptions.Compiled);
    private static Regex CachedRegex() => CachedRegexInstance;
#endif

    /// <summary>
    /// 初始化新的 <see cref="ByteLevelPreTokenizer"/>.
    /// 默认值与 Rust ByteLevel 一致：addPrefixSpace=true, trimOffsets=true。
    /// </summary>
    /// <param name="addPrefixSpace">是否在输入前添加前导空格。</param>
    /// <param name="useRegex">是否在字节编码前应用 GPT-2 正则拆分。</param>
    /// <param name="trimOffsets">是否修剪偏移。</param>
    public ByteLevelPreTokenizer(
        bool addPrefixSpace = true,
        bool useRegex = true,
        bool trimOffsets = true)
    {
        _addPrefixSpace = addPrefixSpace;
        _useRegex = useRegex;
        _trimOffsets = trimOffsets;
    }

    /// <inheritdoc />
    public void PreTokenize(PreTokenizedString pretokenized)
    {
        pretokenized.Split((_, normalized) =>
        {
            // 对 NormalizedString 本身应用前导空格，以保持索引对齐
            if (_addPrefixSpace && (normalized.Length == 0 || normalized.GetSpan()[0] != ' '))
                normalized.Prepend(" ");

            var text = normalized.GetSpan();

            if (_useRegex)
            {
                var textSpan = text;
                var parts = new List<NormalizedString>();

                foreach (ValueMatch match in CachedRegex().EnumerateMatches(textSpan))
                {
                    if (match.Length == 0) continue;
                    parts.Add(normalized.Slice(match.Index, match.Length));
                }

                return parts;
            }

            // 不使用正则时，将整个字符串视为一个片段
            if (text.Length == 0)
                return Enumerable.Empty<NormalizedString>();
            return new[] { normalized.Slice(0, text.Length) };
        });

        // 标准化：将每个字符转换为其字节级表示
        // （与 Rust ByteLevel 预分词的 normalize 步骤一致）
        pretokenized.Normalize(normalized =>
        {
            var textSpan = normalized.GetSpan();
            if (textSpan.IsEmpty) return;

            // 构建 (char, change) 变换以精确跟踪对齐。
            // 对每个字符，编码为 UTF-8 字节，然后将每个字节映射到其可见 Unicode 字符。
            // 第一个字节的 change=0（替换原始字符），
            // 后续字节的 change=1（插入）。
            var transformations = new List<(char, int)>(textSpan.Length * 2);
            Span<byte> byteBuf = stackalloc byte[4];
            foreach (var rune in textSpan.EnumerateRunes())
            {
                int byteCount = rune.EncodeToUtf8(byteBuf);
                for (int i = 0; i < byteCount; i++)
                {
                    transformations.Add((ByteLevelMapping.ByteToChar[byteBuf[i]], i == 0 ? 0 : 1));
                }
            }
            normalized.Transform(transformations, 0);
        });
    }

    /// <summary>
    /// 将字符串编码为字节级表示。
    /// 每个字节映射到可见 Unicode 字符。
    /// </summary>
    /// <param name="text">输入文本。</param>
    /// <returns>字节级编码字符串。</returns>
    public static string Encode(string text)
    {
        if (text.Length == 0) return string.Empty;

        var bytes = global::System.Text.Encoding.UTF8.GetBytes(text);

        // SIMD 优化：使用基于 Span 的字符映射替代 StringBuilder
        return string.Create(bytes.Length, bytes, (dest, src) =>
        {
            for (int i = 0; i < src.Length; i++)
                dest[i] = ByteLevelMapping.ByteToChar[src[i]];
        });
    }

    /// <summary>
    /// 将字节级编码字符串解码回 UTF-8 文本。
    /// </summary>
    /// <param name="encoded">字节级编码字符串。</param>
    /// <returns>原始 UTF-8 文本。</returns>
    public static string Decode(string encoded)
    {
        var bytes = new byte[encoded.Length];
        for (int i = 0; i < encoded.Length; i++)
            bytes[i] = ByteLevelMapping.CharToByte[encoded[i]];
        return global::System.Text.Encoding.UTF8.GetString(bytes);
    }
}
