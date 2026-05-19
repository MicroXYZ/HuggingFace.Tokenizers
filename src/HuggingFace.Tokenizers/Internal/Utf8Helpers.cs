using System.Buffers;
using System.Diagnostics;
using System.Text;
using SysEncoding = System.Text.Encoding;

namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// UTF-8 字节级操作工具类。
/// 参考 ML.NET TiktokenTokenizer 的 EncodeToUtf8 + indexMapping 模式，
/// 提供 UTF-16 string 与 UTF-8 byte[] 之间的转换和偏移映射。
/// </summary>
internal static class Utf8Helpers
{
    /// <summary>
    /// 将 UTF-16 string 编码为 UTF-8 字节数组，同时构建偏移映射。
    /// </summary>
    /// <param name="text">输入的 UTF-16 字符串。</param>
    /// <param name="utf8Bytes">输出的 UTF-8 字节数组。</param>
    /// <param name="indexMapping">byte→char 起始偏移映射。</param>
    /// <param name="byteEndMapping">byte→char 排他结束偏移映射。</param>
    /// <param name="utf8Length">UTF-8 字节的实际有效长度。</param>
    public static void EncodeToUtf8(
        string text,
        out byte[] utf8Bytes,
        out int[] indexMapping,
        out int[] byteEndMapping,
        out int utf8Length)
    {
        EncodeToUtf8(text.AsSpan(), out utf8Bytes, out indexMapping, out byteEndMapping, out utf8Length);
    }

    /// <summary>
    /// 将 UTF-16 ReadOnlySpan 编码为 UTF-8 字节数组，同时构建偏移映射。
    /// 内部使用 ArrayPool 租用缓冲区，调用方负责归还。
    /// </summary>
    public static void EncodeToUtf8(
        ReadOnlySpan<char> text,
        out byte[] utf8Bytes,
        out int[] indexMapping,
        out int[] byteEndMapping,
        out int utf8Length)
    {
        if (text.IsEmpty)
        {
            utf8Bytes = [];
            indexMapping = [];
            byteEndMapping = [];
            utf8Length = 0;
            return;
        }

        int maxUtf8Length = SysEncoding.UTF8.GetMaxByteCount(text.Length);
        utf8Bytes = new byte[maxUtf8Length];
        indexMapping = new int[maxUtf8Length];
        byteEndMapping = new int[maxUtf8Length];

        utf8Length = EncodeToUtf8(text, utf8Bytes, indexMapping, byteEndMapping);
    }

    /// <summary>
    /// 将 UTF-16 ReadOnlySpan 编码到调用方提供的 buffer，同时构建偏移映射。
    /// 调用方使用 stackalloc 或 ArrayPool 传入 destination/indexMapping/byteEndMapping。
    /// 如果 buffer 不够，自动从 ArrayPool 租用并通过 out 参数返回（调用方负责归还）。
    /// </summary>
    /// <returns>写入的 UTF-8 字节数。</returns>
    public static int EncodeToUtf8(
        ReadOnlySpan<char> text,
        Span<byte> destination,
        Span<int> indexMapping,
        Span<int> byteEndMapping,
        out byte[]? pooledUtf8,
        out int[]? pooledIndexMapping,
        out int[]? pooledByteEnd)
    {
        pooledUtf8 = null;
        pooledIndexMapping = null;
        pooledByteEnd = null;

        if (text.IsEmpty)
            return 0;

        int maxUtf8Length = SysEncoding.UTF8.GetMaxByteCount(text.Length);

        // 如果调用方传入的 buffer 不够，从 ArrayPool 租用
        if (destination.Length < maxUtf8Length)
        {
            pooledUtf8 = ArrayPool<byte>.Shared.Rent(maxUtf8Length);
            destination = pooledUtf8;
        }
        if (indexMapping.Length < maxUtf8Length)
        {
            pooledIndexMapping = ArrayPool<int>.Shared.Rent(maxUtf8Length);
            indexMapping = pooledIndexMapping;
        }
        if (byteEndMapping.Length < maxUtf8Length)
        {
            pooledByteEnd = ArrayPool<int>.Shared.Rent(maxUtf8Length);
            byteEndMapping = pooledByteEnd;
        }

        return EncodeToUtf8(text, destination, indexMapping, byteEndMapping);
    }

    /// <summary>
    /// 将 UTF-16 ReadOnlySpan 编码为 UTF-8 字节，写入目标 buffer，同时构建偏移映射。
    /// 参考 ML.NET Helpers.EncodeToUtf8 实现。
    ///
    /// indexMapping[byte_index] = 该字节所属的 UTF-16 char 起始索引。
    /// byteEndMapping[byte_index] = 该字节所属字符的 UTF-16 char 排他结束索引。
    ///
    /// 示例："a😀b"
    ///   byte:  0   1   2   3   4   5
    ///   utf8:  61  F0  9F  98  80  62
    ///   index: 0   1   1   1   1   3
    ///   end:   1   3   3   3   3   4
    /// </summary>
    /// <param name="text">输入的 UTF-16 字符 span。</param>
    /// <param name="destination">目标 UTF-8 字节 buffer（必须足够大）。</param>
    /// <param name="indexMapping">byte→char 起始偏移映射 buffer。</param>
    /// <param name="byteEndMapping">byte→char 排他结束偏移映射 buffer。</param>
    /// <returns>写入的 UTF-8 字节数。</returns>
    public static int EncodeToUtf8(
        ReadOnlySpan<char> text,
        Span<byte> destination,
        Span<int> indexMapping,
        Span<int> byteEndMapping)
    {
        Debug.Assert(destination.Length >= SysEncoding.UTF8.GetMaxByteCount(text.Length));
        Debug.Assert(indexMapping.Length >= destination.Length);
        Debug.Assert(byteEndMapping.Length >= destination.Length);

        // 使用 SIMD 优化的 Utf8.FromUtf16 编码
        System.Text.Unicode.Utf8.FromUtf16(text, destination, out _, out int bytesWritten);

        // 手动构建偏移映射（Utf8 API 不提供逐字节映射）
        BuildIndexMappings(text, bytesWritten, indexMapping, byteEndMapping);

        return bytesWritten;
    }

    /// <summary>
    /// 构建 UTF-8 字节到 UTF-16 char 的偏移映射。
    /// indexMapping[byte_index] = 该字节所属的 UTF-16 char 起始索引。
    /// byteEndMapping[byte_index] = 该字节所属字符的 UTF-16 char 排他结束索引。
    /// </summary>
    private static void BuildIndexMappings(
        ReadOnlySpan<char> text, int utf8Length,
        Span<int> indexMapping, Span<int> byteEndMapping)
    {
        int byteIdx = 0;
        for (int charIdx = 0; charIdx < text.Length && byteIdx < utf8Length; charIdx++)
        {
            uint c = (uint)text[charIdx];
            int byteCount;

            if (c <= 0x7Fu)
                byteCount = 1;
            else if (c <= 0x7FFu)
                byteCount = 2;
            else if (char.IsSurrogatePair((char)c, charIdx + 1 < text.Length ? text[charIdx + 1] : '\0'))
            {
                byteCount = 4;
                // 代理对：所有 4 字节映射到 charIdx（高代理），end = charIdx + 2
                for (int b = 0; b < 4 && byteIdx + b < utf8Length; b++)
                {
                    indexMapping[byteIdx + b] = charIdx;
                    byteEndMapping[byteIdx + b] = charIdx + 2;
                }
                byteIdx += 4;
                charIdx++; // 跳过低代理
                continue;
            }
            else
                byteCount = 3;

            for (int b = 0; b < byteCount && byteIdx + b < utf8Length; b++)
            {
                indexMapping[byteIdx + b] = charIdx;
                byteEndMapping[byteIdx + b] = charIdx + 1;
            }
            byteIdx += byteCount;
        }
    }

    /// <summary>
    /// 将 UTF-8 字节偏移范围转换为 UTF-16 char 偏移范围。
    /// 使用 byteEndMapping 直接获取排他结束位置，无需原始字符串。
    /// </summary>
    /// <summary>
    /// 从 UTF-8 字节解码一个 Rune（Unicode 标量值）。
    /// </summary>
    public static bool TryDecodeRune(ReadOnlySpan<byte> utf8Bytes, int byteOffset, out Rune rune, out int bytesConsumed)
    {
        var status = Rune.DecodeFromUtf8(utf8Bytes.Slice(byteOffset), out rune, out bytesConsumed);
        return status == OperationStatus.Done;
    }

    /// <summary>
    /// 计算 Rune 的 UTF-8 字节长度。
    /// </summary>
    public static int GetUtf8ByteLength(Rune rune) => rune.Utf8SequenceLength;

    /// <summary>
    /// 计算 Rune 的 UTF-16 char 长度。
    /// </summary>
    public static int GetUtf16CharLength(Rune rune) => rune.Utf16SequenceLength;
}
