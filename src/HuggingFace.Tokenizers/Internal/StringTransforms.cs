using System.Text;

namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// 字符串变换构建器。
/// 为各种字符串操作构建 (char, change) 变换对，
/// 供 AlignmentTracker.Transform 使用。
/// </summary>
internal static class StringTransforms
{
    /// <summary>
    /// 将小写变换对直接写入 Span，避免 List 堆分配。
    /// 返回实际写入的元素数。
    /// </summary>
    public static int WriteLowercase(string input, Span<(char Char, int Change)> output)
        => WriteLowercase(input.AsSpan(), output);

    /// <summary>
    /// 将小写变换对直接写入 Span（ReadOnlySpan 重载，避免 string 分配）。
    /// </summary>
    public static int WriteLowercase(ReadOnlySpan<char> input, Span<(char Char, int Change)> output)
    {
        int pos = 0;
        Span<char> loweredBuf = stackalloc char[4];
        foreach (var rune in input.EnumerateRunes())
        {
            int loweredLen = Rune.ToLowerInvariant(rune).EncodeToUtf16(loweredBuf);
            int runeLen = rune.Utf16SequenceLength;
            int extra = runeLen - 1;

            output[pos++] = (loweredBuf[0], -extra);
            for (int i = 1; i < loweredLen; i++)
                output[pos++] = (loweredBuf[i], 1);
        }
        return pos;
    }

    /// <summary>
    /// 将大写变换对直接写入 Span，避免 List 堆分配。
    /// 返回实际写入的元素数。
    /// </summary>
    public static int WriteUppercase(string input, Span<(char Char, int Change)> output)
        => WriteUppercase(input.AsSpan(), output);

    /// <summary>
    /// 将大写变换对直接写入 Span（ReadOnlySpan 重载，避免 string 分配）。
    /// </summary>
    public static int WriteUppercase(ReadOnlySpan<char> input, Span<(char Char, int Change)> output)
    {
        int pos = 0;
        Span<char> upperedBuf = stackalloc char[4];
        foreach (var rune in input.EnumerateRunes())
        {
            int upperedLen = Rune.ToUpperInvariant(rune).EncodeToUtf16(upperedBuf);
            int runeLen = rune.Utf16SequenceLength;
            int extra = runeLen - 1;

            output[pos++] = (upperedBuf[0], -extra);
            for (int i = 1; i < upperedLen; i++)
                output[pos++] = (upperedBuf[i], 1);
        }
        return pos;
    }

    /// <summary>
    /// 构建字符过滤变换对（Rune 版本，正确处理补充平面字符）。
    /// </summary>
    /// <returns>(变换对, initialOffset)。</returns>
    public static (List<(char Char, int Change)> Transforms, int InitialOffset) BuildFilter(
        string input, Func<Rune, bool> predicate)
        => BuildFilter(input.AsSpan(), predicate);

    /// <summary>
    /// 构建字符过滤变换对（ReadOnlySpan 重载，避免 string 分配）。
    /// </summary>
    public static (List<(char Char, int Change)> Transforms, int InitialOffset) BuildFilter(
        ReadOnlySpan<char> input, Func<Rune, bool> predicate)
    {
        int removed = 0;
        int removedStart = 0;
        var result = new List<(char, int)>(input.Length);
        char? lastC = null;
        Span<char> runeBuf = stackalloc char[4];

        foreach (var rune in input.EnumerateRunes())
        {
            if (predicate(rune))
            {
                // 使用 EncodeToUtf16 替代 ToString()，避免堆分配
                int runeLen = rune.EncodeToUtf16(runeBuf);
                if (lastC.HasValue)
                    result.Add((lastC.Value, -removed));
                else
                    removedStart = removed;
                lastC = runeBuf[runeLen - 1];
                // 对于多 char 的 Rune，第一个 char 作为 lastC 的前导
                for (int i = 0; i < runeLen - 1; i++)
                {
                    if (result.Count == 0 && removedStart == removed)
                        removedStart = removed;
                    result.Add((runeBuf[i], 0));
                }
                removed = 0;
            }
            else
            {
                removed += rune.Utf16SequenceLength;
            }
        }
        if (lastC.HasValue)
            result.Add((lastC.Value, -removed));

        return (result, removedStart);
    }

    /// <summary>
    /// 构建首尾空白去除变换对。
    /// </summary>
    /// <returns>(变换对, initialOffset)。</returns>
    public static (List<(char Char, int Change)> Transforms, int InitialOffset) BuildStrip(
        string input, bool left, bool right)
        => BuildStrip(input.AsSpan(), left, right);

    /// <summary>
    /// 构建首尾空白去除变换对（ReadOnlySpan 重载，避免 string 分配）。
    /// </summary>
    public static (List<(char Char, int Change)> Transforms, int InitialOffset) BuildStrip(
        ReadOnlySpan<char> input, bool left, bool right)
    {
        int leadingSpaces = 0;
        int trailingSpaces = 0;

        if (left)
        {
            foreach (var c in input)
            {
                if (char.IsWhiteSpace(c)) leadingSpaces++;
                else break;
            }
        }
        if (right)
        {
            for (int i = input.Length - 1; i >= leadingSpaces; i--)
            {
                if (char.IsWhiteSpace(input[i])) trailingSpaces++;
                else break;
            }
        }

        if (leadingSpaces > 0 || trailingSpaces > 0)
        {
            int count = input.Length;
            var result = new List<(char, int)>(count);
            for (int i = 0; i < count; i++)
            {
                if (i < leadingSpaces || i >= count - trailingSpaces)
                    continue;
                else if (i == count - trailingSpaces - 1)
                    result.Add((input[i], -trailingSpaces));
                else
                    result.Add((input[i], 0));
            }
            return (result, leadingSpaces);
        }

        return (new List<(char, int)>(), 0);
    }

    /// <summary>
    /// 构建 Prepend 的变换对。
    /// content[0] change=0（替换第一个字符），content[1..] change=1（插入），
    /// 原始第一个字符 change=1（插入追加）。
    /// </summary>
    public static List<(char Char, int Change)> BuildPrepend(string content, string firstRuneStr)
    {
        var result = new List<(char, int)>(content.Length + firstRuneStr.Length);
        for (int i = 0; i < content.Length; i++)
            result.Add((content[i], i == 0 ? 0 : 1));
        foreach (var ch in firstRuneStr)
            result.Add((ch, 1));
        return result;
    }

    /// <summary>
    /// 将 Prepend 变换对直接写入 Span。
    /// </summary>
    public static int WritePrepend(string content, string firstRuneStr, Span<(char Char, int Change)> output)
        => WritePrepend(content.AsSpan(), firstRuneStr.AsSpan(), output);

    /// <summary>
    /// 将 Prepend 变换对直接写入 Span（Span 重载，零分配）。
    /// </summary>
    public static int WritePrepend(ReadOnlySpan<char> content, ReadOnlySpan<char> firstRuneStr, Span<(char Char, int Change)> output)
    {
        int pos = 0;
        for (int i = 0; i < content.Length; i++)
            output[pos++] = (content[i], i == 0 ? 0 : 1);
        for (int i = 0; i < firstRuneStr.Length; i++)
            output[pos++] = (firstRuneStr[i], 1);
        return pos;
    }

    /// <summary>
    /// 构建 Append 的变换对。
    /// lastChar change=0（替换最后一个字符），content[0..] change=1（插入）。
    /// </summary>
    public static List<(char Char, int Change)> BuildAppend(string content, string lastRuneStr)
    {
        var result = new List<(char, int)>(lastRuneStr.Length + content.Length);
        foreach (var ch in lastRuneStr)
            result.Add((ch, 0));
        for (int i = 0; i < content.Length; i++)
            result.Add((content[i], 1));
        return result;
    }

    /// <summary>
    /// 将 Append 变换对直接写入 Span。
    /// </summary>
    public static int WriteAppend(string content, string lastRuneStr, Span<(char Char, int Change)> output)
        => WriteAppend(content.AsSpan(), lastRuneStr.AsSpan(), output);

    /// <summary>
    /// 将 Append 变换对直接写入 Span（Span 重载，零分配）。
    /// </summary>
    public static int WriteAppend(ReadOnlySpan<char> content, ReadOnlySpan<char> lastRuneStr, Span<(char Char, int Change)> output)
    {
        int pos = 0;
        for (int i = 0; i < lastRuneStr.Length; i++)
            output[pos++] = (lastRuneStr[i], 0);
        for (int i = 0; i < content.Length; i++)
            output[pos++] = (content[i], 1);
        return pos;
    }

    /// <summary>
    /// 构建字符映射变换对（Rune 版本，正确处理补充平面字符）。
    /// </summary>
    public static List<(char Char, int Change)> BuildMap(string input, Func<Rune, Rune> mapFunc)
    {
        var result = new List<(char, int)>(input.Length);
        (char, int)[]? pooled = null;
        Span<(char, int)> buf = input.Length <= 256
            ? stackalloc (char, int)[input.Length]
            : (pooled = System.Buffers.ArrayPool<(char, int)>.Shared.Rent(input.Length));
        try
        {
            int count = WriteMap(input.AsSpan(), mapFunc, buf);
            for (int i = 0; i < count; i++)
                result.Add(buf[i]);
        }
        finally
        {
            if (pooled is not null) System.Buffers.ArrayPool<(char, int)>.Shared.Return(pooled);
        }
        return result;
    }

    /// <summary>
    /// 将字符映射变换对直接写入 Span。
    /// </summary>
    public static int WriteMap(string input, Func<Rune, Rune> mapFunc, Span<(char Char, int Change)> output)
        => WriteMap(input.AsSpan(), mapFunc, output);

    /// <summary>
    /// 将字符映射变换对直接写入 Span（ReadOnlySpan 重载，避免 string 分配）。
    /// </summary>
    public static int WriteMap(ReadOnlySpan<char> input, Func<Rune, Rune> mapFunc, Span<(char Char, int Change)> output)
    {
        int pos = 0;
        Span<char> mappedBuf = stackalloc char[4];
        foreach (var rune in input.EnumerateRunes())
        {
            int mappedLen = mapFunc(rune).EncodeToUtf16(mappedBuf);
            int runeLen = rune.Utf16SequenceLength;
            int extra = runeLen - 1;

            output[pos++] = (mappedBuf[0], -extra);
            for (int i = 1; i < mappedLen; i++)
                output[pos++] = (mappedBuf[i], 1);
        }
        return pos;
    }

    /// <summary>
    /// 构建基于查表的去音标变换对（优化路径）。
    /// 对拉丁扩展字符直接查表输出 base char，跳过 ICU NFD 分解。
    ///
    /// 变换语义与 BuildFilter 一致：
    /// - 保留的字符：change = -removed（消耗前面跳过的字符）
    /// - 跳过的字符：累加到 removed 计数
    /// - 插入的字符（多字节 Rune 的后续 char）：change = 1
    /// </summary>
    public static (List<(char Char, int Change)> Transforms, int InitialOffset) BuildStripAccents(string input)
        => BuildStripAccents(input.AsSpan());

    /// <summary>
    /// 构建基于查表的去音标变换对（ReadOnlySpan 重载，避免 string 分配）。
    /// </summary>
    public static (List<(char Char, int Change)> Transforms, int InitialOffset) BuildStripAccents(ReadOnlySpan<char> input)
    {
        int removed = 0;
        int removedStart = 0;
        var result = new List<(char, int)>(input.Length);
        char? lastC = null;
        Span<char> runeBuf = stackalloc char[4];

        foreach (var rune in input.EnumerateRunes())
        {
            bool kept = false;

            if (rune.IsBmp)
            {
                char c = (char)rune.Value;

                // 查表：拉丁扩展字符 → 输出 base char
                if (LatinDecompTable.TryGetBaseChar(c, out var baseChar))
                {
                    if (lastC.HasValue)
                        result.Add((lastC.Value, -removed));
                    else
                        removedStart = removed;
                    lastC = baseChar;
                    removed = 0;
                    kept = true;
                }
                // combining mark 单独出现 → 不保留
                else if (LatinDecompTable.IsCombiningMark(c))
                {
                    // 不保留，累加 removed
                }
                else
                {
                    // 其他字符原样保留
                    if (lastC.HasValue)
                        result.Add((lastC.Value, -removed));
                    else
                        removedStart = removed;
                    lastC = c;
                    removed = 0;
                    kept = true;
                }
            }
            else
            {
                // 补充平面字符（emoji 等）原样保留
                // 使用 EncodeToUtf16 替代 ToString()，避免堆分配
                int runeLen = rune.EncodeToUtf16(runeBuf);
                if (lastC.HasValue)
                    result.Add((lastC.Value, -removed));
                else
                    removedStart = removed;

                // 多 char Rune：前面的 char 作为普通输出，最后一个作为 lastC
                for (int i = 0; i < runeLen - 1; i++)
                    result.Add((runeBuf[i], 0));
                lastC = runeBuf[runeLen - 1];
                removed = 0;
                kept = true;
            }

            if (!kept)
            {
                removed += rune.Utf16SequenceLength;
            }
        }

        if (lastC.HasValue)
            result.Add((lastC.Value, -removed));

        return (result, removedStart);
    }

    /// <summary>
    /// 在 source 中查找所有 pattern 出现并替换为 replacement。
    /// 一次性分配构建结果，零中间分配。
    /// </summary>
    internal static string SpanReplace(ReadOnlySpan<char> source, ReadOnlySpan<char> pattern, ReadOnlySpan<char> replacement)
    {
        if (pattern.IsEmpty) return source.ToString();

        int matchCount = 0;
        int capacity = source.Length;
        int pos = 0;
        while (pos <= source.Length - pattern.Length)
        {
            if (source.Slice(pos).StartsWith(pattern))
            {
                matchCount++;
                capacity += replacement.Length - pattern.Length;
                pos += pattern.Length;
            }
            else
            {
                pos++;
            }
        }

        if (matchCount == 0) return source.ToString();

        return string.Create(capacity, (sourceStr: source.ToString(), patternStr: pattern.ToString(), replacementStr: replacement.ToString(), matchCount),
            static (dest, s) =>
            {
                ReadOnlySpan<char> src = s.sourceStr.AsSpan();
                ReadOnlySpan<char> pat = s.patternStr.AsSpan();
                ReadOnlySpan<char> rep = s.replacementStr.AsSpan();
                int written = 0, p = 0;

                while (p <= src.Length - pat.Length)
                {
                    if (src.Slice(p).StartsWith(pat))
                    {
                        rep.CopyTo(dest.Slice(written));
                        written += rep.Length;
                        p += pat.Length;
                    }
                    else
                    {
                        dest[written++] = src[p];
                        p++;
                    }
                }
                src.Slice(p).CopyTo(dest.Slice(written));
            });
    }
}
