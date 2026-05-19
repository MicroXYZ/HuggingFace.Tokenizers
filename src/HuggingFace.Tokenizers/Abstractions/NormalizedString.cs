using System.Buffers;
using System.Runtime.InteropServices;
using System.Text;
using HuggingFace.Tokenizers.Internal;
using SysEncoding = System.Text.Encoding;

namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 追踪原始字符串及其标准化版本，包含对齐信息。
/// 允许在原始和标准化表示之间转换偏移。
/// 对齐在 UTF-8 字节级别追踪：对于标准化字符串中的每个 UTF-8 字节，
/// 存储其在原始字符串中对应的 (start, end) 字节范围。
///
/// 内部委托 AlignmentTracker 处理对齐追踪，
/// 委托 UnicodeNormalizer / StringTransforms 构建变换对。
///
/// E6 优化：_normalized 字段改为通过 tracker 的内部 buffer 访问，
/// Transform 时不再创建中间 string，仅在 Get() 时延迟 materialize。
/// </summary>
public sealed class NormalizedString
{
    private string _original;
    private readonly ReadOnlyMemory<char> _originalMemory;
    private readonly AlignmentTracker _tracker;

    public NormalizedString(string original)
    {
        _original = original;
        _originalMemory = original.AsMemory();

        // NFC-inert 快速路径：跳过对齐列表和 UTF-8 IndexMapping 初始化
        // 覆盖 CJK、ASCII、大部分常见 Unicode（CCC=0 且无 NFD 分解）
        // 节省：~1000 µs/op（中文 50K）
        if (NormalizationTables.IsAllNfcInert(original))
        {
            _tracker = new AlignmentTracker(original, identity: true);
        }
        else
        {
            _tracker = new AlignmentTracker(original);
        }
    }

    /// <summary>
    /// 接受 ReadOnlySpan&lt;char&gt;，内部物化为 string。
    /// 调用方可以直接传 span，无需提前转换。
    /// </summary>
    public NormalizedString(ReadOnlySpan<char> original) : this(original.ToString()) { }

    internal NormalizedString(
        string original,
        string normalized,
        List<(int Start, int End)> alignments,
        int originalShift)
    {
        _original = original;
        _originalMemory = original.AsMemory();
        _tracker = new AlignmentTracker(original, normalized, alignments, originalShift);
    }

    /// <summary>
    /// 内部构造：使用 ReadOnlyMemory 切片避免 string 拷贝。
    /// _original 从 Memory 立即初始化，消除 null 风险。
    /// </summary>
    internal NormalizedString(
        ReadOnlyMemory<char> originalMemory,
        string normalized,
        List<(int Start, int End)> alignments,
        int originalShift)
    {
        _originalMemory = originalMemory;
        _original = originalMemory.ToString();
        _tracker = new AlignmentTracker(_original, normalized, alignments, originalShift);
    }

    // ── 只读属性 ──

    /// <summary>返回标准化后的字符串（延迟 materialize，避免中间 string 分配）。</summary>
    public string Get() => _tracker.GetNormalizedString();

    /// <summary>返回标准化字符串的 Span 视图（零分配）。</summary>
    public ReadOnlySpan<char> GetSpan() => _tracker.GetNormalizedChars();

    /// <summary>返回原始字符串（延迟初始化，从 Memory 创建）。</summary>
    public string GetOriginal() => _original ??= _originalMemory.ToString();

    /// <summary>标准化字符串的 UTF-16 char 长度。</summary>
    public int Length => _tracker.NormalizedCharCount;

    /// <summary>原始字符串的 UTF-16 char 长度。</summary>
    public int LengthOriginal => _originalMemory.Length;

    /// <summary>标准化字符串的 UTF-8 字节长度。</summary>
    public int ByteLength => _tracker.NormalizedUtf8Length;

    /// <summary>原始字符串的 UTF-8 字节长度。</summary>
    public int ByteLengthOriginal => _tracker.OriginalUtf8Length;

    /// <summary>标准化字符串是否为空。</summary>
    public bool IsEmpty => _tracker.NormalizedUtf8Length == 0;

    /// <summary>获取标准化字符串的 UTF-8 字节。</summary>
    public ReadOnlySpan<byte> GetUtf8Bytes() => _tracker.GetNormalizedUtf8Bytes();

    /// <summary>获取原始字符串的 UTF-8 字节。</summary>
    public ReadOnlySpan<byte> GetOriginalUtf8Bytes() => _tracker.GetOriginalUtf8Bytes();

    /// <summary>获取标准化字符串的 byte→char 映射。</summary>
    public ReadOnlySpan<int> GetIndexMapping() => _tracker.GetNormalizedIndexMapping();

    /// <summary>获取标准化字符串的 byte→char end 映射。</summary>
    public ReadOnlySpan<int> GetByteEndMapping() => _tracker.GetNormalizedByteEndMapping();

    /// <summary>原始偏移范围。</summary>
    public (int Start, int End) OffsetsOriginal()
        => (_tracker.OriginalShift, _tracker.OriginalShift + LengthOriginal);

    // ────────────────────────────────────────────────────────────────────────
    //  Convert offsets
    // ────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// 在不同参考系之间转换偏移。
    /// 如果范围越界则返回 null。
    /// </summary>
    public Range? ConvertOffsets(OffsetReferential referential, Range range)
    {
        int targetStart, targetEnd;
        bool isOriginal;

        if (referential == OffsetReferential.Original)
        {
            targetStart = range.Start.GetOffset(LengthOriginal);
            targetEnd = range.End.GetOffset(LengthOriginal);
            isOriginal = true;
        }
        else
        {
            targetStart = range.Start.GetOffset(Length);
            targetEnd = range.End.GetOffset(Length);
            isOriginal = false;
        }

        if (targetStart == targetEnd)
            return targetStart..targetEnd;
        if (targetStart > targetEnd)
            return null;

        if (isOriginal && _originalMemory.Length == 0 && targetStart == 0 && targetEnd == 0)
            return 0..Length;
        if (!isOriginal && Length == 0 && targetStart == 0 && targetEnd == 0)
            return 0..LengthOriginal;

        var alignments = _tracker.Alignments;

        if (isOriginal)
        {
            // 使用二分查找替代线性扫描，将 O(m×n) 降至 O(m×log n)
            int? start = null, end = null;

            // 二分查找第一个 alignment.Start >= targetStart 的位置
            int lo = 0, hi = alignments.Count;
            while (lo < hi)
            {
                int mid = (lo + hi) >> 1;
                if (alignments[mid].Start < targetStart)
                    lo = mid + 1;
                else
                    hi = mid;
            }

            // 从 lo 开始向右扫描找 start 和 end
            for (int i = lo; i < alignments.Count; i++)
            {
                var alignment = alignments[i];
                if (targetEnd < alignment.Start) break;
                if (start is null && targetStart <= alignment.Start && alignment.Start != alignment.End)
                    start = i;
                if (targetEnd >= alignment.End)
                    end = i + 1;
            }
            if (start is null || end is null) return null;
            return start.Value..end.Value;
        }
        else
        {
            if (targetStart >= alignments.Count) return null;
            int end = Math.Min(targetEnd, alignments.Count);
            if (end == 0) return null;
            return AlignmentTracker.ExpandAlignments(
                alignments, targetStart, end);
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    //  GetRange / GetRangeOriginal
    // ────────────────────────────────────────────────────────────────────────

    /// <summary>获取标准化字符串的范围。</summary>
    public string? GetRange(OffsetReferential referential, Range range)
    {
        if (referential == OffsetReferential.Original)
        {
            var converted = ConvertOffsets(OffsetReferential.Original, range);
            if (converted is null) return null;
            int s = converted.Value.Start.GetOffset(Length);
            int e = converted.Value.End.GetOffset(Length);
            if (s > Length || e > Length) return null;
            return GetSpan().Slice(s, e - s).ToString();
        }
        else
        {
            int s = range.Start.GetOffset(Length);
            int e = range.End.GetOffset(Length);
            if (s > Length || e > Length) return null;
            return GetSpan().Slice(s, e - s).ToString();
        }
    }

    /// <summary>获取原始字符串的范围。</summary>
    public string? GetRangeOriginal(OffsetReferential referential, Range range)
    {
        if (referential == OffsetReferential.Normalized)
        {
            var converted = ConvertOffsets(OffsetReferential.Normalized, range);
            if (converted is null) return null;
            int s = converted.Value.Start.GetOffset(LengthOriginal);
            int e = converted.Value.End.GetOffset(LengthOriginal);
            if (s > LengthOriginal || e > LengthOriginal) return null;
            return _originalMemory.Slice(s, e - s).ToString();
        }
        else
        {
            int s = range.Start.GetOffset(LengthOriginal);
            int e = range.End.GetOffset(LengthOriginal);
            if (s > LengthOriginal || e > LengthOriginal) return null;
            return _originalMemory.Slice(s, e - s).ToString();
        }
    }

    // ────────────────────────────────────────────────────────────────────────
    //  Transform — 委托 AlignmentTracker
    // ────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// 使用 (char, change) 对变换整个字符串。
    /// E6 优化：不再传递 _normalized 参数，tracker 从内部 buffer 读取。
    /// </summary>
    public void Transform(List<(char Char, int Change)> transformations, int initialOffset)
    {
        _tracker.Transform(CollectionsMarshal.AsSpan(transformations), initialOffset);
    }

    // ────────────────────────────────────────────────────────────────────────
    //  Unicode normalization — 委托 UnicodeNormalizer
    // ────────────────────────────────────────────────────────────────────────

    /// <summary>应用 NFD Unicode 标准化。</summary>
    public void Nfd() => ApplyNormalization(NormalizationForm.FormD);

    /// <summary>应用 NFC Unicode 标准化。</summary>
    public void Nfc() => ApplyNormalization(NormalizationForm.FormC);

    /// <summary>应用 NFKD Unicode 标准化。</summary>
    public void Nfkd() => ApplyNormalization(NormalizationForm.FormKD);

    /// <summary>应用 NFKC Unicode 标准化。</summary>
    public void Nfkc() => ApplyNormalization(NormalizationForm.FormKC);

    private void ApplyNormalization(NormalizationForm form)
    {
        var chars = GetSpan();

        // 层 1：纯 ASCII 直接跳过（零开销）
        if (IsAsciiOnly(chars))
            return;

        // 层 1.5：NFC 惰性位图快速判断 — O(n) 位测试，跳过不需要变化的文本
        // 覆盖：CJK、基本拉丁扩展、大部分常见 Unicode（CCC=0 且无 NFD 分解）
        if (form == NormalizationForm.FormC && NormalizationTables.IsAllNfcInert(chars))
            return;

        // 层 2：含 ZWJ 时分段处理，避免 ICU 对 ZWJ 序列抛 ArgumentException
        // 层 3：NFC 使用 NfcWithTransform 消除二次 NFD 分解
        List<(char Char, int Change)> transforms;
        string result;

        if (LatinDecompTable.ContainsZwj(chars))
        {
            result = NormalizeWithZwjHandling(form);
            // ZWJ 路径仍需 BuildNormalizationTransform
            var resultSpan = result.AsSpan();
            if (resultSpan.SequenceEqual(chars)) return;
            transforms = UnicodeNormalizer.BuildNormalizationTransform(chars, result, form);
        }
        else if (form == NormalizationForm.FormC)
        {
            // NFC 快速路径：一次性产出结果 + 变换对，消除二次分解
            (result, transforms) = UnicodeConsistency.NfcWithTransform(chars);
            if (result.AsSpan().SequenceEqual(chars)) return;
        }
        else
        {
            result = NormalizeByForm(chars, form);
            var resultSpan = result.AsSpan();
            if (resultSpan.SequenceEqual(chars)) return;
            transforms = UnicodeNormalizer.BuildNormalizationTransform(chars, result, form);
        }

        _tracker.Transform(CollectionsMarshal.AsSpan(transforms), 0);
    }

    /// <summary>
    /// 含 ZWJ 文本的分段 NFD：按 ZWJ 边界切分，每段独立 normalize。
    /// 保留 ZWJ emoji 完整性，同时正确分解音标字符。
    /// </summary>
    private string NormalizeWithZwjHandling(NormalizationForm form)
    {
        var chars = GetSpan();
        // 预估输出长度
        int maxLen = chars.Length * 2 + 16;
        char[]? pooled = null;
        Span<char> buf = maxLen <= 512
            ? stackalloc char[maxLen]
            : (pooled = System.Buffers.ArrayPool<char>.Shared.Rent(maxLen));
        try
        {
            int pos = 0;
            int segStart = 0;

            for (int i = 0; i < chars.Length; i++)
            {
                if (chars[i] != '\u200D') continue;

                // 找到 ZWJ，normalize 前面的段
                if (i > segStart)
                {
                    pos = NormalizeSegment(chars.Slice(segStart, i - segStart), form, buf, pos);
                }
                // 写入 ZWJ 本身
                if (pos < buf.Length) buf[pos++] = '\u200D';
                segStart = i + 1;
            }
            // 处理最后一段
            if (segStart < chars.Length)
            {
                pos = NormalizeSegment(chars.Slice(segStart, chars.Length - segStart), form, buf, pos);
            }

            return new string(buf.Slice(0, pos));
        }
        finally
        {
            if (pooled is not null) System.Buffers.ArrayPool<char>.Shared.Return(pooled);
        }
    }

    /// <summary>
    /// 根据标准化形式对文本进行 Unicode 标准化。
    /// 使用 UnicodeConsistency 确保 JIT/AOT 一致。
    /// 接受 ReadOnlySpan，内部仅在调用 string.Normalize 时才物化。
    /// </summary>
    private static string NormalizeByForm(ReadOnlySpan<char> text, NormalizationForm form)
    {
        return form switch
        {
            NormalizationForm.FormD => UnicodeConsistency.Nfd(text),
            NormalizationForm.FormC => UnicodeConsistency.Nfc(text),
            NormalizationForm.FormKD => UnicodeConsistency.Nfkd(text),
            NormalizationForm.FormKC => UnicodeConsistency.Nfkc(text),
            _ => text.ToString().Normalize(form),
        };
    }

    /// <summary>
    /// 对单个段（无 ZWJ）做 normalize，结果写入 buf[pos..]，返回新的 pos。
    /// 使用 UnicodeConsistency 确保 JIT/AOT 一致。
    /// </summary>
    private static int NormalizeSegment(ReadOnlySpan<char> segment, NormalizationForm form, Span<char> buf, int pos)
    {
        string normalized = NormalizeByForm(segment, form);

        normalized.AsSpan().CopyTo(buf.Slice(pos));
        return pos + normalized.Length;
    }

    /// <summary>
    /// 快速检测纯 ASCII：所有 char 都在 0x00-0x7F 范围。
    /// 使用 SIMD 批量检查（Vector&lt;ushort&gt; 一次处理 8-32 个 char）。
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    private static bool IsAsciiOnly(ReadOnlySpan<char> chars)
    {
        int i = 0;
        // SIMD 路径
        if (System.Numerics.Vector.IsHardwareAccelerated)
        {
            int vecCount = System.Numerics.Vector<ushort>.Count;
            var threshold = new System.Numerics.Vector<ushort>(0x7F);
            for (; i + vecCount <= chars.Length; i += vecCount)
            {
                var chunk = new System.Numerics.Vector<ushort>(System.Runtime.InteropServices.MemoryMarshal.Cast<char, ushort>(chars.Slice(i, vecCount)));
                if (System.Numerics.Vector.GreaterThanAny(chunk, threshold))
                    return false;
            }
        }
        // 标量处理剩余
        for (; i < chars.Length; i++)
            if (chars[i] > 0x7F) return false;
        return true;
    }

    // ────────────────────────────────────────────────────────────────────────
    //  字符串操作 — 委托 StringTransforms
    // ────────────────────────────────────────────────────────────────────────

    /// <summary>将字符串转为小写。</summary>
    public void Lowercase()
    {
        var chars = GetSpan();
        (char, int)[]? pooled = null;
        Span<(char, int)> buf = chars.Length <= 256
            ? stackalloc (char, int)[chars.Length]
            : (pooled = ArrayPool<(char, int)>.Shared.Rent(chars.Length));
        try
        {
            int count = StringTransforms.WriteLowercase(chars, buf);
            _tracker.Transform(buf.Slice(0, count), 0);
        }
        finally
        {
            if (pooled is not null) ArrayPool<(char, int)>.Shared.Return(pooled);
        }
    }

    /// <summary>将字符串转为大写。</summary>
    public void Uppercase()
    {
        var chars = GetSpan();
        (char, int)[]? pooled = null;
        Span<(char, int)> buf = chars.Length <= 256
            ? stackalloc (char, int)[chars.Length]
            : (pooled = ArrayPool<(char, int)>.Shared.Rent(chars.Length));
        try
        {
            int count = StringTransforms.WriteUppercase(chars, buf);
            _tracker.Transform(buf.Slice(0, count), 0);
        }
        finally
        {
            if (pooled is not null) ArrayPool<(char, int)>.Shared.Return(pooled);
        }
    }

    /// <summary>去除音标符号（使用 Rune，匹配 Rust is_combining_mark：Mn + Mc + Me）。</summary>
    public void StripAccents()
    {
        Nfd();
        Filter(rune =>
        {
            var cat = System.Globalization.CharUnicodeInfo.GetUnicodeCategory(rune.Value);
            return cat != System.Globalization.UnicodeCategory.NonSpacingMark
                && cat != System.Globalization.UnicodeCategory.SpacingCombiningMark
                && cat != System.Globalization.UnicodeCategory.EnclosingMark;
        });
    }

    /// <summary>
    /// 快速去除音标符号（优化路径）。
    ///
    /// 分层策略：
    /// 1. 纯 ASCII → 直接跳过（零开销）
    /// 2. 拉丁扩展区字符 → 查表直接输出 base char（跳过 ICU NFD）
    /// 3. 含 ZWJ → 分段处理
    /// 4. 其他 Unicode → 回退到标准 NFD + Filter
    ///
    /// 性能：比标准 StripAccents 快 2-5x（取决于文本中拉丁扩展字符密度）。
    /// </summary>
    public void StripAccentsFast()
    {
        var chars = GetSpan();

        // 层 1：纯 ASCII 跳过
        if (IsAsciiOnly(chars))
            return;

        // 层 2：含 ZWJ → 分段处理
        if (LatinDecompTable.ContainsZwj(chars))
        {
            StripAccentsWithZwjHandling();
            return;
        }

        // 层 3：检查是否有拉丁扩展字符需要处理
        if (!LatinDecompTable.NeedsLatinDecomp(chars))
        {
            // 无拉丁扩展字符，但可能有其他 combining marks（如希腊语、西里尔语）
            // 回退到标准路径
            StripAccents();
            return;
        }

        // 层 4：查表直接去音标（最快路径）
        StripAccentsByTable();
    }

    /// <summary>
    /// 通过查表直接去除音标，完全跳过 ICU。
    /// 处理拉丁扩展区（U+00C0-U+02FF）的 precomposed 字符。
    /// </summary>
    private void StripAccentsByTable()
    {
        var (transforms, initialOffset) = StringTransforms.BuildStripAccents(GetSpan());
        if (transforms.Count > 0)
            _tracker.Transform(CollectionsMarshal.AsSpan(transforms), initialOffset);
    }

    /// <summary>
    /// 含 ZWJ 文本的分段去音标。
    /// 使用变换对正确处理对齐。
    /// </summary>
    private void StripAccentsWithZwjHandling()
    {
        StripAccentsWithZwjTransforms();
    }

    /// <summary>
    /// 对单个段（无 ZWJ）去除音标，结果写入 buf[pos..]，返回新的 pos。
    /// 优先查表，非拉丁字符原样输出。
    /// </summary>
    private static int StripAccentsSegment(ReadOnlySpan<char> segment, Span<char> buf, int pos)
    {
        Span<char> runeBuf = stackalloc char[4];
        foreach (var rune in segment.EnumerateRunes())
        {
            if (rune.IsBmp)
            {
                char c = (char)rune.Value;
                if (LatinDecompTable.TryGetBaseChar(c, out var baseChar))
                {
                    buf[pos++] = baseChar;
                    continue;
                }
                if (LatinDecompTable.IsCombiningMark(c))
                    continue;
                buf[pos++] = c;
            }
            else
            {
                int len = rune.EncodeToUtf16(runeBuf);
                for (int i = 0; i < len; i++)
                    buf[pos++] = runeBuf[i];
            }
        }
        return pos;
    }

    /// <summary>
    /// 对含 ZWJ 文本构建去音标变换对。
    /// 分段处理：按 ZWJ 切分，每段用 BuildStripAccents 构建变换，
    /// ZWJ 本身作为保留字符插入。
    /// </summary>
    private void StripAccentsWithZwjTransforms()
    {
        var chars = GetSpan();
        var allTransforms = new List<(char Char, int Change)>(chars.Length);
        int totalRemoved = 0;
        int segStart = 0;
        bool isFirstSegment = true;
        int initialOffset = 0;

        for (int i = 0; i < chars.Length; i++)
        {
            if (chars[i] != '\u200D') continue;

            if (i > segStart)
            {
                var seg = chars.Slice(segStart, i - segStart);
                var (segTransforms, segInitialOffset) = StringTransforms.BuildStripAccents(seg);

                if (isFirstSegment)
                {
                    initialOffset = segInitialOffset;
                    isFirstSegment = false;
                }
                else
                {
                    // 非首段：将首字符的 change 调整为补偿前面累积的 removed
                    if (segTransforms.Count > 0)
                    {
                        var (firstChar, firstChange) = segTransforms[0];
                        segTransforms[0] = (firstChar, firstChange - totalRemoved);
                    }
                    totalRemoved = 0;
                }

                // 计算本段 removed 的总量
                int segOriginalLen = seg.Length;
                int segResultLen = 0;
                foreach (var (c, _) in segTransforms) segResultLen++;
                totalRemoved += segOriginalLen - segResultLen;

                allTransforms.AddRange(segTransforms);
            }

            // ZWJ：作为保留字符，change = -totalRemoved（消耗前面跳过的字符）
            if (totalRemoved > 0 && allTransforms.Count > 0)
            {
                // 调整 ZWJ 的 change 来补偿
                allTransforms.Add(('\u200D', -totalRemoved));
                totalRemoved = 0;
            }
            else
            {
                allTransforms.Add(('\u200D', 0));
            }

            segStart = i + 1;
        }

        // 处理最后一段
        if (segStart < chars.Length)
        {
            var seg = chars.Slice(segStart);
            var (segTransforms, segInitialOffset) = StringTransforms.BuildStripAccents(seg);

            if (isFirstSegment)
            {
                initialOffset = segInitialOffset;
            }
            else if (segTransforms.Count > 0)
            {
                var (firstChar, firstChange) = segTransforms[0];
                segTransforms[0] = (firstChar, firstChange - totalRemoved);
            }

            allTransforms.AddRange(segTransforms);
        }

        if (allTransforms.Count > 0)
            _tracker.Transform(CollectionsMarshal.AsSpan(allTransforms), initialOffset);
    }

    /// <summary>使用 Rune 谓词过滤字符，正确处理补充平面字符。</summary>
    public void Filter(Func<Rune, bool> predicate)
    {
        var (transforms, initialOffset) = StringTransforms.BuildFilter(GetSpan(), predicate);
        _tracker.Transform(CollectionsMarshal.AsSpan(transforms), initialOffset);
    }

    /// <summary>去除首尾空白。</summary>
    public void Strip(bool left = true, bool right = true)
    {
        var (transforms, initialOffset) = StringTransforms.BuildStrip(GetSpan(), left, right);
        if (transforms.Count > 0)
            _tracker.Transform(CollectionsMarshal.AsSpan(transforms), initialOffset);
    }

    /// <summary>在字符串前插入内容。</summary>
    public void Prepend(string content)
    {
        var chars = GetSpan();
        if (chars.IsEmpty)
        {
            // 空字符串时直接设置内容，对齐信息无意义（原始为空）
            _tracker.ReplaceAlignments([], content.ToCharArray(), content.Length);
            return;
        }

        Span<char> runeBuf = stackalloc char[2];
        Rune.DecodeFromUtf16(chars, out var firstRune, out _);
        int runeLen = firstRune.EncodeToUtf16(runeBuf);
        var firstRuneSpan = runeBuf.Slice(0, runeLen);
        int bufLen = content.Length + firstRuneSpan.Length;

        (char, int)[]? pooled = null;
        Span<(char, int)> buf = bufLen <= 256
            ? stackalloc (char, int)[bufLen]
            : (pooled = ArrayPool<(char, int)>.Shared.Rent(bufLen));
        try
        {
            int count = StringTransforms.WritePrepend(content.AsSpan(), firstRuneSpan, buf);
            TransformRange(OffsetReferential.Normalized, 0..firstRuneSpan.Length, buf.Slice(0, count), 0);
        }
        finally
        {
            if (pooled is not null) ArrayPool<(char, int)>.Shared.Return(pooled);
        }
    }

    /// <summary>在字符串前插入内容（Span 重载，零分配）。</summary>
    public void Prepend(ReadOnlySpan<char> content)
    {
        var chars = GetSpan();
        if (chars.IsEmpty)
        {
            _tracker.ReplaceAlignments([], content.ToArray(), content.Length);
            return;
        }

        Span<char> runeBuf = stackalloc char[2];
        Rune.DecodeFromUtf16(chars, out var firstRune, out _);
        int runeLen = firstRune.EncodeToUtf16(runeBuf);
        var firstRuneSpan = runeBuf.Slice(0, runeLen);
        int bufLen = content.Length + firstRuneSpan.Length;

        (char, int)[]? pooled = null;
        Span<(char, int)> buf = bufLen <= 256
            ? stackalloc (char, int)[bufLen]
            : (pooled = ArrayPool<(char, int)>.Shared.Rent(bufLen));
        try
        {
            int count = StringTransforms.WritePrepend(content, firstRuneSpan, buf);
            TransformRange(OffsetReferential.Normalized, 0..firstRuneSpan.Length, buf.Slice(0, count), 0);
        }
        finally
        {
            if (pooled is not null) ArrayPool<(char, int)>.Shared.Return(pooled);
        }
    }

    /// <summary>在字符串末尾追加内容。</summary>
    public void Append(string content)
    {
        var chars = GetSpan();
        if (chars.IsEmpty)
        {
            _tracker.ReplaceAlignments([], content.ToCharArray(), content.Length);
            return;
        }

        int lookback = char.IsLowSurrogate(chars[^1]) ? 2 : 1;
        Span<char> runeBuf = stackalloc char[2];
        Rune.DecodeFromUtf16(chars.Slice(chars.Length - lookback), out var lastRune, out _);
        int runeLen = lastRune.EncodeToUtf16(runeBuf);
        var lastRuneSpan = runeBuf.Slice(0, runeLen);
        int lastCharStart = chars.Length - lastRuneSpan.Length;
        int bufLen = lastRuneSpan.Length + content.Length;

        (char, int)[]? pooled = null;
        Span<(char, int)> buf = bufLen <= 256
            ? stackalloc (char, int)[bufLen]
            : (pooled = ArrayPool<(char, int)>.Shared.Rent(bufLen));
        try
        {
            int count = StringTransforms.WriteAppend(content.AsSpan(), lastRuneSpan, buf);
            TransformRange(OffsetReferential.Normalized, lastCharStart..chars.Length, buf.Slice(0, count), 0);
        }
        finally
        {
            if (pooled is not null) ArrayPool<(char, int)>.Shared.Return(pooled);
        }
    }

    /// <summary>使用 Rune 函数映射每个字符，正确处理补充平面字符。</summary>
    public void Map(Func<Rune, Rune> mapFunc)
    {
        var chars = GetSpan();
        (char, int)[]? pooled = null;
        Span<(char, int)> buf = chars.Length <= 256
            ? stackalloc (char, int)[chars.Length]
            : (pooled = ArrayPool<(char, int)>.Shared.Rent(chars.Length));
        try
        {
            int count = StringTransforms.WriteMap(chars, mapFunc, buf);
            _tracker.Transform(buf.Slice(0, count), 0);
        }
        finally
        {
            if (pooled is not null) ArrayPool<(char, int)>.Shared.Return(pooled);
        }
    }

    /// <summary>替换所有匹配项。</summary>
    public int Replace(string pattern, string replacement)
    {
        if (string.IsNullOrEmpty(pattern)) return 0;
        return Replace(pattern.AsSpan(), replacement.AsSpan());
    }

    /// <summary>替换所有匹配项（Span 重载，避免 pattern/replacement 的 string 分配）。</summary>
    public int Replace(ReadOnlySpan<char> pattern, ReadOnlySpan<char> replacement)
    {
        if (pattern.IsEmpty) return 0;

        var chars = GetSpan();
        var matches = new List<(int Start, int End)>();
        int idx = 0;
        while (idx <= chars.Length - pattern.Length)
        {
            if (chars.Slice(idx).StartsWith(pattern))
            {
                matches.Add((idx, idx + pattern.Length));
                idx += pattern.Length;
            }
            else
            {
                idx++;
            }
        }

        return matches.Count > 0 ? ApplyReplacements(matches, replacement) : 0;
    }

    /// <summary>使用正则表达式替换。</summary>
    public int Replace(System.Text.RegularExpressions.Regex regex, string replacement)
    {
        var normalizedStr = Get(); // Regex 需要 string
        var matches = regex.Matches(normalizedStr);
        if (matches.Count == 0) return 0;

        var matchList = new List<(int Start, int End)>(matches.Count);
        foreach (System.Text.RegularExpressions.Match m in matches)
            matchList.Add((m.Index, m.Index + m.Length));

        return ApplyReplacements(matchList, replacement);
    }

    /// <summary>
    /// 通用替换核心逻辑：根据匹配列表执行替换并更新对齐信息。
    /// E6 优化：使用 buffer 替代 string 操作。
    /// </summary>
    private int ApplyReplacements(List<(int Start, int End)> matches, ReadOnlySpan<char> replacement)
    {
        var chars = GetSpan();
        // 预估输出长度上界：原文长度 + 匹配数 × 替换串长度
        int capacity = chars.Length + matches.Count * replacement.Length;
        char[] resultBuffer = new char[capacity];
        var alignments = new List<(int Start, int End)>(capacity);
        int written = 0;
        int lastEnd = 0;
        var trackerAlignments = _tracker.Alignments;

        foreach (var (start, end) in matches)
        {
            // 复制匹配前的原文片段
            for (int i = lastEnd; i < start; i++)
            {
                resultBuffer[written++] = chars[i];
                if (i < trackerAlignments.Count)
                    alignments.Add(trackerAlignments[i]);
            }

            // 写入替换串
            var align = start < trackerAlignments.Count ? trackerAlignments[start] : (_tracker.OriginalShift, _tracker.OriginalShift);
            replacement.CopyTo(resultBuffer.AsSpan(written));
            for (int i = 0; i < replacement.Length; i++)
                alignments.Add(align);
            written += replacement.Length;

            lastEnd = end;
        }

        // 复制最后一段原文
        for (int i = lastEnd; i < chars.Length; i++)
        {
            resultBuffer[written++] = chars[i];
            if (i < trackerAlignments.Count)
                alignments.Add(trackerAlignments[i]);
        }

        _tracker.ReplaceAlignments(alignments, resultBuffer, written);
        return matches.Count;
    }

    /// <summary>清空标准化字符串，返回之前的长度。</summary>
    public int Clear()
    {
        int len = Length;
        _tracker.Transform(ReadOnlySpan<(char, int)>.Empty, len);
        return len;
    }

    // ────────────────────────────────────────────────────────────────────────
    //  Split
    // ────────────────────────────────────────────────────────────────────────

    /// <summary>
    /// 使用模式拆分 NormalizedString。
    /// </summary>
    public IReadOnlyList<(NormalizedString Part, bool IsMatch)> Split(
        SplitDelimiterBehavior behavior,
        Func<ReadOnlySpan<char>, IReadOnlyList<(int Start, int End, bool IsMatch)>> findMatches)
    {
        var matches = findMatches(GetSpan());
        var result = new List<(NormalizedString Part, bool IsMatch)>();
        var trackerAlignments = _tracker.Alignments;

        foreach (var (start, end, isMatch) in matches)
        {
            if (start >= end) continue;

            int alignCount = Math.Min(end - start, trackerAlignments.Count - start);
            var sliceAlignments = trackerAlignments.Count > 0
                ? new List<(int Start, int End)>(
                    CollectionsMarshal.AsSpan((List<(int Start, int End)>)trackerAlignments)
                    .Slice(start, alignCount).ToArray())
                : new List<(int Start, int End)>();
            int sliceOriginalShift = _tracker.OriginalShift +
                (sliceAlignments.Count > 0 ? sliceAlignments[0].Start : start);

            ReadOnlyMemory<char> sliceOriginalMemory;
            if (sliceAlignments.Count > 0)
            {
                int origStart = sliceAlignments[0].Start;
                int origEnd = sliceAlignments[^1].End;
                origStart = Math.Max(0, Math.Min(origStart, _originalMemory.Length));
                origEnd = Math.Max(origStart, Math.Min(origEnd, _originalMemory.Length));
                sliceOriginalMemory = _originalMemory.Slice(origStart, origEnd - origStart);
            }
            else
            {
                sliceOriginalMemory = ReadOnlyMemory<char>.Empty;
            }

            var part = new NormalizedString(
                sliceOriginalMemory,
                GetSpan().Slice(start, end - start).ToString(),
                sliceAlignments,
                sliceOriginalShift);

            if (isMatch)
            {
                switch (behavior)
                {
                    case SplitDelimiterBehavior.Removed:
                        continue;
                    case SplitDelimiterBehavior.Isolated:
                        result.Add((part, true));
                        break;
                    case SplitDelimiterBehavior.MergedWithPrevious:
                        if (result.Count > 0)
                        {
                            var (prev, _) = result[^1];
                            result[^1] = (Merge(prev, part), false);
                        }
                        else
                            result.Add((part, false));
                        break;
                    case SplitDelimiterBehavior.MergedWithNext:
                        result.Add((part, true));
                        break;
                    case SplitDelimiterBehavior.Contiguous:
                        result.Add((part, true));
                        break;
                }
            }
            else
            {
                if (behavior == SplitDelimiterBehavior.MergedWithNext && result.Count > 0)
                {
                    var (last, lastIsMatch) = result[^1];
                    if (lastIsMatch)
                    {
                        result[^1] = (Merge(last, part), false);
                        continue;
                    }
                }
                result.Add((part, false));
            }
        }

        return result;
    }

    private static NormalizedString Merge(NormalizedString a, NormalizedString b)
    {
        var aAlignments = a._tracker.Alignments.ToList();
        var bAlignments = b._tracker.Alignments.ToList();
        var newAlignments = new List<(int Start, int End)>(aAlignments.Count + bAlignments.Count);
        newAlignments.AddRange(aAlignments);
        newAlignments.AddRange(bAlignments);

        // string.Concat 已在 .NET 内部优化为单次 Span 拷贝
        var mergedOriginal = string.Concat(a.GetOriginal(), b.GetOriginal());
        var mergedNormalized = string.Concat(a.GetSpan(), b.GetSpan());

        return new NormalizedString(
            mergedOriginal,
            mergedNormalized,
            newAlignments,
            a._tracker.OriginalShift);
    }

    // ────────────────────────────────────────────────────────────────────────
    //  TransformRange
    // ────────────────────────────────────────────────────────────────────────

    /// <summary>使用 (char, change) 对变换字符串的指定范围。</summary>
    public void TransformRange(
        OffsetReferential referential,
        Range range,
        List<(char Char, int Change)> transformations,
        int initialOffset)
    {
        _tracker.TransformRange(
            referential, range, CollectionsMarshal.AsSpan(transformations), initialOffset,
            Length, ConvertOffsets);
    }

    /// <summary>使用 (char, change) 对变换字符串的指定范围（Span 重载，避免 List 分配）。</summary>
    public void TransformRange(
        OffsetReferential referential,
        Range range,
        ReadOnlySpan<(char Char, int Change)> transformations,
        int initialOffset)
    {
        _tracker.TransformRange(
            referential, range, transformations, initialOffset,
            Length, ConvertOffsets);
    }

    // ────────────────────────────────────────────────────────────────────────
    //  Slice
    // ────────────────────────────────────────────────────────────────────────

    /// <summary>获取此 NormalizedString 的切片。</summary>
    public NormalizedString Slice(int start, int length)
    {
        var chars = GetSpan();
        if (start < 0 || length < 0 || start + length > chars.Length)
            throw new ArgumentOutOfRangeException();

        var trackerAlignments = _tracker.Alignments;
        var sliceAlignments = new List<(int Start, int End)>(length);
        for (int ai = start; ai < start + length; ai++)
            sliceAlignments.Add(trackerAlignments[ai]);
        int sliceOriginalShift = _tracker.OriginalShift +
            (sliceAlignments.Count > 0 ? sliceAlignments[0].Start : 0);

        ReadOnlyMemory<char> sliceOriginalMemory;
        if (sliceAlignments.Count > 0)
        {
            int origStart = sliceAlignments[0].Start;
            int origEnd = sliceAlignments[^1].End;
            origStart = Math.Max(0, Math.Min(origStart, _originalMemory.Length));
            origEnd = Math.Max(origStart, Math.Min(origEnd, _originalMemory.Length));
            sliceOriginalMemory = _originalMemory.Slice(origStart, origEnd - origStart);
        }
        else
        {
            sliceOriginalMemory = ReadOnlyMemory<char>.Empty;
        }

        return new NormalizedString(
            sliceOriginalMemory,
            chars.Slice(start, length).ToString(),
            sliceAlignments,
            sliceOriginalShift);
    }

    /// <summary>使用 Original 或 Normalized range 获取切片。</summary>
    public NormalizedString? Slice(OffsetReferential referential, Range range)
    {
        int start, end;
        if (referential == OffsetReferential.Original)
        {
            var converted = ConvertOffsets(OffsetReferential.Original, range);
            if (converted is null) return null;
            start = converted.Value.Start.GetOffset(Length);
            end = converted.Value.End.GetOffset(Length);
        }
        else
        {
            start = range.Start.GetOffset(Length);
            end = range.End.GetOffset(Length);
        }

        if (start > Length || end > Length || start > end) return null;
        return Slice(start, end - start);
    }

    public override string ToString() => Get();
}
