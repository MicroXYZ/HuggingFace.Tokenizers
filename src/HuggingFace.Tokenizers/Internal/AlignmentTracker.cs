using System.Buffers;
using System.Text;
using HuggingFace.Tokenizers.Abstractions;
using SysEncoding = System.Text.Encoding;

namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// 对齐追踪引擎。
/// 管理 NormalizedString 的 UTF-8 字节存储、偏移映射和对齐列表。
/// 所有字符串变换操作最终通过此引擎的 Transform 原语执行。
/// </summary>
internal sealed class AlignmentTracker
{
    // ── ThreadLocal 池化：复用对齐列表，减少 GC 压力 ──
    private static readonly ThreadLocal<List<(int Start, int End)>?> s_pooledAlignments = new();
    private const int MaxPoolCapacity = 4096;

    // ── UTF-8 字节存储（normalized） ──
    private byte[] _normalizedUtf8;
    private int[] _normalizedIndexMapping;
    private int[] _normalizedByteEndMapping;
    private int _normalizedUtf8Length;

    // ── UTF-8 字节存储（original） ──
    private byte[] _originalUtf8;
    private int[] _originalIndexMapping;
    private int[] _originalByteEndMapping;
    private int _originalUtf8Length;

    // ── 对齐映射 ──
    private List<(int Start, int End)> _alignments;
    private int _originalShift;

    // ── 延迟对齐：NFC-inert 文本跳过 150K alignment 初始化 ──
    private bool _isIdentity;

    // ── normalized 字符缓冲区（E6 优化：避免中间 string 分配） ──
    private char[] _normalizedChars;
    private int _normalizedCharCount;
    private string? _normalizedString; // 延迟 materialize

    // ── 延迟重建标记 ──
    private bool _utf8Dirty;

    /// <summary>对齐列表（normalized 中每个字节对应 original 的字节范围）。</summary>
    public IReadOnlyList<(int Start, int End)> Alignments
    {
        get
        {
            if (_isIdentity) MaterializeFromIdentity();
            return _alignments;
        }
    }

    /// <summary>原始字符串偏移。</summary>
    public int OriginalShift => _originalShift;

    /// <summary>normalized 字符缓冲区长度。</summary>
    public int NormalizedCharCount => _normalizedCharCount;

    /// <summary>normalized 字符缓冲区的 Span 视图（零分配）。</summary>
    public ReadOnlySpan<char> GetNormalizedChars() => _normalizedChars.AsSpan(0, _normalizedCharCount);

    /// <summary>延迟 materialize 的 normalized 字符串。</summary>
    public string GetNormalizedString()
        => _normalizedString ??= new string(_normalizedChars, 0, _normalizedCharCount);

    /// <summary>normalized UTF-8 字节有效长度。</summary>
    public int NormalizedUtf8Length
    {
        get
        {
            if (_isIdentity)
                return SysEncoding.UTF8.GetByteCount(_normalizedChars.AsSpan(0, _normalizedCharCount));
            if (_utf8Dirty) RebuildNormalizedUtf8();
            return _normalizedUtf8Length;
        }
    }

    /// <summary>original UTF-8 字节有效长度。</summary>
    public int OriginalUtf8Length => _originalUtf8Length;

    // ── 只读访问 ──

    public ReadOnlySpan<byte> GetNormalizedUtf8Bytes()
    {
        if (_isIdentity) MaterializeFromIdentity();
        if (_utf8Dirty)
            RebuildNormalizedUtf8();
        return _normalizedUtf8.AsSpan(0, _normalizedUtf8Length);
    }

    public ReadOnlySpan<byte> GetOriginalUtf8Bytes()
    {
        if (_isIdentity) MaterializeFromIdentity();
        return _originalUtf8.AsSpan(0, _originalUtf8Length);
    }
    public ReadOnlySpan<int> GetNormalizedIndexMapping()
    {
        if (_isIdentity) MaterializeFromIdentity();
        if (_utf8Dirty)
            RebuildNormalizedUtf8();
        return _normalizedIndexMapping.AsSpan(0, _normalizedUtf8Length);
    }

    public ReadOnlySpan<int> GetNormalizedByteEndMapping()
    {
        if (_isIdentity) MaterializeFromIdentity();
        if (_utf8Dirty)
            RebuildNormalizedUtf8();
        return _normalizedByteEndMapping.AsSpan(0, _normalizedUtf8Length);
    }

    /// <summary>
    /// 延迟重建 normalized 的 UTF-8 数据和偏移映射。
    /// 仅在 GetNormalizedUtf8Bytes/GetNormalizedIndexMapping/GetNormalizedByteEndMapping 被调用时执行。
    /// </summary>
    private void RebuildNormalizedUtf8()
    {
        Utf8Helpers.EncodeToUtf8(_normalizedChars.AsSpan(0, _normalizedCharCount),
            out _normalizedUtf8, out _normalizedIndexMapping,
            out _normalizedByteEndMapping, out _normalizedUtf8Length);
        _utf8Dirty = false;
    }

    /// <summary>
    /// 从原始字符串初始化。
    /// </summary>
    public AlignmentTracker(string original) : this(original, identity: false) { }

    /// <summary>
    /// 从原始字符串初始化，可选 identity 模式。
    /// identity=true 时跳过对齐列表和 UTF-8 IndexMapping 初始化，
    /// 仅保留 UTF-8 字节长度。Transform() 时按需重建。
    /// 适用于 NFC-inert 文本：Nfc() 直接短路，永远不触发 Transform。
    /// </summary>
    internal AlignmentTracker(string original, bool identity)
    {
        _isIdentity = identity;

        if (identity)
        {
            // 快速路径：只计算 UTF-8 长度，跳过编码和对齐
            _originalUtf8Length = SysEncoding.UTF8.GetByteCount(original);
            _originalUtf8 = [];
            _originalIndexMapping = [];
            _originalByteEndMapping = [];

            _normalizedChars = new char[original.Length];
            original.AsSpan().CopyTo(_normalizedChars);
            _normalizedCharCount = original.Length;
            _normalizedString = original;

            _normalizedUtf8 = [];
            _normalizedIndexMapping = [];
            _normalizedByteEndMapping = [];
            _normalizedUtf8Length = _originalUtf8Length;

            _alignments = [];
            _originalShift = 0;
            return;
        }

        Utf8Helpers.EncodeToUtf8(original, out _originalUtf8, out _originalIndexMapping,
            out _originalByteEndMapping, out _originalUtf8Length);

        // normalized 初始与 original 相同，共享 UTF-8 数据
        _normalizedUtf8 = _originalUtf8;
        _normalizedIndexMapping = _originalIndexMapping;
        _normalizedByteEndMapping = _originalByteEndMapping;
        _normalizedUtf8Length = _originalUtf8Length;

        // 初始化 normalized 缓冲区（从 original 复制）
        _normalizedChars = new char[original.Length];
        original.AsSpan().CopyTo(_normalizedChars);
        _normalizedCharCount = original.Length;
        _normalizedString = original; // 初始与 original 相同，直接引用

        // 字节级对齐：每个字节映射到自身（1:1）
        // 尝试从 ThreadLocal 池复用列表
        var pooled = s_pooledAlignments.Value;
        if (pooled is not null)
        {
            s_pooledAlignments.Value = null;
            pooled.Clear();
            pooled.Capacity = Math.Max(pooled.Capacity, _originalUtf8Length);
            _alignments = pooled;
        }
        else
        {
            _alignments = new List<(int Start, int End)>(_originalUtf8Length);
        }
        for (int i = 0; i < _originalUtf8Length; i++)
            _alignments.Add((i, i + 1));
        _originalShift = 0;
    }

    /// <summary>
    /// 按需重建 identity 模式下的对齐列表和 UTF-8 数据。
    /// 仅在 Transform() 或 Alignments 访问时调用。
    /// </summary>
    private void MaterializeFromIdentity()
    {
        if (!_isIdentity) return;
        _isIdentity = false;

        var original = _normalizedString ?? new string(_normalizedChars, 0, _normalizedCharCount);

        // 编码原始 UTF-8
        Utf8Helpers.EncodeToUtf8(original, out _originalUtf8, out _originalIndexMapping,
            out _originalByteEndMapping, out _originalUtf8Length);

        // 共享
        _normalizedUtf8 = _originalUtf8;
        _normalizedIndexMapping = _originalIndexMapping;
        _normalizedByteEndMapping = _originalByteEndMapping;
        _normalizedUtf8Length = _originalUtf8Length;
        _utf8Dirty = false;

        // 构建对齐列表
        var pooled = s_pooledAlignments.Value;
        if (pooled is not null)
        {
            s_pooledAlignments.Value = null;
            pooled.Clear();
            pooled.Capacity = Math.Max(pooled.Capacity, _originalUtf8Length);
            _alignments = pooled;
        }
        else
        {
            _alignments = new List<(int Start, int End)>(_originalUtf8Length);
        }
        for (int i = 0; i < _originalUtf8Length; i++)
            _alignments.Add((i, i + 1));
    }

    /// <summary>
    /// 从已有数据初始化（用于 Slice/Split 构造子 NormalizedString）。
    /// </summary>
    public AlignmentTracker(
        string original,
        string normalized,
        List<(int Start, int End)> alignments,
        int originalShift)
    {
        _alignments = alignments;
        _originalShift = originalShift;

        // 初始化 normalized 缓冲区
        _normalizedChars = new char[normalized.Length];
        normalized.AsSpan().CopyTo(_normalizedChars);
        _normalizedCharCount = normalized.Length;
        _normalizedString = normalized;

        Utf8Helpers.EncodeToUtf8(original, out _originalUtf8, out _originalIndexMapping,
            out _originalByteEndMapping, out _originalUtf8Length);
        Utf8Helpers.EncodeToUtf8(normalized, out _normalizedUtf8, out _normalizedIndexMapping,
            out _normalizedByteEndMapping, out _normalizedUtf8Length);
    }

    /// <summary>
    /// 核心变换方法：使用 (char, change) 对变换 normalized 字符串，同步更新对齐。
    /// E6 优化：写入内部 buffer 而非创建中间 string，仅在 GetNormalizedString() 时延迟 materialize。
    /// </summary>
    /// <param name="transformations">变换对序列。</param>
    /// <param name="initialOffset">开头跳过的旧字符数。</param>
    public void Transform(
        ReadOnlySpan<(char Char, int Change)> transformations,
        int initialOffset)
    {
        // identity 模式：按需重建对齐列表
        if (_isIdentity) MaterializeFromIdentity();

        // 保存旧对齐列表的引用，读取旧数据写入新列表
        var oldAlignments = _alignments;

        // 尝试从 ThreadLocal 池复用新列表
        var pooled = s_pooledAlignments.Value;
        if (pooled is not null)
        {
            s_pooledAlignments.Value = null;
            pooled.Clear();
            pooled.Capacity = Math.Max(pooled.Capacity, transformations.Length);
            _alignments = pooled;
        }
        else
        {
            _alignments = new List<(int Start, int End)>(transformations.Length);
        }

        // 确保 buffer 足够大
        if (_normalizedChars.Length < transformations.Length)
        {
            // 归还旧 buffer 不需要（ArrayPool 会自动管理）
            _normalizedChars = new char[Math.Max(transformations.Length, _normalizedChars.Length * 2)];
        }

        int charCount = 0;
        int oldIdx = initialOffset;

        for (int i = 0; i < transformations.Length; i++)
        {
            var (c, change) = transformations[i];
            _normalizedChars[charCount++] = c;

            if (change > 0)
            {
                // 插入：不消耗旧字符
                if (oldIdx > 0 && oldIdx <= oldAlignments.Count)
                    _alignments.Add(oldAlignments[oldIdx - 1]);
                else if (initialOffset > 0 && oldAlignments.Count > 0)
                    _alignments.Add(oldAlignments[0]);
                else
                    _alignments.Add((0, 0));
            }
            else if (change == 0)
            {
                // 替换：1 对 1
                if (oldIdx < oldAlignments.Count)
                    _alignments.Add(oldAlignments[oldIdx]);
                else
                    _alignments.Add((_originalShift + oldIdx, _originalShift + oldIdx + 1));
                oldIdx++;
            }
            else // change < 0
            {
                // 替换 + 移除
                int removed = -change;
                if (oldIdx < oldAlignments.Count)
                    _alignments.Add(oldAlignments[oldIdx]);
                else
                    _alignments.Add((_originalShift + oldIdx, _originalShift + oldIdx + 1));
                oldIdx += 1 + removed;
            }
        }

        _normalizedCharCount = charCount;
        _normalizedString = null; // 使延迟 materialize 缓存失效
        _utf8Dirty = true;

        // 将旧对齐列表归还 ThreadLocal 池（容量限制防止无限增长）
        if (oldAlignments.Capacity <= MaxPoolCapacity && s_pooledAlignments.Value is null)
            s_pooledAlignments.Value = oldAlignments;
    }

    /// <summary>
    /// 范围变换：变换 normalized 字符串的指定范围。
    /// E6 优化：从内部 buffer 读取，无需传入 string 参数。
    /// </summary>
    public void TransformRange(
        OffsetReferential referential,
        Range range,
        ReadOnlySpan<(char Char, int Change)> transformations,
        int initialOffset,
        int normalizedLength,
        Func<OffsetReferential, Range, Range?> convertOffsets)
    {
        int nStart, nEnd;

        if (referential == OffsetReferential.Original)
        {
            var converted = convertOffsets(OffsetReferential.Original, range);
            if (converted is null) return;
            nStart = converted.Value.Start.GetOffset(normalizedLength);
            nEnd = converted.Value.End.GetOffset(normalizedLength);
        }
        else
        {
            nStart = range.Start.GetOffset(normalizedLength);
            nEnd = range.End.GetOffset(normalizedLength);
        }

        // 构建完整变换：prefix + transformed range + suffix
        int totalLength = nStart + transformations.Length + (_normalizedCharCount - nEnd);
        (char, int)[]? pooled = null;
        Span<(char, int)> fullTransform = totalLength <= 256
            ? stackalloc (char, int)[totalLength]
            : (pooled = ArrayPool<(char, int)>.Shared.Rent(totalLength));

        try
        {
            int pos = 0;
            for (int i = 0; i < nStart; i++)
                fullTransform[pos++] = (_normalizedChars[i], 0);

            transformations.CopyTo(fullTransform.Slice(pos));
            pos += transformations.Length;

            for (int i = nEnd; i < _normalizedCharCount; i++)
                fullTransform[pos++] = (_normalizedChars[i], 0);

            Transform(fullTransform.Slice(0, pos), initialOffset);
        }
        finally
        {
            if (pooled is not null)
                ArrayPool<(char, int)>.Shared.Return(pooled);
        }
    }

    /// <summary>
    /// 扩展对齐范围到 original 的字节范围。
    /// </summary>
    public static Range? ExpandAlignments(IReadOnlyList<(int Start, int End)> alignments, int start, int end)
    {
        if (start >= end) return null;
        int minStart = alignments[start].Start;
        int maxEnd = alignments[end - 1].End;
        return minStart..maxEnd;
    }

    /// <summary>
    /// 替换对齐列表并标记延迟重建。
    /// E6 优化：接受 buffer 而非 string。
    /// </summary>
    public void ReplaceAlignments(List<(int Start, int End)> newAlignments, char[] newBuffer, int newLength)
    {
        _alignments = newAlignments;
        _normalizedChars = newBuffer;
        _normalizedCharCount = newLength;
        _normalizedString = null;
        _utf8Dirty = true;
    }
}
