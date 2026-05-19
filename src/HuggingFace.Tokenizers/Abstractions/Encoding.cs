namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 表示分词器的输出 — ID、token、偏移、注意力掩码等。
/// 内部使用 Memory&lt;T&gt; + Copy-on-Write，支持零拷贝 Slice。
/// </summary>
public sealed class Encoding
{
    // ─── 内部存储：Memory&lt;T&gt; + 范围 + 共享标记 ───
    private Memory<uint> _idsMem;
    private Memory<uint> _typeIdsMem;
    private Memory<string> _tokensMem;
    private Memory<uint?> _wordsMem;
    private Memory<(int Start, int End)> _offsetsMem;
    private Memory<uint> _specialMaskMem;
    private Memory<uint> _attentionMem;
    private int _start;
    private int _length;
    private bool _isShared;

    // ─── 构造函数 ───

    /// <summary>
    /// 公共构造函数，接收 T[] 数组。
    /// </summary>
    public Encoding(
        uint[] ids,
        uint[] typeIds,
        string[] tokens,
        uint?[] words,
        (int Start, int End)[] offsets,
        uint[] specialTokensMask,
        uint[] attentionMask,
        List<Encoding>? overflowing = null,
        Dictionary<int, Range>? sequenceRanges = null)
        : this(ids.AsMemory(), typeIds.AsMemory(), tokens.AsMemory(), words.AsMemory(),
               offsets.AsMemory(), specialTokensMask.AsMemory(), attentionMask.AsMemory(),
               start: 0, length: ids.Length, shared: false, overflowing, sequenceRanges)
    { }

    /// <summary>
    /// 内部构造函数，Memory&lt;T&gt; + 范围 + 共享标记。
    /// </summary>
    private Encoding(
        Memory<uint> ids, Memory<uint> typeIds, Memory<string> tokens, Memory<uint?> words,
        Memory<(int Start, int End)> offsets, Memory<uint> specialMask, Memory<uint> attention,
        int start, int length, bool shared,
        List<Encoding>? overflowing = null, Dictionary<int, Range>? sequenceRanges = null)
    {
        _idsMem = ids; _typeIdsMem = typeIds; _tokensMem = tokens; _wordsMem = words;
        _offsetsMem = offsets; _specialMaskMem = specialMask; _attentionMem = attention;
        _start = start; _length = length; _isShared = shared;
        Overflowing = overflowing ?? [];
        SequenceRanges = sequenceRanges ?? [];
    }

    // ─── 零拷贝 Slice ───

    /// <summary>
    /// 零拷贝切片：共享底层数组，只记录范围。
    /// </summary>
    internal Encoding Slice(int start, int length)
    {
        return new Encoding(
            _idsMem, _typeIdsMem, _tokensMem, _wordsMem,
            _offsetsMem, _specialMaskMem, _attentionMem,
            _start + start, length, shared: true);
    }

    // ─── Copy-on-Write ───

    /// <summary>
    /// 标记当前 Encoding 为共享状态（COW）。
    /// 用于缓存的 Encoding，防止 MergeWith 等操作原地修改。
    /// </summary>
    internal void MarkAsShared() => _isShared = true;

    /// <summary>
    /// 确保当前 Encoding 拥有独立的底层数组（非共享）。
    /// 必须在任何修改操作前调用。
    /// </summary>
    private void EnsureWritable()
    {
        if (!_isShared) return;

        _idsMem = IdsSpan.ToArray();
        _typeIdsMem = TypeIdsSpan.ToArray();
        _tokensMem = TokensSpan.ToArray();
        _wordsMem = WordsSpan.ToArray();
        _offsetsMem = OffsetsSpan.ToArray();
        _specialMaskMem = SpecialTokensMaskSpan.ToArray();
        _attentionMem = AttentionMaskSpan.ToArray();
        _start = 0;
        _isShared = false;
    }

    /// <summary>
    /// 确保容量足够，不足时扩容。
    /// </summary>
    private void EnsureCapacity(int required)
    {
        if (_idsMem.Length - _start >= required) return;

        int newCap = Math.Max(required, _length * 2);

        var newIds = new uint[newCap];
        var newTypeIds = new uint[newCap];
        var newTokens = new string[newCap];
        var newWords = new uint?[newCap];
        var newOffsets = new (int Start, int End)[newCap];
        var newSpecial = new uint[newCap];
        var newAttention = new uint[newCap];

        IdsSpan.CopyTo(newIds);
        TypeIdsSpan.CopyTo(newTypeIds);
        TokensSpan.CopyTo(newTokens);
        WordsSpan.CopyTo(newWords);
        OffsetsSpan.CopyTo(newOffsets);
        SpecialTokensMaskSpan.CopyTo(newSpecial);
        AttentionMaskSpan.CopyTo(newAttention);

        _idsMem = newIds;
        _typeIdsMem = newTypeIds;
        _tokensMem = newTokens;
        _wordsMem = newWords;
        _offsetsMem = newOffsets;
        _specialMaskMem = newSpecial;
        _attentionMem = newAttention;
        _start = 0;
        _isShared = false;
    }

    // ─── 基本属性 ───

    /// <summary>编码中的 token 数量。</summary>
    public int Length => _length;
    /// <summary>编码是否为空（无 token）。</summary>
    public bool IsEmpty => _length == 0;
    /// <summary>编码中包含的序列数量。</summary>
    public int SequenceCount => SequenceRanges.Count == 0 ? 1 : SequenceRanges.Count;

    /// <summary>溢出编码列表（截断时产生）。</summary>
    public List<Encoding> Overflowing { get; }
    /// <summary>序列 ID 到偏移范围的映射。</summary>
    public Dictionary<int, Range> SequenceRanges { get; }

    // ─── 公共 API（向后兼容，返回拷贝）───

    public uint[] GetIds() => IdsSpan.ToArray();
    public string[] GetTokens() => TokensSpan.ToArray();
    public uint?[] GetWordIds() => WordsSpan.ToArray();
    public (int Start, int End)[] GetOffsets() => OffsetsSpan.ToArray();
    public uint[] GetTypeIds() => TypeIdsSpan.ToArray();
    public uint[] GetAttentionMask() => AttentionMaskSpan.ToArray();
    public uint[] GetSpecialTokensMask() => SpecialTokensMaskSpan.ToArray();
    public List<Encoding> GetOverflowing() => Overflowing;

    // ─── 零拷贝 Span API（内部使用）───

    /// <summary>获取 Ids 的只读 Span 视图（零分配）。</summary>
    public ReadOnlySpan<uint> IdsSpan => _idsMem.Span.Slice(_start, _length);
    /// <summary>获取 TypeIds 的只读 Span 视图（零分配）。</summary>
    public ReadOnlySpan<uint> TypeIdsSpan => _typeIdsMem.Span.Slice(_start, _length);
    /// <summary>获取 Tokens 的只读 Span 视图（零分配）。</summary>
    public ReadOnlySpan<string> TokensSpan => _tokensMem.Span.Slice(_start, _length);
    /// <summary>获取 Words 的只读 Span 视图（零分配）。</summary>
    public ReadOnlySpan<uint?> WordsSpan => _wordsMem.Span.Slice(_start, _length);
    /// <summary>获取 Offsets 的只读 Span 视图（零分配）。</summary>
    public ReadOnlySpan<(int Start, int End)> OffsetsSpan => _offsetsMem.Span.Slice(_start, _length);
    /// <summary>获取 SpecialTokensMask 的只读 Span 视图（零分配）。</summary>
    public ReadOnlySpan<uint> SpecialTokensMaskSpan => _specialMaskMem.Span.Slice(_start, _length);
    /// <summary>获取 AttentionMask 的只读 Span 视图（零分配）。</summary>
    public ReadOnlySpan<uint> AttentionMaskSpan => _attentionMem.Span.Slice(_start, _length);

    // ─── 序列相关 ───

    public void SetSequenceId(int sequenceId)
    {
        SequenceRanges[sequenceId] = 0..Length;
    }

    public uint?[] GetSequenceIds()
    {
        var sequences = new uint?[Length];
        for (int seqId = 0; seqId < SequenceCount; seqId++)
        {
            var range = SequenceRange(seqId);
            for (int i = range.Start.Value; i < range.End.Value; i++)
                sequences[i] = (uint)seqId;
        }
        return sequences;
    }

    /// <summary>
    /// 获取指定序列 ID 的偏移范围。
    /// </summary>
    public Range SequenceRange(int sequenceId)
    {
        if (SequenceRanges.TryGetValue(sequenceId, out var range))
            return range;
        return SequenceRanges.Count == 0 ? 0..Length : 0..0;
    }

    // ─── 修改操作（COW）───

    public void SetTypeIds(uint[] typeIds)
    {
        EnsureWritable();
        typeIds.AsSpan(0, Math.Min(typeIds.Length, _length))
               .CopyTo(_typeIdsMem.Span);
    }

    /// <summary>
    /// 设置指定索引处的偏移量。
    /// </summary>
    public void SetOffset(int index, (int Start, int End) value)
    {
        EnsureWritable();
        _offsetsMem.Span[index] = value;
    }

    // ─── Offset / word mapping helpers ───

    /// <summary>
    /// 将 token 索引映射到原始字符串中的字符偏移范围。
    /// </summary>
    public (int Start, int End)? TokenToChars(int tokenIndex)
    {
        if (tokenIndex < 0 || tokenIndex >= _length)
            return null;
        return OffsetsSpan[tokenIndex];
    }

    /// <summary>
    /// 将字符位置映射到覆盖该位置的 token 索引。
    /// </summary>
    public int? CharToToken(int charPos, int sequenceId = 0)
    {
        var range = SequenceRange(sequenceId);
        var offsets = OffsetsSpan;
        for (int i = range.Start.Value; i < range.End.Value; i++)
        {
            var (start, end) = offsets[i];
            if (charPos >= start && charPos < end)
                return i;
        }
        return null;
    }

    /// <summary>
    /// 将 token 索引映射到其 word 标识符。
    /// </summary>
    public uint? TokenToWord(int tokenIndex)
    {
        if (tokenIndex < 0 || tokenIndex >= _length)
            return null;
        return WordsSpan[tokenIndex];
    }

    /// <summary>
    /// 将字符位置映射到覆盖该位置的 token 的 word 标识符。
    /// </summary>
    public uint? CharToWord(int charPos, int sequenceId = 0)
    {
        var tokenIdx = CharToToken(charPos, sequenceId);
        return tokenIdx.HasValue ? WordsSpan[tokenIdx.Value] : null;
    }

    /// <summary>
    /// 将 word 标识符映射到 token 范围（起始索引、数量）。
    /// </summary>
    public (int Start, int Count)? WordToTokens(uint wordId, int sequenceId = 0)
    {
        var range = SequenceRange(sequenceId);
        var words = WordsSpan;
        int start = -1;
        int count = 0;

        for (int i = range.Start.Value; i < range.End.Value; i++)
        {
            if (words[i] == wordId)
            {
                if (start < 0) start = i;
                count++;
            }
        }

        return start >= 0 ? (start, count) : null;
    }

    /// <summary>
    /// 将 word 标识符映射到其覆盖的字符偏移范围。
    /// </summary>
    public (int Start, int End)? WordToChars(uint wordId, int sequenceId = 0)
    {
        var tokenRange = WordToTokens(wordId, sequenceId);
        if (tokenRange is null || tokenRange.Value.Count == 0)
            return null;

        var offsets = OffsetsSpan;
        int start = offsets[tokenRange.Value.Start].Start;
        int end = offsets[tokenRange.Value.Start + tokenRange.Value.Count - 1].End;
        return (start, end);
    }

    /// <summary>
    /// 将 token 索引映射到其所属的序列 ID。
    /// </summary>
    public int? TokenToSequence(int tokenIndex)
    {
        if (tokenIndex < 0 || tokenIndex >= _length)
            return null;

        foreach (var (seqId, range) in SequenceRanges)
        {
            if (tokenIndex >= range.Start.Value && tokenIndex < range.End.Value)
                return seqId;
        }
        return 0;
    }

    // ─── Pad（COW）───

    /// <summary>
    /// 将此编码填充到指定长度。
    /// </summary>
    public void Pad(int padLength, uint padId = 0, uint padTypeId = 0,
        string padToken = "[PAD]", PaddingDirection direction = PaddingDirection.Right)
    {
        if (ReferenceEquals(this, s_empty))
            throw new InvalidOperationException("不能修改 Encoding.Empty 共享实例，请先调用 Clone()。");

        if (padLength <= _length) return;

        // COW + 扩容
        EnsureWritable();
        EnsureCapacity(padLength);

        int padCount = padLength - _length;

        if (direction == PaddingDirection.Right)
        {
            _idsMem.Span.Slice(_length, padCount).Fill(padId);
            _typeIdsMem.Span.Slice(_length, padCount).Fill(padTypeId);
            _tokensMem.Span.Slice(_length, padCount).Fill(padToken);
            // _wordsMem 保持 default (null)
            // _offsetsMem 保持 default (0,0)
            _specialMaskMem.Span.Slice(_length, padCount).Fill(1u);
            _attentionMem.Span.Slice(_length, padCount).Fill(0u);
        }
        else // Left
        {
            // 右移现有数据
            var ids = _idsMem.Span;
            var typeIds = _typeIdsMem.Span;
            var tokens = _tokensMem.Span;
            var words = _wordsMem.Span;
            var offsets = _offsetsMem.Span;
            var special = _specialMaskMem.Span;
            var attention = _attentionMem.Span;

            // 从后往前复制，避免重叠
            for (int i = _length - 1; i >= 0; i--)
            {
                ids[i + padCount] = ids[i];
                typeIds[i + padCount] = typeIds[i];
                tokens[i + padCount] = tokens[i];
                words[i + padCount] = words[i];
                offsets[i + padCount] = offsets[i];
                special[i + padCount] = special[i];
                attention[i + padCount] = attention[i];
            }

            // 填充左侧
            _idsMem.Span.Slice(0, padCount).Fill(padId);
            _typeIdsMem.Span.Slice(0, padCount).Fill(padTypeId);
            _tokensMem.Span.Slice(0, padCount).Fill(padToken);
            _specialMaskMem.Span.Slice(0, padCount).Fill(1u);
            _attentionMem.Span.Slice(0, padCount).Fill(0u);

            // 更新 SequenceRanges
            var updatedRanges = new Dictionary<int, Range>(SequenceRanges.Count);
            foreach (var (seqId, range) in SequenceRanges)
                updatedRanges[seqId] = (range.Start.Value + padCount)..(range.End.Value + padCount);
            SequenceRanges.Clear();
            foreach (var kvp in updatedRanges)
                SequenceRanges[kvp.Key] = kvp.Value;
        }

        _length = padLength;
    }

    // ─── MergeWith（COW）───

    /// <summary>
    /// 将另一个编码合并到此编码（追加）。
    /// </summary>
    public void MergeWith(Encoding other, bool growingOffsets, int depth = 0)
    {
        if (depth == 0 && ReferenceEquals(this, s_empty))
            throw new InvalidOperationException("不能修改 Encoding.Empty 共享实例，请先调用 Clone()。");

        const int MaxMergeDepth = 32;
        if (depth > MaxMergeDepth)
            throw new InvalidOperationException(
                $"MergeWith 递归深度超过 {MaxMergeDepth}，可能存在极端截断配置。");

        bool hasOverflowing = Overflowing.Count > 0 || other.Overflowing.Count > 0;
        Encoding? selfSnapshot = hasOverflowing ? CloneWithoutOverflowing() : null;
        var selfOverflows = hasOverflowing ? Overflowing.ToList() : null;

        // COW：确保可写
        EnsureWritable();

        int originalLength = _length;
        int otherLength = other._length;
        int newLength = originalLength + otherLength;

        // 扩容并追加
        EnsureCapacity(newLength);

        // 用 Span 操作追加（零拷贝读取 other 的数据）
        other.IdsSpan.CopyTo(_idsMem.Span.Slice(originalLength));
        other.TypeIdsSpan.CopyTo(_typeIdsMem.Span.Slice(originalLength));
        other.TokensSpan.CopyTo(_tokensMem.Span.Slice(originalLength));
        other.WordsSpan.CopyTo(_wordsMem.Span.Slice(originalLength));

        if (growingOffsets && originalLength > 0)
        {
            int startingOffset = _offsetsMem.Span[_start + originalLength - 1].End;
            var otherOffsets = other.OffsetsSpan;
            for (int i = 0; i < otherLength; i++)
            {
                var (s, e) = otherOffsets[i];
                _offsetsMem.Span[originalLength + i] = (s + startingOffset, e + startingOffset);
            }
        }
        else
        {
            other.OffsetsSpan.CopyTo(_offsetsMem.Span.Slice(originalLength));
        }

        other.SpecialTokensMaskSpan.CopyTo(_specialMaskMem.Span.Slice(originalLength));
        other.AttentionMaskSpan.CopyTo(_attentionMem.Span.Slice(originalLength));

        _length = newLength;

        if (!hasOverflowing)
            return;

        // 慢路径：有溢出时处理交叉积
        var newOverflowing = new List<Encoding>();

        if (selfOverflows!.Count > 0 && other.Overflowing.Count > 0)
        {
            // 双方都有溢出：完整交叉积
            foreach (var selfO in selfOverflows)
            {
                // self_o × other
                var n1 = selfO.Clone();
                n1.MergeWith(other, growingOffsets, depth + 1);
                newOverflowing.Add(n1);

                // self_o × other_o
                foreach (var otherO in other.Overflowing)
                {
                    var n2 = selfO.Clone();
                    n2.MergeWith(otherO, growingOffsets, depth + 1);
                    newOverflowing.Add(n2);
                }
            }

            // selfSnapshot × other_o
            foreach (var otherO in other.Overflowing)
            {
                var n3 = selfSnapshot!.Clone();
                n3.MergeWith(otherO, growingOffsets, depth + 1);
                newOverflowing.Add(n3);
            }
        }
        else if (selfOverflows.Count > 0)
        {
            // 只有 self 有溢出：self_o × other
            foreach (var selfO in selfOverflows)
            {
                var n = selfO.Clone();
                n.MergeWith(other, growingOffsets, depth + 1);
                newOverflowing.Add(n);
            }
        }
        else if (other.Overflowing.Count > 0)
        {
            // 只有 other 有溢出：selfSnapshot × other_o
            foreach (var otherO in other.Overflowing)
            {
                var n = selfSnapshot!.Clone();
                n.MergeWith(otherO, growingOffsets, depth + 1);
                newOverflowing.Add(n);
            }
        }

        Overflowing.Clear();
        Overflowing.AddRange(newOverflowing);
    }

    // ─── Clone（COW 浅拷贝）───

    /// <summary>
    /// 创建此编码的 COW 浅拷贝：共享底层数组，标记为共享。
    /// Overflowing 仍需深拷贝（因为它们会被独立修改）。
    /// </summary>
    public Encoding Clone()
    {
        return new Encoding(
            _idsMem, _typeIdsMem, _tokensMem, _wordsMem,
            _offsetsMem, _specialMaskMem, _attentionMem,
            _start, _length, shared: true,
            Overflowing.Select(o => o.Clone()).ToList(),
            new Dictionary<int, Range>(SequenceRanges));
    }

    /// <summary>
    /// 创建不包含溢出编码的 COW 浅拷贝。
    /// </summary>
    internal Encoding CloneWithoutOverflowing()
    {
        return new Encoding(
            _idsMem, _typeIdsMem, _tokensMem, _wordsMem,
            _offsetsMem, _specialMaskMem, _attentionMem,
            _start, _length, shared: true,
            [],
            new Dictionary<int, Range>(SequenceRanges));
    }

    // ─── Merge ───

    /// <summary>
    /// 合并多个编码为一个。
    /// </summary>
    public static Encoding Merge(List<Encoding> encodings, bool growingOffsets)
    {
        if (encodings.Count == 0) return Empty;
        if (encodings.Count == 1) return encodings[0];

        // 对共享实例做深拷贝，防止修改缓存的 Encoding
        var first = encodings[0];
        if (first._isShared)
            first = first.DeepCopy();

        for (int i = 1; i < encodings.Count; i++)
            first.MergeWith(encodings[i], growingOffsets);
        return first;
    }

    /// <summary>
    /// 完整深拷贝：复制所有数据和溢出编码。不共享 Memory。
    /// </summary>
    private Encoding DeepCopy()
    {
        return new Encoding(
            IdsSpan.ToArray(),
            TypeIdsSpan.ToArray(),
            TokensSpan.ToArray(),
            WordsSpan.ToArray(),
            OffsetsSpan.ToArray(),
            SpecialTokensMaskSpan.ToArray(),
            AttentionMaskSpan.ToArray(),
            Overflowing.Select(o => o.Clone()).ToList(),
            new Dictionary<int, Range>(SequenceRanges));
    }

    // ─── Empty ───

    private static readonly Encoding s_empty = new([], [], [], [], [], [], []);
    /// <summary>
    /// 返回共享的空 Encoding 实例。空数组不可变，可安全共享。
    /// </summary>
    public static Encoding Empty => s_empty;
}
