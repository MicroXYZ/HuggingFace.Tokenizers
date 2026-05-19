namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 增量流式解码器，可在 token ID 逐个到达时产生有效的字符串片段。
/// 适用于 LLM 流式输出场景，某些 token（如 byte_fallback）需要多个 ID 才能产生有效的 UTF-8。
///
/// 与 Rust DecodeStream 行为完全对齐：追踪 prefix + prefix_index，
/// 使用 starts_with 验证 + drain 精确裁剪，确保 strip decoder 等场景的正确性。
/// </summary>
public sealed class DecodeStream
{
    /// <summary>累积 token ID 的最大数量，防止无效序列导致无限增长。</summary>
    private const int MaxBufferedIds = 4096;

    private readonly Tokenizer _tokenizer;
    private readonly bool _skipSpecialTokens;
    private readonly List<uint> _ids;
    /// <summary>
    /// 前一次成功输出时的完整解码字符串。
    /// 用于截取新增部分：newText = decoded[prefix.Length..]。
    /// </summary>
    private string _prefix;
    /// <summary>
    /// prefix 对应的 ids 起始索引（相对于 _ids 逻辑起始位置）。
    /// </summary>
    private int _prefixIndex;
    /// <summary>
    /// 逻辑起始偏移量，用于延迟裁剪避免 RemoveRange 的 O(n) 开销。
    /// 真正裁剪仅在 _ids 超过阈值时执行。
    /// </summary>
    private int _idStartOffset;

    /// <summary>
    /// 创建增量流式解码器。
    /// </summary>
    /// <param name="tokenizer">分词器实例。</param>
    /// <param name="skipSpecialTokens">是否跳过特殊 token。</param>
    public DecodeStream(Tokenizer tokenizer, bool skipSpecialTokens = true)
    {
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
        _skipSpecialTokens = skipSpecialTokens;
        _ids = new List<uint>();
        _prefix = "";
        _prefixIndex = 0;
    }

    /// <summary>
    /// 处理单个 token ID，如果可能则返回解码后的字符串片段。
    /// 如果需要更多 token 才能产生有效字符串（如不完整的 byte_fallback 序列），返回 null。
    /// </summary>
    /// <param name="id">token ID。</param>
    /// <returns>解码后的字符串片段，或 null。</returns>
    /// <exception cref="InvalidOperationException">解码结果不以 prefix 开头时抛出（内部状态损坏）。</exception>
    public string? Step(uint id)
    {
        // 与 Rust step_decode_stream 对齐：
        // 1. 如果 prefix 为空且有 ids，尝试解码得到初始 prefix
        if (_prefix.Length == 0 && LogicalCount > 0)
        {
            var newPrefix = SafeDecode(GetLogicalIds());
            if (newPrefix is not null && !newPrefix.EndsWith('\uFFFD'))
            {
                _prefix = newPrefix;
                _prefixIndex = LogicalCount;
            }
        }

        // 2. 追加新 id
        _ids.Add(id);

        // 3. 缓冲区保护：超过上限时强制刷新已累积的内容
        if (LogicalCount > MaxBufferedIds)
        {
            var flushed = SafeDecode(GetLogicalIds());
            _ids.Clear();
            _prefix = "";
            _prefixIndex = 0;
            _idStartOffset = 0;
            if (flushed is not null && flushed.Length > 0)
                return flushed;
            return null;
        }

        // 4. 解码全部逻辑 ids
        var decoded = SafeDecode(GetLogicalIds());
        if (decoded is null)
            return null;

        // 5. 检查是否有新内容可输出
        if (decoded.Length <= _prefix.Length || decoded.EndsWith('\uFFFD'))
            return null;

        // 6. 验证 decoded 以 prefix 开头（与 Rust starts_with 检查对齐）
        if (!decoded.StartsWith(_prefix, StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"DecodeStream 内部状态异常：解码结果不以 prefix 开头。" +
                $"Token ID: {id}, Expected prefix: '{_prefix}', Actual: '{decoded}'");
        }

        // 7. 截取新增部分
        var newText = decoded[_prefix.Length..];

        // 8. 逻辑裁剪：推进偏移量，延迟物理裁剪（避免 RemoveRange 的 O(n) 开销）
        var newPrefixIndex = LogicalCount - _prefixIndex;
        _idStartOffset += _prefixIndex;
        _prefixIndex = newPrefixIndex;

        // 仅在缓冲区膨胀时执行真正的物理裁剪（防内存泄漏）
        if (_ids.Count > MaxBufferedIds * 2 || _idStartOffset > _ids.Count / 2)
        {
            _ids.RemoveRange(0, _idStartOffset);
            _idStartOffset = 0;
        }

        // 重新解码得到新 prefix
        _prefix = SafeDecode(GetLogicalIds()) ?? "";

        return newText;
    }

    /// <summary>
    /// 逻辑 id 数量（排除已逻辑裁剪的部分）。
    /// </summary>
    private int LogicalCount => _ids.Count - _idStartOffset;

    /// <summary>
    /// 获取逻辑 id 列表的切片视图（避免物理复制）。
    /// </summary>
    private IReadOnlyList<uint> GetLogicalIds()
    {
        if (_idStartOffset == 0) return _ids;
        return new ListSlice(_ids, _idStartOffset);
    }

    /// <summary>
    /// 安全解码，捕获解码异常返回 null（与 Rust 的 Result 传播 + 替换检查对齐）。
    /// 仅捕获数据相关的异常，不吞没逻辑错误（InvalidOperationException）。
    /// </summary>
    private string? SafeDecode(IReadOnlyList<uint> ids)
    {
        try
        {
            return _tokenizer.Decode(ids, _skipSpecialTokens);
        }
        catch (Exception ex) when (ex is ArgumentException
            or System.Text.DecoderFallbackException)
        {
            // 流式解码容错：decoder 操作字符串可能抛异常
            // （如 malformed token），不应因单个坏 token 中断整个流。
            // InvalidOperationException 表示逻辑错误，不应吞没。
            return null;
        }
    }

    /// <summary>
    /// 返回迄今累积的 token ID（逻辑视图）。
    /// </summary>
    public IReadOnlyList<uint> Ids => GetLogicalIds();

    /// <summary>
    /// 将流重置为初始状态。
    /// </summary>
    public void Reset()
    {
        _ids.Clear();
        _prefix = "";
        _prefixIndex = 0;
        _idStartOffset = 0;
    }

    /// <summary>
    /// IReadOnlyList 的只读切片视图，避免物理复制。
    /// 使用 struct 避免堆分配（通过接口使用时会装箱，但仅在 _idStartOffset > 0 时创建）。
    /// </summary>
    private readonly struct ListSlice : IReadOnlyList<uint>
    {
        private readonly IReadOnlyList<uint> _source;
        private readonly int _offset;

        public ListSlice(IReadOnlyList<uint> source, int offset)
        {
            _source = source;
            _offset = offset;
        }

        public uint this[int index] => _source[index + _offset];
        public int Count => _source.Count - _offset;
        public IEnumerator<uint> GetEnumerator()
        {
            for (int i = 0; i < Count; i++)
                yield return this[i];
        }
        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();
    }
}
