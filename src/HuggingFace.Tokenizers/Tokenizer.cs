using System.Collections.Concurrent;
using HuggingFace.Tokenizers.Padding;

namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 主分词器，编排完整管道：
/// Normalizer → PreTokenizer → Model → PostProcessor → Decoder
/// </summary>
public sealed partial class Tokenizer
{
    private volatile INormalizer? _normalizer;
    private volatile IPreTokenizer? _preTokenizer;
    private volatile IModel _model;
    private volatile IPostProcessor? _postProcessor;
    private volatile IDecoder? _decoder;

    private TruncationParams? _truncation;
    private PaddingParams? _padding;

    private readonly AddedVocabulary _addedVocabulary = new();

    public Tokenizer(IModel model)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
    }

    // --- Pipeline component accessors ---

    public INormalizer? Normalizer
    {
        get => _normalizer;
        set => _normalizer = value;
    }

    public IPreTokenizer? PreTokenizer
    {
        get => _preTokenizer;
        set => _preTokenizer = value;
    }

    public IModel Model
    {
        get => _model;
        set => _model = value ?? throw new ArgumentNullException(nameof(value));
    }

    public IPostProcessor? PostProcessor
    {
        get => _postProcessor;
        set => _postProcessor = value;
    }

    public IDecoder? Decoder
    {
        get => _decoder;
        set => _decoder = value;
    }

    public TruncationParams? Truncation
    {
        get => _truncation;
        set => _truncation = value;
    }

    public PaddingParams? Padding
    {
        get => _padding;
        set => _padding = value;
    }

    /// <summary>访问已添加词表，用于高级 token 管理。</summary>
    public AddedVocabulary AddedVocabulary => _addedVocabulary;

    // --- Token lookup ---

    /// <summary>
    /// 查找 token 字符串对应的 ID。
    /// 优先搜索已添加词表，然后搜索模型词表。
    /// </summary>
    public uint? TokenToId(string token)
    {
        // Check added vocabulary first
        var addedId = _addedVocabulary.TokenToId(token, _model);
        if (addedId.HasValue) return addedId;

        // Fall back to model
        return _model.TokenToId(token);
    }

    /// <summary>
    /// 查找 ID 对应的 token 字符串。
    /// 优先搜索已添加词表，然后搜索模型词表。
    /// </summary>
    public string? IdToToken(uint id)
    {
        // Check added vocabulary first
        var addedToken = _addedVocabulary.IdToToken(id, _model);
        if (addedToken is not null) return addedToken;

        // Fall back to model
        return _model.IdToToken(id);
    }

    /// <summary>
    /// 获取或设置编码时是否包含特殊 token。
    /// 为 false 时，输入文本中的特殊 token 不会被匹配。
    /// </summary>
    public bool EncodeSpecialTokens
    {
        get => _addedVocabulary.EncodeSpecialTokens;
        set => _addedVocabulary.EncodeSpecialTokens = value;
    }

    // --- Vocabulary access ---

    /// <summary>
    /// 获取完整词表（模型词表 + 已添加 token）。
    /// 与 Rust get_vocab(with_added_tokens=true) 行为一致。
    /// </summary>
    public IReadOnlyDictionary<string, uint> GetVocabWithAddedTokens()
    {
        var vocab = new Dictionary<string, uint>(_model.GetVocab(), StringComparer.Ordinal);
        foreach (var kv in _addedVocabulary.GetAddedTokens())
            vocab[kv.Key] = kv.Value;
        return vocab;
    }

    /// <summary>
    /// 获取完整词表大小（模型词表 + 已添加 token）。
    /// 与 Rust get_vocab_size(with_added_tokens=true) 行为一致。
    /// </summary>
    public int GetVocabSizeWithAddedTokens()
    {
        return _model.GetVocabSize() + _addedVocabulary.GetAddedTokens().Count;
    }

    // --- Parallelism configuration ---

    /// <summary>
    /// 获取或设置批量编码的最大并行度。
    /// 设为 1 可禁用并行。默认值：处理器核心数。
    /// 也可通过 TOKENIZERS_PARALLELISM 环境变量配置。
    /// </summary>
    public static int MaxDegreeOfParallelism { get; set; } = GetDefaultParallelism();

    private static int GetDefaultParallelism()
    {
        // Check environment variable first
        var envVar = Environment.GetEnvironmentVariable("TOKENIZERS_PARALLELISM");
        if (int.TryParse(envVar, out var envValue) && envValue > 0)
            return envValue;
        return Environment.ProcessorCount;
    }

    // --- Added tokens management ---

    public void AddToken(AddedToken token)
    {
        if (token.IsSpecial)
            _addedVocabulary.AddSpecialTokens([token], _model, _normalizer);
        else
            _addedVocabulary.AddTokens([token], _model, _normalizer);
    }

    public void AddTokens(IReadOnlyList<AddedToken> tokens)
    {
        var specials = new List<AddedToken>();
        var regulars = new List<AddedToken>();
        for (int i = 0; i < tokens.Count; i++)
        {
            if (tokens[i].IsSpecial) specials.Add(tokens[i]);
            else regulars.Add(tokens[i]);
        }

        if (specials.Count > 0)
            _addedVocabulary.AddSpecialTokens(specials, _model, _normalizer);
        if (regulars.Count > 0)
            _addedVocabulary.AddTokens(regulars, _model, _normalizer);
    }

    // --- Encoding (tokenize) ---

    /// <summary>
    /// 使用字节偏移编码单个字符串（默认）。
    /// 产生与 Rust/Python/JS 分词器实现一致的偏移。
    /// string 隐式转换为 ReadOnlySpan&lt;char&gt;，调用方不受影响。
    /// </summary>
    public Encoding Encode(ReadOnlySpan<char> text, bool addSpecialTokens = true)
    {
        return EncodeWithOffsetType(text, default, addSpecialTokens, OffsetType.Byte);
    }

    /// <summary>
    /// 快速编码，仅返回 token ID，跳过偏移追踪。
    /// 对于只需要 ID 的场景（如嵌入、相似度评分、ML 模型输入），
    /// 比 <see cref="Encode"/> 显著更快。
    /// </summary>
    public uint[] EncodeFast(ReadOnlySpan<char> text, bool addSpecialTokens = true)
    {
        uint[] ids;

        // 截断 early-exit：当配置了截断时，使用 TokenizeWithLimit 限制分词数量，
        // 避免对全部文本完整编码后再截断（如 50K 字符编码后只保留 512 个 token）。
        if (_truncation is not null && _truncation.Strategy != TruncationStrategy.OnlySecond)
        {
            int maxLen = _truncation.MaxLength;
            if (addSpecialTokens && _postProcessor is not null)
            {
                int nAdded = _postProcessor.AddedTokens(false);
                if (nAdded > 0 && maxLen >= nAdded)
                    maxLen -= nAdded;
            }

            var pretokenized = TokenizeToPreTokenizedWithLimit(text, maxLen, _truncation.Direction);
            ids = pretokenized.ToIds();
        }
        else
        {
            // 无截断：使用 TokenRef 零分配分词
            var pretokenized = TokenizeToPreTokenizedRef(text);
            ids = pretokenized.ToIds();
        }

        // 3. Apply post-processor (adds special tokens like [CLS], [SEP])
        if (_postProcessor is not null && addSpecialTokens)
        {
            var encoding = CreateMinimalEncoding(ids, 0);
            return _postProcessor.Process(encoding, null, true).GetIds();
        }

        return ids;
    }

    /// <summary>
    /// 使用字符级偏移编码。
    /// 将字节偏移转换为 .NET char 偏移 — 适用于在 .NET 字符串操作中直接使用结果 Encoding。
    /// </summary>
    public Encoding EncodeCharOffsets(ReadOnlySpan<char> text, bool addSpecialTokens = true)
    {
        return EncodeWithOffsetType(text, default, addSpecialTokens, OffsetType.Char);
    }

    /// <summary>
    /// 使用字符级偏移编码配对字符串。
    /// </summary>
    public Encoding EncodeCharOffsetsPair(ReadOnlySpan<char> text, ReadOnlySpan<char> pairText, bool addSpecialTokens = true)
    {
        return EncodeWithOffsetType(text, pairText, addSpecialTokens, OffsetType.Char);
    }

    /// <summary>
    /// 使用字节偏移编码配对字符串（默认）。
    /// </summary>
    public Encoding EncodePair(ReadOnlySpan<char> text, ReadOnlySpan<char> pairText, bool addSpecialTokens = true)
    {
        return EncodeWithOffsetType(text, pairText, addSpecialTokens, OffsetType.Byte);
    }

    /// <summary>
    /// 核心编码方法，接受偏移类型参数。
    /// </summary>
    private Encoding EncodeWithOffsetType(ReadOnlySpan<char> text, ReadOnlySpan<char> pairText, bool addSpecialTokens, OffsetType offsetType)
    {
        bool hasPair = !pairText.IsEmpty;

        // WithLimit 路径（与 Rust do_tokenize 对齐）：
        // 单序列、无 Fixed padding 时，先用 WithLimit 停止分词后续 splits，
        // 再用 Truncate 做精确截断 + overflow 创建。
        // WithLimit 只避免分词多余的 splits，Truncate 负责精确截断。
        // stride>0 时 WithLimit + Truncate 仍然正确（滑动窗口由 Truncate 处理）。
        if (!hasPair && _truncation is not null
            && _truncation.Strategy != TruncationStrategy.OnlySecond
            && (_padding is null || _padding.Strategy != PaddingStrategy.Fixed))
        {
            int maxLen = _truncation.MaxLength;
            if (addSpecialTokens && _postProcessor is not null)
            {
                int nAdded = _postProcessor.AddedTokens(false);
                if (nAdded > 0 && maxLen >= nAdded)
                    maxLen -= nAdded;
            }

            // 步骤 1：WithLimit 停止分词后续 splits（与 Rust tokenize_with_limit 对齐）
            var pretokenized = TokenizeToPreTokenizedWithLimit(text, maxLen, _truncation.Direction);
            var encoding = pretokenized.ToEncoding(0, offsetType);

            // 步骤 2：Truncate 做精确截断 + overflow（与 Rust truncate_encodings 对齐）
            if (encoding.Length > maxLen)
            {
                var truncParams = new TruncationParams
                {
                    MaxLength = maxLen,
                    Strategy = _truncation.Strategy,
                    Stride = _truncation.Stride,
                    Direction = _truncation.Direction
                };
                (encoding, _) = Truncate(encoding, null, truncParams);
            }

            if (_postProcessor is not null && addSpecialTokens)
                encoding = _postProcessor.Process(encoding, null, true);

            return encoding;
        }

        var encoding2 = EncodeSingle(text, offsetType);
        Encoding? pairEncoding = hasPair ? EncodeSingle(pairText, offsetType) : null;

        // Apply truncation if configured.
        // When addSpecialTokens is true, subtract the number of tokens the
        // post-processor will add (e.g. [CLS], [SEP]) so the *final* encoding
        // length matches MaxLength. This matches Rust's post_process logic.
        if (_truncation is not null)
        {
            var truncParams = _truncation;
            if (addSpecialTokens && _postProcessor is not null)
            {
                int nAdded = _postProcessor.AddedTokens(hasPair);
                if (nAdded > 0 && _truncation.MaxLength > nAdded)
                {
                    truncParams = new TruncationParams
                    {
                        MaxLength = _truncation.MaxLength - nAdded,
                        Strategy = _truncation.Strategy,
                        Stride = _truncation.Stride,
                        Direction = _truncation.Direction
                    };
                }
            }
            (encoding2, pairEncoding) = Truncate(encoding2, pairEncoding, truncParams);
        }

        // Apply post-processor (adds special tokens)
        if (_postProcessor is not null)
        {
            encoding2 = _postProcessor.Process(encoding2, pairEncoding, addSpecialTokens);
        }
        else if (pairEncoding is not null)
        {
            encoding2 = Encoding.Merge([encoding2, pairEncoding], false);
        }

        // 应用填充（Fixed 策略在单条编码时填充，BatchLongest 在 EncodeBatch 中处理）
        if (_padding is not null && _padding.Strategy == PaddingStrategy.Fixed)
        {
            encoding2.Pad(_padding.MaxLength, _padding.PadId, _padding.PadTypeId, _padding.PadToken, _padding.Direction);
        }

        return encoding2;
    }

    // --- Batch encoding (ReadOnlyMemory<char> + Range 分区) ---

    /// <summary>
    /// 将字符串集合转换为连续 buffer + Range 数组，供批量编码 API 使用。
    /// </summary>
    public static (ReadOnlyMemory<char> Buffer, Range[] Ranges) CombineTexts(IReadOnlyList<string> texts)
    {
        int totalLen = 0;
        for (int i = 0; i < texts.Count; i++)
            totalLen += texts[i].Length;

        var buffer = new char[totalLen];
        var ranges = new Range[texts.Count];
        int pos = 0;
        for (int i = 0; i < texts.Count; i++)
        {
            texts[i].AsSpan().CopyTo(buffer.AsSpan(pos));
            ranges[i] = new Range(pos, pos + texts[i].Length);
            pos += texts[i].Length;
        }
        return (buffer, ranges);
    }

    /// <summary>
    /// 批量编码。当 <see cref="MaxDegreeOfParallelism"/> 大于 1 时使用并行处理。
    /// 所有文本拼接到一个连续 char[] buffer，用 Range 标记每条文本的位置。
    /// 单次分配、cache 友好、支持并行。
    /// </summary>
    /// <param name="combinedBuffer">所有文本拼接后的连续字符缓冲区。</param>
    /// <param name="textRanges">每条文本在 combinedBuffer 中的 Range。</param>
    /// <param name="addSpecialTokens">是否添加特殊 token。</param>
    public Encoding[] EncodeBatch(ReadOnlyMemory<char> combinedBuffer, ReadOnlySpan<Range> textRanges, bool addSpecialTokens = true)
    {
        var results = new Encoding[textRanges.Length];
        BatchExecuteSpan(textRanges, (i, range) => results[i] = Encode(combinedBuffer.Span[range], addSpecialTokens));

        if (_padding is not null)
            PaddingHelper.PadEncodings(results, _padding);

        return results;
    }

    /// <summary>
    /// 批量编码配对字符串（单 buffer + Range 分区）。
    /// 所有 first 和 second 文本拼接到同一个 char[] buffer，
    /// 用 firstRanges 和 secondRanges 分别标记每对文本的位置。
    /// </summary>
    public Encoding[] EncodeBatchPair(
        ReadOnlyMemory<char> combinedBuffer,
        ReadOnlySpan<Range> firstRanges,
        ReadOnlySpan<Range> secondRanges,
        bool addSpecialTokens = true)
    {
        if (firstRanges.Length != secondRanges.Length)
            throw new ArgumentException("firstRanges 和 secondRanges 长度必须一致。");

        var results = new Encoding[firstRanges.Length];
        BatchExecuteSpan(firstRanges, secondRanges, (i, first, second) =>
            results[i] = EncodePair(combinedBuffer.Span[first], combinedBuffer.Span[second], addSpecialTokens));

        if (_padding is not null)
            PaddingHelper.PadEncodings(results, _padding);

        return results;
    }

    /// <summary>
    /// 批量编码，使用字符级偏移。
    /// </summary>
    public Encoding[] EncodeBatchCharOffsets(ReadOnlyMemory<char> combinedBuffer, ReadOnlySpan<Range> textRanges, bool addSpecialTokens = true)
    {
        var results = new Encoding[textRanges.Length];
        BatchExecuteSpan(textRanges, (i, range) => results[i] = EncodeCharOffsets(combinedBuffer.Span[range], addSpecialTokens));

        if (_padding is not null)
            PaddingHelper.PadEncodings(results, _padding);

        return results;
    }

    /// <summary>
    /// 批量快速编码，仅返回 token ID。
    /// </summary>
    public uint[][] EncodeBatchFast(ReadOnlyMemory<char> combinedBuffer, ReadOnlySpan<Range> textRanges, bool addSpecialTokens = true)
    {
        var results = new uint[textRanges.Length][];
        BatchExecuteSpan(textRanges, (i, range) => results[i] = EncodeFast(combinedBuffer.Span[range], addSpecialTokens));
        return results;
    }

    // --- Batch encoding (IReadOnlyList<string> 便捷重载) ---

    /// <summary>
    /// 批量编码。内部调用 <see cref="CombineTexts"/> 拼接文本，再委托给 buffer+Range 版本。
    /// </summary>
    /// <param name="texts">待编码的字符串列表。</param>
    /// <param name="addSpecialTokens">是否添加特殊 token。</param>
    public Encoding[] EncodeBatch(IReadOnlyList<string> texts, bool addSpecialTokens = true)
    {
        var (buf, ranges) = CombineTexts(texts);
        return EncodeBatch(buf, ranges, addSpecialTokens);
    }

    /// <summary>
    /// 批量编码配对字符串。内部拼接两个列表到同一 buffer，再委托给 buffer+Range 版本。
    /// </summary>
    public Encoding[] EncodeBatchPair(
        IReadOnlyList<string> firstTexts,
        IReadOnlyList<string?> secondTexts,
        bool addSpecialTokens = true)
    {
        if (firstTexts.Count != secondTexts.Count)
            throw new ArgumentException("firstTexts 和 secondTexts 长度必须一致。");

        int totalLen = 0;
        for (int i = 0; i < firstTexts.Count; i++)
        {
            totalLen += firstTexts[i].Length;
            totalLen += secondTexts[i]?.Length ?? 0;
        }

        var buffer = new char[totalLen];
        var firstRanges = new Range[firstTexts.Count];
        var secondRanges = new Range[firstTexts.Count];
        int pos = 0;
        for (int i = 0; i < firstTexts.Count; i++)
        {
            firstTexts[i].AsSpan().CopyTo(buffer.AsSpan(pos));
            firstRanges[i] = new Range(pos, pos + firstTexts[i].Length);
            pos += firstTexts[i].Length;

            if (secondTexts[i] is not null)
            {
                secondTexts[i]!.AsSpan().CopyTo(buffer.AsSpan(pos));
                secondRanges[i] = new Range(pos, pos + secondTexts[i]!.Length);
                pos += secondTexts[i]!.Length;
            }
            else
            {
                secondRanges[i] = new Range(pos, pos);
            }
        }

        return EncodeBatchPair(buffer, firstRanges, secondRanges, addSpecialTokens);
    }

    /// <summary>
    /// 批量编码，使用字符级偏移。内部调用 <see cref="CombineTexts"/>。
    /// </summary>
    public Encoding[] EncodeBatchCharOffsets(IReadOnlyList<string> texts, bool addSpecialTokens = true)
    {
        var (buf, ranges) = CombineTexts(texts);
        return EncodeBatchCharOffsets(buf, ranges, addSpecialTokens);
    }

    /// <summary>
    /// 批量快速编码，仅返回 token ID。内部调用 <see cref="CombineTexts"/>。
    /// </summary>
    public uint[][] EncodeBatchFast(IReadOnlyList<string> texts, bool addSpecialTokens = true)
    {
        var (buf, ranges) = CombineTexts(texts);
        return EncodeBatchFast(buf, ranges, addSpecialTokens);
    }

    /// <summary>
    /// 批量解码 token ID 序列回文本。
    /// </summary>
    public string[] DecodeBatch(IReadOnlyList<uint>[] sentences, bool skipSpecialTokens = true)
    {
        var results = new string[sentences.Length];
        for (int i = 0; i < sentences.Length; i++)
            results[i] = Decode(sentences[i], skipSpecialTokens);
        return results;
    }

    /// <summary>
    /// 通用批量处理方法，支持 ReadOnlySpan。
    /// 顺序路径直接使用 span，并行路径拷贝到数组（Parallel.ForEach lambda 不能捕获 ref struct）。
    /// 使用 Parallel.ForEach + 索引分区器实现负载均衡（对长短不一的文本更友好）。
    /// </summary>
    private void BatchExecuteSpan<TInput>(ReadOnlySpan<TInput> inputs, Action<int, TInput> action)
    {
        int count = inputs.Length;
        if (count == 0) return;

        if (MaxDegreeOfParallelism <= 1 || count <= 1 || count <= MaxDegreeOfParallelism)
        {
            for (int i = 0; i < count; i++)
                action(i, inputs[i]);
            return;
        }

        // 并行路径：ReadOnlySpan 不能在 lambda 中捕获，拷贝到数组
        var inputsArray = inputs.ToArray();

        // 使用索引分区器：Parallel.ForEach 内置 work-stealing，
        // 对长短不一的文本（如 batch 中混合短/长文本）负载均衡更好。
        Parallel.ForEach(
            Partitioner.Create(0, count, Math.Max(1, count / (MaxDegreeOfParallelism * 4))),
            new ParallelOptions { MaxDegreeOfParallelism = MaxDegreeOfParallelism },
            range =>
            {
                for (int i = range.Item1; i < range.Item2; i++)
                    action(i, inputsArray[i]);
            });
    }

    /// <summary>
    /// 双输入并行批处理：两个 ReadOnlySpan 配对执行。
    /// 使用 Parallel.ForEach + 索引分区器实现负载均衡。
    /// </summary>
    private void BatchExecuteSpan<TInput>(ReadOnlySpan<TInput> inputs1, ReadOnlySpan<TInput> inputs2, Action<int, TInput, TInput> action)
    {
        int count = inputs1.Length;
        if (count == 0) return;

        if (MaxDegreeOfParallelism <= 1 || count <= 1 || count <= MaxDegreeOfParallelism)
        {
            for (int i = 0; i < count; i++)
                action(i, inputs1[i], inputs2[i]);
            return;
        }

        var arr1 = inputs1.ToArray();
        var arr2 = inputs2.ToArray();

        Parallel.ForEach(
            Partitioner.Create(0, count, Math.Max(1, count / (MaxDegreeOfParallelism * 4))),
            new ParallelOptions { MaxDegreeOfParallelism = MaxDegreeOfParallelism },
            range =>
            {
                for (int i = range.Item1; i < range.Item2; i++)
                    action(i, arr1[i], arr2[i]);
            });
    }

    // --- Pre-tokenized encoding ---

    /// <summary>
    /// 编码预分词输入（已拆分为片段）。
    /// 跳过标准化器和预分词器；每个片段由模型独立分词，结果合并为单个 <see cref="Encoding"/>。
    /// </summary>
    /// <param name="tokens">预拆分的文本片段。</param>
    /// <param name="addSpecialTokens">是否通过后处理器添加特殊 token。</param>
    public Encoding EncodePreTokenized(IReadOnlyList<string> tokens, bool addSpecialTokens = true)
    {
        return EncodePreTokenizedPair(tokens, null, addSpecialTokens);
    }

    /// <summary>
    /// 编码预分词配对输入。
    /// 跳过标准化器和预分词器；每个片段由模型独立分词，结果合并。
    /// </summary>
    /// <param name="tokens">第一个序列的预拆分文本片段。</param>
    /// <param name="pairTokens">第二个（配对）序列的预拆分文本片段，或 <c>null</c>。</param>
    /// <param name="addSpecialTokens">是否通过后处理器添加特殊 token。</param>
    public Encoding EncodePreTokenizedPair(
        IReadOnlyList<string> tokens,
        IReadOnlyList<string>? pairTokens,
        bool addSpecialTokens = true)
    {
        // 1. Tokenize each segment with the model (skip normalizer & pre-tokenizer)
        var allTokens = new List<Token>();
        foreach (var segment in tokens)
        {
            var segmentTokens = _model.Tokenize(segment);
            allTokens.AddRange(segmentTokens);
        }

        // 2. Handle pair
        Encoding? pairEncoding = null;
        if (pairTokens is not null)
        {
            var pairAllTokens = new List<Token>();
            foreach (var segment in pairTokens)
            {
                var segmentTokens = _model.Tokenize(segment);
                pairAllTokens.AddRange(segmentTokens);
            }
            pairEncoding = TokensToEncoding(pairAllTokens, 1);
        }

        // 3. Build encoding from tokens
        var encoding = TokensToEncoding(allTokens, 0);

        // 4. Apply truncation if configured
        if (_truncation is not null)
            (encoding, pairEncoding) = Truncate(encoding, pairEncoding, _truncation);

        // 5. Apply post-processor
        if (_postProcessor is not null)
            encoding = _postProcessor.Process(encoding, pairEncoding, addSpecialTokens);
        else if (pairEncoding is not null)
            encoding = Encoding.Merge([encoding, pairEncoding], false);

        return encoding;
    }

    /// <summary>
    /// 辅助方法：将 <see cref="Token"/> 列表转换为 <see cref="Encoding"/>。
    /// </summary>
    private Encoding TokensToEncoding(List<Token> tokens, int sequenceId)
    {
        var ids = new uint[tokens.Count];
        var typeIds = new uint[tokens.Count];
        var tokenStrs = new string[tokens.Count];
        var words = new uint?[tokens.Count];
        var offsets = new (int Start, int End)[tokens.Count];
        var specialMask = new uint[tokens.Count];
        var attention = new uint[tokens.Count];

        for (int i = 0; i < tokens.Count; i++)
        {
            ids[i] = tokens[i].Id;
            tokenStrs[i] = tokens[i].Value;
            offsets[i] = (tokens[i].Start, tokens[i].End);
            attention[i] = 1;
        }

        var encoding = new Encoding(ids, typeIds, tokenStrs, words, offsets, specialMask, attention);
        encoding.SetSequenceId(sequenceId);
        return encoding;
    }

    // --- Decoding ---

    /// <summary>
    /// 将 token ID 解码回文本。
    /// </summary>
    public string Decode(IReadOnlyList<uint> ids, bool skipSpecialTokens = true)
    {
        var model = _model;
        var decoder = _decoder;

        var tokens = new List<string>(ids.Count);
        foreach (var id in ids)
        {
            var token = _addedVocabulary.IdToToken(id, model);
            if (token is null) continue;
            if (skipSpecialTokens && _addedVocabulary.IsSpecialToken(token)) continue;
            tokens.Add(token);
        }

        return decoder is not null
            ? decoder.Decode(tokens)
            : string.Join(" ", tokens);
    }

    /// <summary>
    /// 批量解码 token ID 序列回文本。
    // --- Internal pipeline ---

    private Encoding EncodeSingle(ReadOnlySpan<char> text, OffsetType offsetType = OffsetType.Byte)
    {
        var pretokenized = TokenizeToPreTokenized(text);
        return pretokenized.ToEncoding(0, offsetType);
    }

    /// <summary>
    /// 类似 <see cref="EncodeSingle"/> 但仅返回 token ID，
    /// 通过 <see cref="PreTokenizedString.ToIds"/> 跳过偏移、token 字符串、
    /// typeId、wordId、specialMask 和 attentionMask 的构建。
    /// </summary>
    private uint[] EncodeSingleIds(ReadOnlySpan<char> text)
    {
        var pretokenized = TokenizeToPreTokenized(text);
        return pretokenized.ToIds();
    }

    /// <summary>
    /// 共享的分词管道（ExtractAndNormalize → PreTokenize → Model Tokenize）。
    /// EncodeSingle 和 EncodeSingleIds 共用此方法，消除重复代码。
    /// </summary>
    private PreTokenizedString TokenizeToPreTokenized(ReadOnlySpan<char> text)
    {
        // 快照到局部变量，保证单次调用内管道组件一致
        var normalizer = _normalizer;
        var preTokenizer = _preTokenizer;
        var model = _model;

        // 1. Extract added tokens and normalize the rest
        var pretokenized = _addedVocabulary.ExtractAndNormalize(normalizer, text);

        // 2. Pre-tokenize (splits non-added-token parts further)
        preTokenizer?.PreTokenize(pretokenized);

        // 3. Tokenize with model (only non-tokenized splits)
        // 特化路径：对常见模型类型直接调用，避免接口虚方法分派
        if (model is Models.BPE.BpeModel bpe)
            pretokenized.Tokenize(ns => bpe.Tokenize(ns.GetSpan()));
        else if (model is Models.Unigram.UnigramModel unigram)
            pretokenized.Tokenize(ns => unigram.Tokenize(ns.GetSpan()));
        else if (model is Models.WordPiece.WordPieceModel wp)
            pretokenized.Tokenize(ns => wp.Tokenize(ns.GetSpan()));
        else
            pretokenized.Tokenize(ns => model.Tokenize(ns.GetSpan()));

        return pretokenized;
    }

    /// <summary>
    /// 带 token 限制的分词管道（截断 early-exit）。
    /// 当配置了截断时，使用 TokenizeWithLimit 在分词阶段就限制 token 数量，
    /// 避免对全部文本完整编码后再截断。
    /// </summary>
    private PreTokenizedString TokenizeToPreTokenizedWithLimit(ReadOnlySpan<char> text, int maxTokens, TruncationDirection direction)
    {
        var normalizer = _normalizer;
        var preTokenizer = _preTokenizer;
        var model = _model;

        var pretokenized = _addedVocabulary.ExtractAndNormalize(normalizer, text);
        preTokenizer?.PreTokenize(pretokenized);

        // 使用 TokenizeWithLimit 限制分词数量
        if (model is Models.BPE.BpeModel bpe)
            pretokenized.TokenizeWithLimit(ns => bpe.Tokenize(ns.GetSpan()), maxTokens, direction);
        else if (model is Models.Unigram.UnigramModel unigram)
            pretokenized.TokenizeWithLimit(ns => unigram.Tokenize(ns.GetSpan()), maxTokens, direction);
        else if (model is Models.WordPiece.WordPieceModel wp)
            pretokenized.TokenizeWithLimit(ns => wp.Tokenize(ns.GetSpan()), maxTokens, direction);
        else
            pretokenized.TokenizeWithLimit(ns => model.Tokenize(ns.GetSpan()), maxTokens, direction);

        return pretokenized;
    }

    /// <summary>
    /// 使用轻量级 TokenRef 的分词管道（无 string 分配）。
    /// 用于 EncodeFast 的非截断路径。
    /// </summary>
    private PreTokenizedString TokenizeToPreTokenizedRef(ReadOnlySpan<char> text)
    {
        var normalizer = _normalizer;
        var preTokenizer = _preTokenizer;
        var model = _model;

        var pretokenized = _addedVocabulary.ExtractAndNormalize(normalizer, text);
        preTokenizer?.PreTokenize(pretokenized);

        if (model is Models.BPE.BpeModel bpe)
            pretokenized.TokenizeRef(ns => bpe.TokenizeRef(ns.GetSpan()));
        else if (model is Models.Unigram.UnigramModel unigram)
            pretokenized.TokenizeRef(ns => ((IModel)unigram).TokenizeRef(ns.GetSpan()));
        else if (model is Models.WordPiece.WordPieceModel wp)
            pretokenized.TokenizeRef(ns => wp.TokenizeRef(ns.GetSpan()));
        else
            pretokenized.TokenizeRef(ns => model.TokenizeRef(ns.GetSpan()));

        return pretokenized;
    }
    /// 由 <see cref="EncodeFast"/> 内部使用，当只需要 ID 时为后处理器提供输入。
    /// 其他字段使用默认/空值。
    /// </summary>
    private static Encoding CreateMinimalEncoding(uint[] ids, uint typeId)
    {
        int len = ids.Length;
        var typeIds = new uint[len];
        var tokens = new string[len];
        var words = new uint?[len];
        var offsets = new (int Start, int End)[len];
        var specialMask = new uint[len];
        var attention = new uint[len];

        if (typeId != 0)
            Array.Fill(typeIds, typeId);
        Array.Fill(tokens, string.Empty);
        Array.Fill(attention, 1u);

        return new Encoding(ids, typeIds, tokens, words, offsets, specialMask, attention);
    }

    // ──────────────────────────────────────────────
    //  Stream decoding
    // ──────────────────────────────────────────────

    /// <summary>
    /// 创建 <see cref="DecodeStream"/> 用于增量逐 token 解码。
    /// 适用于 LLM 流式输出场景，token 逐个到达。
    /// </summary>
    public DecodeStream CreateDecodeStream(bool skipSpecialTokens = true)
        => new(this, skipSpecialTokens);

    // ──────────────────────────────────────────────
    //  序列化 / 反序列化
    // ──────────────────────────────────────────────

    /// <summary>
    /// 将分词器序列化为 JSON 字符串（tokenizer.json 格式）。
    /// </summary>
    /// <param name="pretty">是否缩进格式化。</param>
    public string ToJson(bool pretty = false)
        => Serialization.TokenizerSerializer.Serialize(this, pretty);

    /// <summary>
    /// 将分词器保存到文件（tokenizer.json 格式）。
    /// </summary>
    /// <param name="path">保存路径。</param>
    /// <param name="pretty">是否缩进格式化。</param>
    public void Save(string path, bool pretty = false)
    {
        var json = ToJson(pretty);
        File.WriteAllText(path, json);
    }

    /// <summary>
    /// 从 tokenizer.json 文件加载分词器。
    /// </summary>
    /// <param name="path">tokenizer.json 文件路径。</param>
    /// <returns>加载的分词器实例。</returns>
    public static Tokenizer FromFile(string path)
        => Serialization.TokenizerLoader.FromFile(path);

    /// <summary>
    /// 从 JSON 字符串加载分词器。
    /// </summary>
    /// <param name="json">tokenizer.json 格式的 JSON 字符串。</param>
    /// <returns>加载的分词器实例。</returns>
    public static Tokenizer FromJson(string json)
        => Serialization.TokenizerLoader.FromJson(json);

    /// <summary>
    /// 从 JSON 字节数组加载分词器。
    /// </summary>
    /// <param name="data">tokenizer.json 格式的 UTF-8 字节数组。</param>
    /// <returns>加载的分词器实例。</returns>
    public static Tokenizer FromBytes(byte[] data)
        => Serialization.TokenizerLoader.FromBytes(data);
}
