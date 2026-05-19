using System.Buffers;
using System.Collections.Frozen;
using System.Text;
using System.Text.Json;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;
using SysEncoding = System.Text.Encoding;

namespace HuggingFace.Tokenizers.Models.BPE;

/// <summary>
/// 字节对编码（BPE）分词模型。
/// 实现 GPT-2、RoBERTa 等模型使用的 BPE 算法。
/// 使用基于 ID 的合并映射和链表 Word，实现 O(n log n) 分词。
/// </summary>
[TokenizerComponent("BPE")]
public sealed class BpeModel : DictionaryVocabModel, IDisposable
{
    private readonly List<(string First, string Second)> _merges;
    private FrozenDictionary<(uint, uint), (uint Rank, uint NewId)> _mergeRanks;
    private Dictionary<(uint, uint), (uint Rank, uint NewId)> _mergeRanksMutable;
    private MergeBitmapLookup _mergeBitmap;
    private Utf8Vocab _utf8Vocab;
    private readonly string? _continuingSubwordPrefix;
    private readonly byte[]? _continuingSubwordPrefixUtf8;
    private readonly string? _endOfWordSuffix;
    private readonly byte[]? _endOfWordSuffixUtf8;
    private readonly bool _fuseUnk;
    private readonly bool _byteFallback;
    private readonly bool _ignoreMerges;
    private readonly int _cacheCapacity;
    private readonly float _dropout;

    // ByteFallback 直接索引数组：byte → token_id，O(1) 无哈希
    // uint.MaxValue 表示该字节未注册为 byte fallback token
    private readonly uint[] _byteFallbackIds;

    // 进程级单调计数器，用于分配全局唯一的 generation id，
    // 使 per-thread 缓存永远不会因不同 BPE 实例而冲突。
    private static long _nextCacheId;

    // 当前 BPE 实例的 generation id，ClearCache() 时递增，
    // 使所有线程的旧缓存条目一次性失效（O(1）。
    private long _cacheGeneration;

    // per-thread 缓存，与 Rust BPE_LOCAL_CACHE 行为一致。
    private readonly ThreadLocal<Dictionary<long, Dictionary<string, List<Token>>>>? _cache;

    /// <summary>
    /// 获取有序的合并对列表。
    /// </summary>
    public IReadOnlyList<(string First, string Second)> Merges => _merges;

    /// <summary>
    /// 获取基于 ID 的合并映射: (token_id_a, token_id_b) → (rank, new_token_id).
    /// </summary>
    public IReadOnlyDictionary<(uint, uint), (uint Rank, uint NewId)> MergeRanks => _mergeRanks;

    private BpeModel(
        Dictionary<string, uint> vocab,
        List<(string First, string Second)> merges,
        string? unkToken,
        string? continuingSubwordPrefix,
        string? endOfWordSuffix,
        bool fuseUnk,
        bool byteFallback,
        bool ignoreMerges,
        int cacheCapacity,
        float dropout) : base(vocab, unkToken)
    {
        _merges = merges;

        // 构建基于 ID 的合并映射，与 Rust MergeMap 对齐
        int prefixLen = continuingSubwordPrefix?.Length ?? 0;
        _mergeRanksMutable = new Dictionary<(uint, uint), (uint Rank, uint NewId)>(merges.Count);
        for (int i = 0; i < merges.Count; i++)
        {
            var (a, b) = merges[i];
            if (!_vocab.TryGetValue(a, out var aId)) continue;
            if (!_vocab.TryGetValue(b, out var bId)) continue;

            string merged;
            if (prefixLen > 0 && b.Length > prefixLen)
                merged = a + b[prefixLen..];
            else if (prefixLen > 0 && b.Length <= prefixLen)
                merged = a;
            else
                merged = a + b;

            if (!_vocab.TryGetValue(merged, out var newId)) continue;
            _mergeRanksMutable[(aId, bId)] = ((uint)i, newId);
        }

        // 冻结为 FrozenDictionary（查询用）
        _mergeRanks = _mergeRanksMutable.ToFrozenDictionary();

        // 构建位图查找表（替代 FrozenDictionary 哈希查找）
        _mergeBitmap = new MergeBitmapLookup(_mergeRanks, vocab.Count);

        _utf8Vocab = new Utf8Vocab(vocab);
        _continuingSubwordPrefix = continuingSubwordPrefix;
        _continuingSubwordPrefixUtf8 = continuingSubwordPrefix is not null
            ? SysEncoding.UTF8.GetBytes(continuingSubwordPrefix) : null;
        _endOfWordSuffix = endOfWordSuffix;
        _endOfWordSuffixUtf8 = endOfWordSuffix is not null
            ? SysEncoding.UTF8.GetBytes(endOfWordSuffix) : null;
        _fuseUnk = fuseUnk;
        _byteFallback = byteFallback;
        _ignoreMerges = ignoreMerges;
        _cacheCapacity = cacheCapacity;
        _dropout = dropout;

        // 预计算 byte → token_id 直接索引，避免每次 byte fallback 走哈希
        _byteFallbackIds = new uint[256];
        if (_byteFallback)
        {
            Array.Fill(_byteFallbackIds, uint.MaxValue);
            for (int b = 0; b < 256; b++)
            {
                var hexToken = RuneHelpers.FormatByteFallbackTokenUtf8((byte)b);
                if (_utf8Vocab.TryGetId(hexToken, out var id))
                    _byteFallbackIds[b] = id;
            }
        }

        if (_cacheCapacity > 0)
        {
            _cacheGeneration = Interlocked.Increment(ref _nextCacheId);
            _cache = new ThreadLocal<Dictionary<long, Dictionary<string, List<Token>>>>(
                () => new Dictionary<long, Dictionary<string, List<Token>>>());
        }
    }

    /// <inheritdoc />
    public override List<Token> Tokenize(ReadOnlySpan<char> sequence)
    {
        if (sequence.IsEmpty)
            return [];

        if (_ignoreMerges)
        {
            var vocabLookup = _vocab.GetAlternateLookup<ReadOnlySpan<char>>();
            if (vocabLookup.TryGetValue(sequence, out var wholeId))
            {
                var seqStr = sequence.ToString();
                return [new Token(wholeId, seqStr, 0, sequence.Length)];
            }
        }

        bool useCache = _cache is not null && (_dropout <= 0f);

        if (useCache)
        {
            var gen = Volatile.Read(ref _cacheGeneration);
            var threadCache = _cache!.Value!;

            if (!threadCache.TryGetValue(gen, out var local))
            {
                // 清理旧 generation 的缓存条目，防止内存持续增长
                if (threadCache.Count > 1)
                {
                    var staleKeys = new List<long>();
                    foreach (var key in threadCache.Keys)
                    {
                        if (key < gen) staleKeys.Add(key);
                    }
                    foreach (var key in staleKeys)
                        threadCache.Remove(key);
                }

                local = new Dictionary<string, List<Token>>(_cacheCapacity, StringComparer.Ordinal);
                threadCache[gen] = local;
            }

            // 零分配查找：通过 AlternateLookup 直接用 span 查缓存
            var cacheLookup = local.GetAlternateLookup<ReadOnlySpan<char>>();
            if (cacheLookup.TryGetValue(sequence, out var cached))
                return cached;

            var tokens = _ignoreMerges
                ? TokenizeWithoutMerges(sequence)
                : TokenizeWithBpe(sequence);

            const int MaxCacheLength = 128;
            if (sequence.Length < MaxCacheLength && local.Count < _cacheCapacity)
            {
                var seqStr = sequence.ToString();
                local[seqStr] = tokens;
            }

            return tokens;
        }

        return _ignoreMerges
            ? TokenizeWithoutMerges(sequence)
            : TokenizeWithBpe(sequence);
    }

    private List<Token> TokenizeWithBpe(ReadOnlySpan<char> sequence)
    {
        var word = MergeWord(sequence);
        return WordToTokens(word);
    }

    private List<Token> TokenizeWithoutMerges(ReadOnlySpan<char> sequence)
    {
        var word = SplitToWord(sequence);
        return WordToTokens(word);
    }

    /// <summary>
    /// 将输入字符串拆分为带有 uint token ID 的 Word，然后应用 BPE 合并。
    /// 使用 Utf8Vocab 进行 UTF-8 字节级查找，避免中间 string 分配。
    /// </summary>
    private Word MergeWord(ReadOnlySpan<char> sequence)
    {
        var word = new Word(sequence.Length);
        (uint Id, int ByteLen)? unk = null;

        byte[]? pooledBuf = null;
        int bufLen = sequence.Length * 4 + 16;
        Span<byte> byteBuf = bufLen <= 256
            ? stackalloc byte[bufLen]
            : (pooledBuf = ArrayPool<byte>.Shared.Rent(bufLen));
        try
        {
            bool isFirst = true;
            int idx = 0;

            foreach (var rune in sequence.EnumerateRunes())
            {
                int byteLen = rune.Utf8SequenceLength;
                bool hasNext = idx + rune.Utf16SequenceLength < sequence.Length;

                // 编码 rune 到 byteBuf 末尾（留空间给 prefix）
                int prefixLen = (!isFirst && _continuingSubwordPrefixUtf8 is { Length: > 0 } p) ? p.Length : 0;
                int runeWritten = rune.EncodeToUtf8(byteBuf.Slice(prefixLen));

                // 如果有 prefix，把 prefix 写到 byteBuf 开头
                if (prefixLen > 0)
                    _continuingSubwordPrefixUtf8!.AsSpan().CopyTo(byteBuf);

                int keyLen = prefixLen + runeWritten;

                if (hasNext)
                {
                    if (_utf8Vocab.TryGetId(byteBuf.Slice(0, keyLen), out var id))
                    {
                        if (unk.HasValue) { word.Add(unk.Value.Id, unk.Value.ByteLen); unk = null; }
                        word.Add(id, byteLen);
                    }
                    else if (_byteFallback)
                    {
                        if (TryAddByteFallbackUtf8(rune, byteBuf, word, ref unk)) { }
                        else HandleUnkToken(byteLen, ref unk, word);
                    }
                    else
                    {
                        HandleUnkToken(byteLen, ref unk, word);
                    }
                    isFirst = false;
                    idx += rune.Utf16SequenceLength;
                    continue;
                }

                // 最后一个 rune：可能需要 endOfWordSuffix
                if (_endOfWordSuffixUtf8 is { Length: > 0 } suffixBytes)
                {
                    suffixBytes.AsSpan().CopyTo(byteBuf.Slice(keyLen));
                    keyLen += suffixBytes.Length;
                }

                if (_utf8Vocab.TryGetId(byteBuf.Slice(0, keyLen), out var lastId))
                {
                    if (unk.HasValue) { word.Add(unk.Value.Id, unk.Value.ByteLen); unk = null; }
                    word.Add(lastId, byteLen);
                }
                else if (_byteFallback)
                {
                    if (TryAddByteFallbackUtf8(rune, byteBuf, word, ref unk)) { }
                    else HandleUnkToken(byteLen, ref unk, word);
                }
                else
                {
                    HandleUnkToken(byteLen, ref unk, word);
                }

                idx += rune.Utf16SequenceLength;
            }

            if (unk.HasValue) word.Add(unk.Value.Id, unk.Value.ByteLen);

            word.MergeAll(_mergeBitmap, _dropout > 0f ? _dropout : null);
            return word;
        }
        finally
        {
            if (pooledBuf is not null) ArrayPool<byte>.Shared.Return(pooledBuf);
        }
    }

    /// <summary>
    /// UTF-8 字节级 byte fallback，避免 string 分配。
    /// 使用直接索引数组 _byteFallbackIds[byte]，O(1) 无哈希。
    /// </summary>
    private bool TryAddByteFallbackUtf8(Rune rune, Span<byte> byteBuf, Word word, ref (uint Id, int ByteLen)? unk)
    {
        int written = rune.EncodeToUtf8(byteBuf);
        // 快速检查：所有字节是否都有对应的 byte fallback token
        for (int b = 0; b < written; b++)
        {
            if (_byteFallbackIds[byteBuf[b]] == uint.MaxValue)
                return false;
        }

        if (unk.HasValue) { word.Add(unk.Value.Id, unk.Value.ByteLen); unk = null; }
        for (int b = 0; b < written; b++)
            word.Add(_byteFallbackIds[byteBuf[b]], 1);
        return true;
    }

    private void HandleUnkToken(int byteLen, ref (uint Id, int ByteLen)? unk, Word word)
    {
        if (_unkToken is not null && _vocab.TryGetValue(_unkToken, out var unkId))
        {
            if (unk.HasValue && _fuseUnk)
                unk = (unk.Value.Id, unk.Value.ByteLen + byteLen);
            else
            {
                if (unk.HasValue) word.Add(unk.Value.Id, unk.Value.ByteLen);
                unk = (unkId, byteLen);
            }
        }
    }

    /// <summary>
    /// 将输入字符串拆分为单独的字符符号，不执行合并。
    /// 使用 Utf8Vocab 进行 UTF-8 字节级查找，避免中间 string 分配。
    /// </summary>
    private Word SplitToWord(ReadOnlySpan<char> sequence)
    {
        var word = new Word(sequence.Length);

        if (_byteFallback)
        {
            int maxUtf8 = SysEncoding.UTF8.GetMaxByteCount(sequence.Length);
            byte[]? pooledBytes = null;
            Span<byte> bytes = maxUtf8 <= 256
                ? stackalloc byte[maxUtf8]
                : (pooledBytes = ArrayPool<byte>.Shared.Rent(maxUtf8));
            try
            {
                int byteCount = SysEncoding.UTF8.GetBytes(sequence, bytes);
                for (int b = 0; b < byteCount; b++)
                {
                    uint id = _byteFallbackIds[bytes[b]];
                    if (id != uint.MaxValue)
                        word.Add(id, 1);
                }
            }
            finally
            {
                if (pooledBytes is not null) ArrayPool<byte>.Shared.Return(pooledBytes);
            }
            return word;
        }

        byte[]? pooledBuf = null;
        int maxLen = sequence.Length * 4 + 16;
        Span<byte> byteBuf = maxLen <= 256
            ? stackalloc byte[maxLen]
            : (pooledBuf = ArrayPool<byte>.Shared.Rent(maxLen));
        try
        {
            bool isFirst = true;
            Rune lastRune = default;

            foreach (var rune in sequence.EnumerateRunes())
            {
                int byteLen = rune.Utf8SequenceLength;
                int prefixLen = (!isFirst && _continuingSubwordPrefixUtf8 is { Length: > 0 } p) ? p.Length : 0;
                int runeWritten = rune.EncodeToUtf8(byteBuf.Slice(prefixLen));

                if (prefixLen > 0)
                    _continuingSubwordPrefixUtf8!.AsSpan().CopyTo(byteBuf);

                int keyLen = prefixLen + runeWritten;

                if (_utf8Vocab.TryGetId(byteBuf.Slice(0, keyLen), out var id))
                    word.Add(id, byteLen);
                else if (_unkToken is not null && _utf8Vocab.TryGetId(_unkToken.AsSpan(), out var unkId))
                    word.Add(unkId, byteLen);

                lastRune = rune;
                isFirst = false;
            }

            if (!string.IsNullOrEmpty(_endOfWordSuffix) && word.Length > 0)
            {
                // 直接用 lastRune，无需第二次遍历
                int runeWritten2 = lastRune.EncodeToUtf8(byteBuf);
                if (_endOfWordSuffixUtf8 is { Length: > 0 } suffixBytes)
                {
                    suffixBytes.AsSpan().CopyTo(byteBuf.Slice(runeWritten2));
                    int suffixKeyLen = runeWritten2 + suffixBytes.Length;

                    if (_utf8Vocab.TryGetId(byteBuf.Slice(0, suffixKeyLen), out var suffixId))
                    {
                        var newWord = new Word(sequence.Length);

                        for (int i = 0; i < word.Length; i++)
                        {
                            if (i == word.Length - 1)
                                newWord.Add(suffixId, suffixKeyLen);
                            else
                                newWord.Add(word.GetSymbolId(i), word.GetSymbolLen(i));
                        }

                        return newWord;
                    }
                }
            }

            return word;
        }
        finally
        {
            if (pooledBuf is not null) ArrayPool<byte>.Shared.Return(pooledBuf);
        }
    }

    private List<Token> WordToTokens(Word word)
    {
        var tokens = new List<Token>(word.Length);
        int pos = 0;
        for (int i = 0; i < word.Length; i++)
        {
            int newPos = pos + word.GetSymbolLen(i);
            uint id = word.GetSymbolId(i);
            // _vocabR.TryGetValue 返回已有 string 引用，无分配
            if (_vocabR.TryGetValue(id, out var value))
                tokens.Add(new Token(id, value, pos, newPos));
            else
                // string.Create 一次性分配，避免插值 boxing + 临时 string
                tokens.Add(new Token(id, FormatUnknownToken(id), pos, newPos));
            pos = newPos;
        }
        return tokens;
    }

    /// <summary>
    /// 将 Word 转换为轻量级 TokenRef 列表，不分配 string。
    /// 用于内部编码路径（EncodeFast、批量编码等）。
    /// </summary>
    private List<TokenRef> WordToTokenRefs(Word word)
    {
        var refs = new List<TokenRef>(word.Length);
        int pos = 0;
        for (int i = 0; i < word.Length; i++)
        {
            int newPos = pos + word.GetSymbolLen(i);
            refs.Add(new TokenRef(word.GetSymbolId(i), pos, newPos));
            pos = newPos;
        }
        return refs;
    }

    /// <inheritdoc />
    public List<TokenRef> TokenizeRef(ReadOnlySpan<char> sequence)
    {
        if (sequence.IsEmpty)
            return [];

        if (_ignoreMerges)
        {
            // ignoreMerges 模式：跳过合并，直接用 SplitToWord
            var word = SplitToWord(sequence);
            return WordToTokenRefs(word);
        }

        var mergedWord = MergeWord(sequence);
        return WordToTokenRefs(mergedWord);
    }

    /// <summary>
    /// 格式化未知 token ID 为 "&lt;id&gt;" 字符串。
    /// 使用 string.Create 一次性分配，避免插值 boxing。
    /// </summary>
    private static string FormatUnknownToken(uint id)
    {
        // uint 最大 10 位 + 尖括号 = 最多 12 字符
        return string.Create(12, id, (buf, v) =>
        {
            buf[0] = '<';
            if (v.TryFormat(buf[1..], out int written))
                buf[1 + written] = '>';
        });
    }

    /// <summary>
    /// 使用新的词汇表和合并规则重建模型（供训练器调用）。
    /// 采用 copy-on-write 策略：先构建新的映射和词表，再原子替换引用。
    /// </summary>
    /// <summary>
    /// 重建词表和合并映射（训练 finalize 阶段调用）。
    /// 注意：此方法非线程安全，仅允许在训练期间单线程调用。
    /// 调用时必须确保没有并发的 Encode/Decode 操作。
    /// </summary>
    internal void Rebuild(Dictionary<string, uint> newVocab, List<(string First, string Second)> newMerges)
    {
        // 先构建新的合并映射（不修改现有实例）
        var newMergeRanks = new Dictionary<(uint, uint), (uint Rank, uint NewId)>(newMerges.Count);
        int prefixLen = _continuingSubwordPrefix?.Length ?? 0;
        for (int i = 0; i < newMerges.Count; i++)
        {
            var (a, b) = newMerges[i];
            if (!newVocab.TryGetValue(a, out var aId)) continue;
            if (!newVocab.TryGetValue(b, out var bId)) continue;

            string merged;
            if (prefixLen > 0 && b.Length > prefixLen) merged = a + b[prefixLen..];
            else if (prefixLen > 0 && b.Length <= prefixLen) merged = a;
            else merged = a + b;

            if (!newVocab.TryGetValue(merged, out var newId)) continue;
            newMergeRanks[(aId, bId)] = ((uint)i, newId);
        }

        // 构建新的 Utf8Vocab
        var newUtf8Vocab = new Utf8Vocab(newVocab);

        // 原子替换所有数据结构
        RebuildVocab(newVocab);

        _merges.Clear();
        _merges.AddRange(newMerges);

        _mergeRanksMutable = newMergeRanks;
        Volatile.Write(ref _mergeRanks, newMergeRanks.ToFrozenDictionary());
        _mergeBitmap = new MergeBitmapLookup(_mergeRanks, newVocab.Count);

        _utf8Vocab = newUtf8Vocab;

        // 重建 byte fallback 直接索引
        if (_byteFallback)
        {
            Array.Fill(_byteFallbackIds, uint.MaxValue);
            for (int b = 0; b < 256; b++)
            {
                var hexToken = RuneHelpers.FormatByteFallbackTokenUtf8((byte)b);
                if (_utf8Vocab.TryGetId(hexToken, out var id))
                    _byteFallbackIds[b] = id;
            }
        }

        // 使缓存失效
        if (_cache is not null)
            Interlocked.Increment(ref _cacheGeneration);
    }

    /// <summary>
    /// 清除分词缓存。
    /// </summary>
    public void ClearCache()
    {
        if (_cache is not null)
            Interlocked.Increment(ref _cacheGeneration);
    }

    /// <summary>
    /// 释放 ThreadLocal 缓存资源。
    /// </summary>
    public void Dispose()
    {
        _cache?.Dispose();
    }

    /// <summary>
    /// 获取当前缓存容量。
    /// </summary>
    public int CacheCapacity => _cacheCapacity;

    /// <inheritdoc />
    public override IReadOnlyList<string> Save(string folder, string? prefix = null)
    {
        Directory.CreateDirectory(folder);

        var prefixStr = prefix is not null ? $"{prefix}." : "";
        var vocabPath = Path.Combine(folder, $"{prefixStr}vocab.json");
        var mergesPath = Path.Combine(folder, $"{prefixStr}merges.txt");

        using (var fs = File.Create(vocabPath))
        using (var writer = new Utf8JsonWriter(fs, new JsonWriterOptions { Indented = false }))
        {
            writer.WriteStartObject();
            foreach (var kvp in _vocab.OrderBy(k => k.Value))
            {
                writer.WritePropertyName(kvp.Key);
                writer.WriteNumberValue(kvp.Value);
            }
            writer.WriteEndObject();
        }

        using (var writer = new StreamWriter(mergesPath, false, global::System.Text.Encoding.UTF8))
        {
            writer.WriteLine("#version: 0.2");
            foreach (var (first, second) in _merges)
                writer.WriteLine($"{first} {second}");
        }

        return new List<string> { vocabPath, mergesPath };
    }

    /// <inheritdoc />

    /// <inheritdoc />
    public override IEnumerable<string> GetMerges()
    {
        foreach (var (first, second) in _merges)
            yield return $"{first} {second}";
    }

    /// <inheritdoc />
    public override string? ContinuingSubwordPrefix => _continuingSubwordPrefix;

    /// <inheritdoc />
    public override string? EndOfWordSuffix => _endOfWordSuffix;

    /// <summary>
    /// 从 vocab.json 和 merges.txt 文件加载 BPE 模型。
    /// </summary>
    public static BpeModel Load(
        string vocabPath,
        string mergesPath,
        string? unkToken = null,
        string? continuingSubwordPrefix = null,
        string? endOfWordSuffix = null,
        bool fuseUnk = false,
        bool byteFallback = false,
        bool ignoreMerges = false,
        int cacheCapacity = 10_000,
        float dropout = 0f)
    {
        var vocab = LoadVocab(vocabPath);
        var merges = LoadMerges(mergesPath);

        return new BpeModel(vocab, merges, unkToken, continuingSubwordPrefix,
            endOfWordSuffix, fuseUnk, byteFallback, ignoreMerges, cacheCapacity, dropout);
    }

    private static Dictionary<string, uint> LoadVocab(string path)
    {
        var json = File.ReadAllText(path);
        var vocab = new Dictionary<string, uint>(StringComparer.Ordinal);

        using var doc = JsonDocument.Parse(json);
        foreach (var property in doc.RootElement.EnumerateObject())
            vocab[property.Name] = property.Value.GetUInt32();

        return vocab;
    }

    private static List<(string, string)> LoadMerges(string path)
    {
        var merges = new List<(string, string)>();
        foreach (var line in File.ReadLines(path))
        {
            if (string.IsNullOrWhiteSpace(line) || line.StartsWith('#')) continue;
            var parts = line.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 2) merges.Add((parts[0], parts[1]));
        }
        return merges;
    }

    /// <summary>
    /// 构建器，用于构建 <see cref="BpeModel"/> 实例的工厂方法。
    /// </summary>
    public sealed class BpeBuilder
    {
        private Dictionary<string, uint>? _vocab;
        private List<(string, string)>? _merges;
        private string? _unkToken;
        private string? _continuingSubwordPrefix;
        private string? _endOfWordSuffix;
        private bool _fuseUnk;
        private bool _byteFallback;
        private bool _ignoreMerges;
        private int _cacheCapacity = 10_000;
        private float _dropout;

        /// <summary>设置词表（token → ID 映射）。</summary>
        public BpeBuilder SetVocab(Dictionary<string, uint> vocab) { _vocab = vocab; return this; }
        /// <summary>设置合并对列表。</summary>
        public BpeBuilder SetMerges(List<(string, string)> merges) { _merges = merges; return this; }
        /// <summary>设置未知 token。</summary>
        public BpeBuilder SetUnkToken(string? unkToken) { _unkToken = unkToken; return this; }
        /// <summary>设置连续子词前缀（如 GPT-2 的 "Ġ"）。</summary>
        public BpeBuilder SetContinuingSubwordPrefix(string? prefix) { _continuingSubwordPrefix = prefix; return this; }
        /// <summary>设置词尾后缀（如 "</w>"）。</summary>
        public BpeBuilder SetEndOfWordSuffix(string? suffix) { _endOfWordSuffix = suffix; return this; }
        /// <summary>设置是否融合连续 UNK token。</summary>
        public BpeBuilder SetFuseUnk(bool fuseUnk) { _fuseUnk = fuseUnk; return this; }
        /// <summary>设置是否启用 byte fallback（未知字节展开为 &lt;0xNN&gt;）。</summary>
        public BpeBuilder SetByteFallback(bool byteFallback) { _byteFallback = byteFallback; return this; }
        /// <summary>设置是否忽略合并（直接返回整个 token）。</summary>
        public BpeBuilder SetIgnoreMerges(bool ignoreMerges) { _ignoreMerges = ignoreMerges; return this; }
        /// <summary>设置分词缓存容量（0 禁用缓存）。</summary>
        public BpeBuilder SetCacheCapacity(int capacity) { _cacheCapacity = capacity; return this; }
        /// <summary>设置合并 dropout 概率（0 不 dropout，1 完全不合并）。</summary>
        public BpeBuilder SetDropout(float dropout) { _dropout = dropout; return this; }

        /// <summary>构建 <see cref="BpeModel"/> 实例。</summary>
        public BpeModel Build()
        {
            var vocab = _vocab ?? throw new InvalidOperationException("Vocabulary must be set before building.");
            var merges = _merges ?? new List<(string, string)>();

            return new BpeModel(vocab, merges, _unkToken, _continuingSubwordPrefix,
                _endOfWordSuffix, _fuseUnk, _byteFallback, _ignoreMerges, _cacheCapacity, _dropout);
        }
    }
}
