using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Models.Unigram;

/// <summary>
/// Unigram 语言分词模型。
/// 使用 Viterbi 算法找到最可能的分词结果。
/// 用于 SentencePiece 和 T5 模型。
/// </summary>
[TokenizerComponent("Unigram")]
public sealed class UnigramModel : IModel
{
    private readonly List<(string Token, double LogProb)> _vocab;
    private readonly Dictionary<string, uint> _tokenIdMap;
    private readonly Dictionary<uint, string> _idTokenMap;
    private readonly Trie _trie;
    private readonly UnigramDoubleArrayTrie? _dat;
    private string? _unkToken;
    private readonly double _unkScore;
    private readonly bool _byteFallback;
    private readonly bool _fuseUnk;
    private readonly double _alpha;
    private readonly int _nbestSize;

    // ByteFallback 直接索引数组：byte → token_id，O(1) 无哈希
    private readonly uint[] _byteFallbackIds;

    // Tokenization 缓存（与 Rust Cache 对齐）：纯 Viterbi 模式下缓存短文本结果
    private readonly ConcurrentDictionary<string, List<Token>>? _cache;
    private const int MaxCacheKeyLength = 256;
    private const int MaxCacheSize = 10_000;

    public IReadOnlyList<(string Token, double LogProb)> Vocab => _vocab;

    /// <summary>
    /// 是否启用 byte_fallback（未知 token 展开为 &lt;0xNN&gt;）。
    /// </summary>
    public bool ByteFallback => _byteFallback;

    /// <summary>
    /// 采样温度。alpha > 0 时启用 nbest 采样路径（encode_unoptimized）。
    /// alpha = 0 时使用 Viterbi 最优路径（encode_optimized）。
    /// </summary>
    public double Alpha => _alpha;

    /// <summary>
    /// nbest 采样大小。nbest_size > 0 且 alpha > 0 时从 nbest 个候选中采样。
    /// </summary>
    public int NbestSize => _nbestSize;

    private UnigramModel(
        List<(string Token, double LogProb)> vocab,
        string? unkToken,
        double unkScore,
        bool byteFallback,
        bool fuseUnk,
        double alpha = 0.0,
        int nbestSize = 0)
    {
        _vocab = vocab;
        _tokenIdMap = new Dictionary<string, uint>(vocab.Count, StringComparer.Ordinal);
        _idTokenMap = new Dictionary<uint, string>(vocab.Count);
        _trie = new Trie();

        for (int i = 0; i < vocab.Count; i++)
        {
            _tokenIdMap[vocab[i].Token] = (uint)i;
            _idTokenMap[(uint)i] = vocab[i].Token;
            _trie.Insert(vocab[i].Token, i);
        }

        _unkToken = unkToken;
        // UNK score：如果未显式指定（NaN），则使用 min_score - 10.0（与 Rust 一致）
        _unkScore = double.IsNaN(unkScore) ? ComputeUnkScore() : unkScore;
        _byteFallback = byteFallback;
        _fuseUnk = fuseUnk;
        _alpha = alpha;
        _nbestSize = nbestSize;

        // 预计算 byte → token_id 直接索引
        _byteFallbackIds = new uint[256];
        if (_byteFallback)
        {
            Array.Fill(_byteFallbackIds, uint.MaxValue);
            for (int b = 0; b < 256; b++)
            {
                var byteToken = Internal.RuneHelpers.FormatByteFallbackToken((byte)b);
                if (_tokenIdMap.TryGetValue(byteToken, out var byteId))
                    _byteFallbackIds[b] = byteId;
            }
        }

        // 纯 Viterbi 模式（alpha == 0）启用缓存
        _cache = alpha == 0.0 ? new ConcurrentDictionary<string, List<Token>>(StringComparer.Ordinal) : null;

        // 构建 DAT（仅在有词表时）
        if (vocab.Count > 0)
        {
            var sortedTokens = new List<(string Token, int TokenId)>(vocab.Count);
            for (int i = 0; i < vocab.Count; i++)
                sortedTokens.Add((vocab[i].Token, i));
            sortedTokens.Sort((a, b) => string.CompareOrdinal(a.Token, b.Token));
            _dat = UnigramDoubleArrayTrie.Build(sortedTokens);
        }
    }

    /// <inheritdoc />
    public List<Token> Tokenize(ReadOnlySpan<char> sequence)
    {
        if (sequence.IsEmpty)
            return [];

        // 缓存查找（纯 Viterbi 模式 + 短文本）：零分配通过 AlternateLookup
        if (_cache is not null && sequence.Length <= MaxCacheKeyLength)
        {
            var cacheLookup = _cache.GetAlternateLookup<ReadOnlySpan<char>>();
            if (cacheLookup.TryGetValue(sequence, out var cached))
                return cached;
        }

        // Unigram 内部需要 string（Lattice、EncodeOptimized），此处转换一次
        var seqStr = sequence.ToString();

        // alpha == 0 且 nbestSize <= 0 时使用快速路径（不构建 Lattice）
        List<Token> result;
        if (_alpha == 0.0 && _nbestSize <= 0)
            result = EncodeOptimized(seqStr);
        else
            result = EncodeUnoptimized(seqStr);

        // 缓存存储
        if (_cache is not null && seqStr.Length <= MaxCacheKeyLength && _cache.Count < MaxCacheSize)
            _cache.TryAdd(seqStr, result);

        return result;
    }

    /// <summary>
    /// 快速 Viterbi 路径（与 Rust encode_optimized 对齐）。
    /// 不构建 Lattice，直接用一维 DP 数组，减少 ~60-80% 内存分配。
    /// </summary>
    /// <summary>
    /// DP 节点，用于 EncodeOptimized 的 Viterbi 路径。
    /// 合并 score/startsAt/tokenId 为单一结构体，提升缓存局部性。
    /// </summary>
    private struct BestPathNode
    {
        public double BestPathScore;
        public int StartsAt;   // -1 表示未访问
        public int TokenId;    // -1 表示 UNK
    }

    private List<Token> EncodeOptimized(string sequence)
    {
        int size = sequence.Length;
        double unkScore = _unkScore;

        // 池化分配：避免每次调用堆分配 3 个数组（参考 Microsoft ML BestPathNode）
        var bestPathEndsAt = System.Buffers.ArrayPool<BestPathNode>.Shared.Rent(size + 1);
        // 初始化 DP 数组（ArrayPool 可能返回更大缓冲区，只初始化需要的部分）
        for (int i = 0; i <= size; i++)
            bestPathEndsAt[i] = new BestPathNode { BestPathScore = 0, StartsAt = -1, TokenId = -1 };

        // 可复用的前缀缓冲区
        var prefixBuffer = new List<(int End, int TokenId)>(32);

        int startPos = 0;
        while (startPos < size)
        {
            double bestScoreTillHere = bestPathEndsAt[startPos].BestPathScore;
            bool hasSingleNode = false;

            // 获取当前码位长度
            if (!Rune.TryGetRuneAt(sequence, startPos, out var rune))
            {
                // 无效代理对，跳过
                int nextPos = startPos + 1;
                double candidate = unkScore + bestScoreTillHere;
                ref var node = ref bestPathEndsAt[nextPos];
                if (node.StartsAt < 0 || candidate > node.BestPathScore)
                {
                    node.BestPathScore = candidate;
                    node.StartsAt = startPos;
                    node.TokenId = -1; // UNK
                }
                startPos = nextPos;
                continue;
            }
            int mblen = rune.Utf16SequenceLength;

            if (_dat is not null)
                _dat.FindPrefixes(sequence, startPos, prefixBuffer);
            else
                _trie.FindPrefixes(sequence, startPos, prefixBuffer);
            for (int j = 0; j < prefixBuffer.Count; j++)
            {
                var (end, id) = prefixBuffer[j];
                double score = _vocab[id].LogProb;
                int keyPos = end;
                double candidate = score + bestScoreTillHere;

                ref var targetNode = ref bestPathEndsAt[keyPos];
                if (targetNode.StartsAt < 0 || candidate > targetNode.BestPathScore)
                {
                    targetNode.BestPathScore = candidate;
                    targetNode.StartsAt = startPos;
                    targetNode.TokenId = id;
                }
                if (!hasSingleNode && end - startPos == mblen)
                    hasSingleNode = true;
            }

            if (!hasSingleNode)
            {
                int keyPos = startPos + mblen;
                double candidate = unkScore + bestScoreTillHere;
                ref var targetNode = ref bestPathEndsAt[keyPos];
                if (targetNode.StartsAt < 0 || candidate > targetNode.BestPathScore)
                {
                    targetNode.BestPathScore = candidate;
                    targetNode.StartsAt = startPos;
                    targetNode.TokenId = -1; // UNK
                }
            }
            startPos += mblen;
        }

        // 回溯重建路径（带 fuse_unk + byte_fallback）
        // 单次回溯：收集结果后 Reverse，byte_fallback 展开只执行一次
        var rawTokensList = new List<(string Surface, uint Id)>(size / 4);
        {
            int p = size;
            while (p > 0)
            {
                ref var node = ref bestPathEndsAt[p];
                int s = node.StartsAt;
                if (s < 0) break;
                int tid = node.TokenId;
                var surface = sequence[s..p];

                if (tid >= 0)
                {
                    rawTokensList.Add((surface, (uint)tid));
                }
                else if (_byteFallback)
                {
                    var utf8Bytes = System.Text.Encoding.UTF8.GetBytes(surface);
                    int initCount = rawTokensList.Count;
                    bool allExpanded = true;
                    for (int b = 0; b < utf8Bytes.Length; b++)
                    {
                        uint byteId = _byteFallbackIds[utf8Bytes[b]];
                        if (byteId != uint.MaxValue)
                            rawTokensList.Add((Internal.RuneHelpers.FormatByteFallbackToken(utf8Bytes[b]), byteId));
                        else
                        {
                            allExpanded = false;
                            break;
                        }
                    }
                    if (!allExpanded)
                    {
                        // 回滚本次添加的 byte tokens
                        rawTokensList.RemoveRange(initCount, rawTokensList.Count - initCount);
                        if (_unkToken is not null && _tokenIdMap.TryGetValue(_unkToken, out var unkId))
                            rawTokensList.Add((surface, unkId));
                        else
                            rawTokensList.Add((surface, 0));
                    }
                }
                else if (_unkToken is not null && _tokenIdMap.TryGetValue(_unkToken, out var unkId2))
                {
                    rawTokensList.Add((surface, unkId2));
                }
                else
                {
                    rawTokensList.Add((surface, 0));
                }

                p = s;
            }
        }

        // 归还池化数组
        System.Buffers.ArrayPool<BestPathNode>.Shared.Return(bestPathEndsAt);

        rawTokensList.Reverse();
        var rawTokens = rawTokensList.ToArray();

        // fuse_unk + 构建最终 Token 列表
        return BuildFinalTokens(rawTokens);
    }

    /// <summary>
    /// 完整 Lattice 路径（采样/nbest 场景）。
    /// </summary>
    private List<Token> EncodeUnoptimized(string sequence)
    {
        var lattice = BuildLattice(sequence);

        IReadOnlyList<Lattice.Node> path;
        if (_alpha > 0.0 && _nbestSize > 0)
            path = lattice.Sample(_alpha);
        else
            path = lattice.Viterbi();

        var rawTokens = new List<(string Surface, uint Id)>(path.Count);
        foreach (var node in path)
        {
            var surface = lattice.Piece(node);
            if (_tokenIdMap.TryGetValue(surface, out var id))
            {
                rawTokens.Add((surface, id));
            }
            else if (_byteFallback)
            {
                var utf8Bytes = System.Text.Encoding.UTF8.GetBytes(surface);
                bool allExpanded = true;
                var expandedTokens = new List<(string Surface, uint Id)>(utf8Bytes.Length);
                foreach (var b in utf8Bytes)
                {
                    uint byteId = _byteFallbackIds[b];
                    if (byteId != uint.MaxValue)
                        expandedTokens.Add((Internal.RuneHelpers.FormatByteFallbackToken(b), byteId));
                    else
                    {
                        allExpanded = false;
                        break;
                    }
                }
                if (allExpanded)
                    rawTokens.AddRange(expandedTokens);
                else if (_unkToken is not null && _tokenIdMap.TryGetValue(_unkToken, out var unkId))
                    rawTokens.Add((surface, unkId));
                else
                    rawTokens.Add((surface, (uint)node.Id));
            }
            else if (_unkToken is not null && _tokenIdMap.TryGetValue(_unkToken, out var unkId2))
            {
                rawTokens.Add((surface, unkId2));
            }
            else
            {
                rawTokens.Add((surface, (uint)node.Id));
            }
        }

        Lattice.Return(lattice);
        return BuildFinalTokens(rawTokens.ToArray());
    }

    /// <summary>
    /// 通用后处理：fuse_unk + 构建最终 Token 列表。
    /// EncodeOptimized 和 EncodeUnoptimized 共用此逻辑。
    /// </summary>
    private List<Token> BuildFinalTokens((string Surface, uint Id)[] rawTokens)
    {
        var tokens = new List<Token>(rawTokens.Length);
        int offset = 0;

        if (_fuseUnk && _unkToken is not null && _tokenIdMap.TryGetValue(_unkToken, out var fuseUnkId))
        {
            var fusedSb = new ValueStringBuilder(stackalloc char[128]);
            int fusedStart = 0;

            for (int i = 0; i < rawTokens.Length; i++)
            {
                var (surface, tokenId) = rawTokens[i];

                if (tokenId == fuseUnkId)
                {
                    if (fusedSb.Length == 0)
                        fusedStart = offset;
                    fusedSb.Append(surface);
                }
                else
                {
                    if (fusedSb.Length > 0)
                    {
                        var fused = fusedSb.ToString();
                        tokens.Add(new Token(fuseUnkId, fused, fusedStart, fusedStart + fused.Length));
                        fusedSb.Length = 0;
                    }
                    tokens.Add(new Token(tokenId, surface, offset, offset + surface.Length));
                }

                offset += surface.Length;
            }

            if (fusedSb.Length > 0)
            {
                var fused = fusedSb.ToString();
                tokens.Add(new Token(fuseUnkId, fused, fusedStart, fusedStart + fused.Length));
            }
        }
        else
        {
            foreach (var (surface, tokenId) in rawTokens)
            {
                int start = offset;
                int end = offset + surface.Length;
                tokens.Add(new Token(tokenId, surface, start, end));
                offset = end;
            }
        }

        return tokens;
    }

    private Lattice BuildLattice(string sequence)
    {
        // BOS/EOS ID — 使用词表大小作为哨兵 ID（不在词表中）
        int sentinelBos = _vocab.Count;
        int sentinelEos = _vocab.Count + 1;
        var lattice = Lattice.Rent(sequence, sentinelBos, sentinelEos);

        // 可复用的前缀结果缓冲区，避免每帧分配 List
        var prefixBuffer = new List<(int End, int TokenId)>(32);

        // 使用 Rune 遍历，与修改后的 Trie 保持一致
        // i 是 char 索引（UTF-16 偏移量），用于 Lattice 的 pos/length
        int i = 0;
        while (i < sequence.Length)
        {
            if (_dat is not null)
                _dat.FindPrefixes(sequence, i, prefixBuffer);
            else
                _trie.FindPrefixes(sequence, i, prefixBuffer);

            if (prefixBuffer.Count > 0)
            {
                for (int j = 0; j < prefixBuffer.Count; j++)
                {
                    var (end, tokenId) = prefixBuffer[j];
                    var logProb = _vocab[tokenId].LogProb;
                    lattice.Insert(i, end - i, logProb, tokenId);
                }
            }
            else
            {
                // 获取当前位置的完整码位（而非单个 char）
                if (!Rune.TryGetRuneAt(sequence, i, out var rune))
                {
                    // 无效代理对，跳过
                    i++;
                    continue;
                }
                int charLen = rune.Utf16SequenceLength;
                // 直接用码位查找，避免 ToString() 分配
                int charId = _trie.GetTokenIdByCodepoint(rune.Value);

                if (charId >= 0)
                    lattice.Insert(i, charLen, _vocab[charId].LogProb, charId);
                else if (_byteFallback)
                    lattice.Insert(i, charLen, _unkScore, -1);
                else if (_unkToken is not null)
                    lattice.Insert(i, charLen, _unkScore, -1);
                else
                    lattice.Insert(i, charLen, 0.0, -1);

                // 按当前码位的 UTF-16 长度前进（复用 rune 变量）
                i += charLen;
                continue;
            }

            // 按当前码位的 UTF-16 长度前进
            if (!Rune.TryGetRuneAt(sequence, i, out var currentRune))
            {
                i++;
            }
            else
            {
                i += currentRune.Utf16SequenceLength;
            }
        }

        return lattice;
    }

    /// <inheritdoc />
    public uint? TokenToId(string token) =>
        _tokenIdMap.TryGetValue(token, out var id) ? id : null;

    /// <inheritdoc />
    public string? IdToToken(uint id) =>
        _idTokenMap.TryGetValue(id, out var token) ? token : null;

    /// <inheritdoc />
    public IReadOnlyDictionary<string, uint> GetVocab() => _tokenIdMap;

    /// <inheritdoc />
    public int GetVocabSize() => _vocab.Count;

    /// <summary>
    /// 获取 UNK token 的 ID（与 Rust 序列化对齐）。
    /// </summary>
    internal uint? UnkId =>
        _unkToken is not null && _tokenIdMap.TryGetValue(_unkToken, out var id) ? id : null;

    /// <inheritdoc />
    public IReadOnlyList<string> Save(string folder, string? prefix = null)
    {
        Directory.CreateDirectory(folder);
        var prefixStr = prefix is not null ? $"{prefix}." : "";
        var modelPath = Path.Combine(folder, $"{prefixStr}unigram.json");

        using (var fs = File.Create(modelPath))
        using (var writer = new Utf8JsonWriter(fs, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();

            // type 字段（与 Rust 序列化对齐）
            writer.WriteString("type", "Unigram");

            writer.WritePropertyName("vocab");
            writer.WriteStartArray();
            foreach (var (token, logProb) in _vocab)
            {
                writer.WriteStartArray();
                writer.WriteStringValue(token);
                writer.WriteNumberValue(logProb);
                writer.WriteEndArray();
            }
            writer.WriteEndArray();

            // unk_id 字段（与 Rust 序列化对齐：None 时写 null）
            if (_unkToken is not null && _tokenIdMap.TryGetValue(_unkToken, out var unkId))
                writer.WriteNumber("unk_id", unkId);
            else
                writer.WriteNull("unk_id");

            // byte_fallback 字段（与 Rust 序列化对齐）
            writer.WriteBoolean("byte_fallback", _byteFallback);

            writer.WriteEndObject();
        }

        return [modelPath];
    }

    /// <inheritdoc />

    /// <summary>
    /// 使用新的词汇表重建模型（供训练器调用）。
    /// 采用 copy-on-write 策略：构建新数据结构后原子替换引用，
    /// 避免并发读取时看到中间状态。
    /// 保持原有配置（unkToken、unkScore、byteFallback、fuseUnk）不变。
    /// </summary>
    /// <summary>
    /// 重建词表和内部映射（训练 finalize 阶段调用）。
    /// 注意：此方法非线程安全，仅允许在训练期间单线程调用。
    /// 调用时必须确保没有并发的 Encode/Decode 操作。
    /// </summary>
    internal void Rebuild(List<(string Token, double LogProb)> newVocab, string? unkToken = null)
    {
        // 构建新的映射和 Trie（不修改现有实例）
        var newTokenIdMap = new Dictionary<string, uint>(newVocab.Count, StringComparer.Ordinal);
        var newIdTokenMap = new Dictionary<uint, string>(newVocab.Count);

        for (int i = 0; i < newVocab.Count; i++)
        {
            newTokenIdMap[newVocab[i].Token] = (uint)i;
            newIdTokenMap[(uint)i] = newVocab[i].Token;
        }

        // 原子替换词表
        _vocab.Clear();
        _vocab.AddRange(newVocab);

        // 原子替换映射
        _tokenIdMap.Clear();
        foreach (var kvp in newTokenIdMap) _tokenIdMap[kvp.Key] = kvp.Value;

        _idTokenMap.Clear();
        foreach (var kvp in newIdTokenMap) _idTokenMap[kvp.Key] = kvp.Value;

        // 原子替换 Trie
        _trie.Clear();
        for (int i = 0; i < newVocab.Count; i++)
            _trie.Insert(newVocab[i].Token, i);

        // 更新 UNK token（与 Rust unk_id 对齐）
        if (unkToken is not null)
            _unkToken = unkToken;
    }

    /// <summary>
    /// 获取词表列表（供训练器 finalize 使用）。
    /// </summary>
    internal IReadOnlyList<(string Token, double Score)> GetVocabList() => _vocab;

    /// <summary>
    /// 获取所有 piece 中的最小 score（供训练器 finalize 使用）。
    /// </summary>
    internal double MinScore
    {
        get
        {
            double min = double.PositiveInfinity;
            foreach (var (_, score) in _vocab)
            {
                if (!double.IsNaN(score) && score < min)
                    min = score;
            }
            return double.IsPositiveInfinity(min) ? -10 : min;
        }
    }

    /// <summary>
    /// 与 Rust K_UNK_PENALTY 一致的常量。
    /// </summary>
    private const double KUnkPenalty = 10.0;

    /// <summary>
    /// 计算 UNK score：min_score - 10.0（与 Rust Unigram 一致）。
    /// </summary>
    private double ComputeUnkScore() => MinScore - KUnkPenalty;

    /// <summary>
    /// 获取指定 token ID 的 score。
    /// </summary>
    internal double GetScore(int tokenId) =>
        tokenId >= 0 && tokenId < _vocab.Count ? _vocab[tokenId].LogProb : double.NaN;

    /// <summary>
    /// 查找指定位置的所有 token 前缀（供训练器 Lattice 使用）。
    /// 返回 (endPos, tokenId, tokenString) 列表。
    /// </summary>
    internal List<(int End, int TokenId, string Token)> FindPrefixes(string sentence, int pos)
    {
        var result = new List<(int, int, string)>();
        var trieResults = _trie.FindPrefixes(sentence, pos);
        foreach (var (end, tokenId) in trieResults)
        {
            if (tokenId >= 0 && tokenId < _vocab.Count)
                result.Add((end, tokenId, _vocab[tokenId].Token));
        }
        return result;
    }

    public static UnigramModel Load(
        string modelPath,
        string? unkToken = null,
        double unkScore = double.NaN,
        bool byteFallback = false,
        bool fuseUnk = true)
    {
        var json = File.ReadAllText(modelPath);
        var vocab = new List<(string Token, double LogProb)>();

        using var doc = JsonDocument.Parse(json);
        if (doc.RootElement.TryGetProperty("vocab", out var vocabElement))
        {
            foreach (var item in vocabElement.EnumerateArray())
            {
                var token = item[0].GetString()!;
                var logProb = item[1].GetDouble();
                vocab.Add((token, logProb));
            }
        }

        // 读取序列化字段（与 Rust 反序列化对齐）
        if (doc.RootElement.TryGetProperty("byte_fallback", out var bfProp) && bfProp.ValueKind == JsonValueKind.True)
            byteFallback = true;

        if (doc.RootElement.TryGetProperty("unk_id", out var unkIdProp) && unkIdProp.ValueKind == JsonValueKind.Number)
        {
            uint unkId = unkIdProp.GetUInt32();
            if (unkId < vocab.Count)
                unkToken = vocab[(int)unkId].Token;
        }

        return new UnigramModel(vocab, unkToken, unkScore, byteFallback, fuseUnk);
    }

    public sealed class UnigramBuilder
    {
        private List<(string Token, double LogProb)>? _vocab;
        private string? _unkToken;
        private double _unkScore = double.NaN; // NaN → 自动计算 min_score - 10.0
        private bool _byteFallback;
        private bool _fuseUnk = true;
        private double _alpha;
        private int _nbestSize;

        /// <summary>设置词表（token, log概率）列表。</summary>
        public UnigramBuilder SetVocab(List<(string Token, double LogProb)> vocab) { _vocab = vocab; return this; }
        /// <summary>设置未知 token。</summary>
        public UnigramBuilder SetUnkToken(string? unkToken) { _unkToken = unkToken; return this; }
        /// <summary>设置 UNK 评分（NaN 则自动计算 min_score - 10.0）。</summary>
        public UnigramBuilder SetUnkScore(double unkScore) { _unkScore = unkScore; return this; }
        /// <summary>设置是否启用 byte fallback。</summary>
        public UnigramBuilder SetByteFallback(bool byteFallback) { _byteFallback = byteFallback; return this; }
        /// <summary>设置是否融合连续 UNK token。</summary>
        public UnigramBuilder SetFuseUnk(bool fuseUnk) { _fuseUnk = fuseUnk; return this; }
        /// <summary>设置采样温度（alpha > 0 启用 nbest 采样）。</summary>
        public UnigramBuilder SetAlpha(double alpha) { _alpha = alpha; return this; }
        /// <summary>设置 nbest 采样大小。</summary>
        public UnigramBuilder SetNbestSize(int nbestSize) { _nbestSize = nbestSize; return this; }

        /// <summary>构建 UnigramModel 实例。</summary>
        public UnigramModel Build()
        {
            var vocab = _vocab ?? throw new InvalidOperationException("Vocabulary must be set.");
            return new UnigramModel(vocab, _unkToken, _unkScore, _byteFallback, _fuseUnk, _alpha, _nbestSize);
        }

        /// <summary>
        /// 从词表直接创建模型（供训练器使用）。
        /// </summary>
        internal static UnigramModel CreateFromVocab(List<(string Token, double LogProb)> vocab, string? unkToken, bool byteFallback = false, double alpha = 0.0, int nbestSize = 0)
        {
            return new UnigramModel(vocab, unkToken, double.NaN, byteFallback, true, alpha, nbestSize);
        }
    }
}
