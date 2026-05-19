namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 管理在模型词表之上添加的 token。
/// 处理特殊 token（[CLS]、[SEP]、&lt;s&gt;、&lt;/s&gt;）和用户添加的 token。
///
/// 实现与 Rust 参考一致的两阶段拆分策略：
///   阶段 1：对原始文本按非标准化 token（normalized=false）拆分。
///   阶段 2：标准化非 token 部分，然后按标准化 token（normalized=true）拆分。
///
/// 使用 Aho-Corasick 自动机进行多模式匹配（而非正则表达式），
/// 与 Rust 的 DoubleArrayAhoCorasick 方式一致，具有更优的渐近性能。
/// </summary>
public sealed class AddedVocabulary
{
    // Token content → ID mapping (added tokens override model vocab)
    private readonly Dictionary<string, uint> _addedTokensEncoder = new();

    // ID → AddedToken mapping
    private readonly Dictionary<uint, AddedToken> _addedTokensDecoder = new();

    // All AddedToken objects keyed by content
    private readonly Dictionary<string, AddedToken> _addedTokens = new();

    // Phase 1 AC matcher: matches non-normalized tokens (normalized=false) against raw text
    private AhoCorasickMatcher? _splitMatcher;
    private MatcherEntry[] _splitEntries = [];

    // Phase 2 AC matcher: matches normalized tokens (normalized=true) against normalized text
    private AhoCorasickMatcher? _splitNormalizedMatcher;
    private MatcherEntry[] _splitNormalizedEntries = [];

    // Cache: token ID → normalized content (for normalized=true tokens)
    private readonly Dictionary<uint, string> _normalizedContentCache = new();

    // Reverse lookup: normalized content → token ID (for resolving Phase 2 matches)
    private readonly Dictionary<string, uint> _normalizedContentToId = new();

    // Whether to encode special tokens in the input text
    private bool _encodeSpecialTokens;

    /// <summary>
    /// 将 AC 模式索引映射回原始 AddedToken 及其词表 ID。
    /// </summary>
    private readonly record struct MatcherEntry(AddedToken Token, uint TokenId);

    /// <summary>获取已添加 token → ID 的映射。</summary>
    public IReadOnlyDictionary<string, uint> GetAddedTokens()
        => _addedTokensEncoder;

    /// <summary>获取 ID → AddedToken 的映射。</summary>
    public IReadOnlyDictionary<uint, AddedToken> GetAddedTokensDecoder()
        => _addedTokensDecoder;

    /// <summary>是否应从输入文本中编码特殊 token。</summary>
    public bool EncodeSpecialTokens
    {
        get => _encodeSpecialTokens;
        set
        {
            if (_encodeSpecialTokens != value)
            {
                _encodeSpecialTokens = value;
                RebuildMatcher();
            }
        }
    }

    /// <summary>将 token 转换为 ID，优先查找已添加的 token，然后查找模型词表。</summary>
    public uint? TokenToId(string token, IModel model)
    {
        if (_addedTokensEncoder.TryGetValue(token, out var id))
            return id;
        return null; // 不再回退到 model，由调用方处理
    }

    /// <summary>
    /// 将 ID 转换为 token 字符串，优先查找已添加的 token，然后查找模型词表。
    /// 对于标准化 token，返回缓存的标准化形式（用于解码）。
    /// 与 Rust 的 simple_id_to_token 行为一致。
    /// </summary>
    public string? IdToToken(uint id, IModel model)
    {
        if (_addedTokensDecoder.TryGetValue(id, out var addedToken))
        {
            // Return normalized form if available (for decoding)
            if (_normalizedContentCache.TryGetValue(id, out var normalizedContent))
                return normalizedContent;
            return addedToken.Content;
        }
        return model.IdToToken(id);
    }

    /// <summary>检查指定内容的 token 是否为特殊 token。</summary>
    public bool IsSpecialToken(string token)
        => _addedTokens.TryGetValue(token, out var addedToken) && addedToken.IsSpecial;

    /// <summary>
    /// 向词表添加特殊 token。
    /// </summary>
    /// <returns>实际新增的 token 数量。</returns>
    public int AddSpecialTokens(IEnumerable<AddedToken> tokens, IModel model, INormalizer? normalizer)
    {
        int added = 0;

        foreach (var token in tokens)
        {
            // Force IsSpecial = true
            var specialToken = new AddedToken(
                token.Content,
                id: token.Id,
                isSpecial: true,
                lStrip: token.LStrip,
                rStrip: token.RStrip,
                singleWord: token.SingleWord,
                normalized: token.Normalized);

            uint id;
            if (!_addedTokensEncoder.ContainsKey(specialToken.Content))
            {
                // 优先使用 token 指定的 id，否则自动分配
                id = specialToken.Id ?? NextAvailableId(model);
                _addedTokensEncoder[specialToken.Content] = id;
                _addedTokensDecoder[id] = specialToken;
                _addedTokens[specialToken.Content] = specialToken;
                added++;
            }
            else
            {
                // Update metadata even if already present
                id = _addedTokensEncoder[specialToken.Content];
                _addedTokensDecoder[id] = specialToken;
                _addedTokens[specialToken.Content] = specialToken;
            }

            CacheNormalizedContent(id, specialToken, normalizer);
        }

        RebuildMatcher();
        return added;
    }

    /// <summary>
    /// 向词表添加常规（非特殊）token。
    /// </summary>
    /// <returns>实际新增的 token 数量。</returns>
    public int AddTokens(IEnumerable<AddedToken> tokens, IModel model, INormalizer? normalizer)
    {
        int added = 0;

        foreach (var token in tokens)
        {
            uint id;
            if (!_addedTokensEncoder.ContainsKey(token.Content))
            {
                // 优先使用 token 指定的 id，否则自动分配
                id = token.Id ?? NextAvailableId(model);
                _addedTokensEncoder[token.Content] = id;
                _addedTokensDecoder[id] = token;
                _addedTokens[token.Content] = token;
                added++;
            }
            else
            {
                // Update metadata
                id = _addedTokensEncoder[token.Content];
                _addedTokensDecoder[id] = token;
                _addedTokens[token.Content] = token;
            }

            CacheNormalizedContent(id, token, normalizer);
        }

        if (added > 0)
            RebuildMatcher();

        return added;
    }

    /// <summary>
    /// 使用两阶段拆分策略从输入文本中提取已添加的 token：
    ///   阶段 1：对原始文本按非标准化 token（normalized=false）拆分。
    ///   阶段 2：标准化非 token 部分，然后按标准化 token（normalized=true）拆分。
    /// 返回已拆分并完成分词的 PreTokenizedString。
    /// </summary>
    public PreTokenizedString ExtractAndNormalize(INormalizer? normalizer, ReadOnlySpan<char> text)
    {
        // No added tokens at all — just normalize everything
        if (_addedTokensEncoder.Count == 0)
        {
            var normalized = new NormalizedString(text);
            normalizer?.Normalize(normalized);
            return new PreTokenizedString(text.ToString(), [new Split(normalized)]);
        }

        // 有 added tokens 时需要 string（AC 匹配、切片）
        var textStr = text.ToString();

        // ===== Phase 1: Split on non-normalized tokens (normalized=false) =====
        var splits = new List<Split>();

        if (_splitMatcher is not null && _splitEntries.Length > 0)
        {
            var rawMatches = _splitMatcher.FindLeftmostLongest(textStr);
            var candidates = ApplyFlags(rawMatches, _splitEntries, textStr);

            int lastIndex = 0;
            foreach (var (pid, matchStart, matchEnd, contentStart, contentEnd) in candidates)
            {
                if (matchStart < lastIndex)
                    continue; // overlapping — skip

                // Non-added-token part before this match
                if (matchStart > lastIndex)
                    splits.Add(new Split(new NormalizedString(textStr.AsSpan(lastIndex, matchStart - lastIndex))));

                // The added token — already "tokenized"
                var entry = _splitEntries[pid];
                var token = new Token(entry.TokenId, entry.Token.Content, contentStart, contentEnd);
                splits.Add(new Split(new NormalizedString(entry.Token.Content), [token]));

                lastIndex = matchEnd;
            }

            // Remaining text after last match
            if (lastIndex < textStr.Length)
                splits.Add(new Split(new NormalizedString(textStr.AsSpan(lastIndex))));
        }
        else
        {
            // No non-normalized token regex — the entire text is non-token
            splits.Add(new Split(new NormalizedString(text)));
        }

        // ===== Phase 2: Normalize non-token parts, split on normalized tokens =====
        if (_splitNormalizedMatcher is not null && _splitNormalizedEntries.Length > 0)
        {
            var phase2Splits = new List<Split>(splits.Count);

            foreach (var split in splits)
            {
                // Token splits from Phase 1 pass through unchanged
                if (split.Tokens is not null)
                {
                    phase2Splits.Add(split);
                    continue;
                }

                // Normalize this non-token part
                normalizer?.Normalize(split.Normalized);
                var normalizedText = split.Normalized.GetSpan();

                if (normalizedText.IsEmpty)
                    continue;

                // Split the normalized text on normalized token patterns
                var rawMatches = _splitNormalizedMatcher.FindLeftmostLongest(normalizedText);
                var candidates = ApplyFlags(rawMatches, _splitNormalizedEntries, split.Normalized.GetSpan());

                int subLastIndex = 0;
                foreach (var (pid, matchStart, matchEnd, contentStart, contentEnd) in candidates)
                {
                    if (matchStart < subLastIndex)
                        continue; // overlapping — skip

                    // Non-token part before this match
                    if (matchStart > subLastIndex)
                    {
                        var slice = split.Normalized.Slice(subLastIndex, matchStart - subLastIndex);
                        if (!slice.IsEmpty)
                            phase2Splits.Add(new Split(slice));
                    }

                    // The normalized token match
                    var entry = _splitNormalizedEntries[pid];
                    var token = new Token(entry.TokenId, entry.Token.Content, contentStart, contentEnd);
                    var matchSlice = split.Normalized.Slice(matchStart, matchEnd - matchStart);
                    phase2Splits.Add(new Split(matchSlice, [token]));

                    subLastIndex = matchEnd;
                }

                // Remaining text after last normalized token match
                if (subLastIndex < normalizedText.Length)
                {
                    var slice = split.Normalized.Slice(subLastIndex, normalizedText.Length - subLastIndex);
                    if (!slice.IsEmpty)
                        phase2Splits.Add(new Split(slice));
                }
            }

            if (phase2Splits.Count == 0)
                phase2Splits.Add(new Split(new NormalizedString("")));

            return new PreTokenizedString(textStr, phase2Splits);
        }
        else
        {
            // No normalized token regex — just normalize non-token splits
            foreach (var split in splits)
            {
                if (split.Tokens is null)
                    normalizer?.Normalize(split.Normalized);
            }
        }

        if (splits.Count == 0)
            splits.Add(new Split(new NormalizedString("")));

        return new PreTokenizedString(textStr, splits);
    }

    /// <summary>
    /// 对原始 AC 匹配结果应用 LStrip/RStrip/SingleWord 标志并解决重叠。
    /// 返回按位置排序的非重叠匹配结果。
    /// </summary>
    private static List<(int pid, int matchStart, int matchEnd, int contentStart, int contentEnd)> ApplyFlags(
        List<(int patternIndex, int start, int length)> rawMatches,
        MatcherEntry[] entries,
        ReadOnlySpan<char> text)
    {
        var candidates = new List<(int pid, int matchStart, int matchEnd, int contentStart, int contentEnd)>(rawMatches.Count);

        foreach (var (pid, contentStart, contentLen) in rawMatches)
        {
            var entry = entries[pid];
            int contentEnd = contentStart + contentLen;

            // SingleWord: check word boundaries at content edges
            if (entry.Token.SingleWord)
            {
                if (contentStart > 0 && IsWordChar(text[contentStart - 1]))
                    continue;
                if (contentEnd < text.Length && IsWordChar(text[contentEnd]))
                    continue;
            }

            int matchStart = contentStart;
            int matchEnd = contentEnd;

            // LStrip: extend match start backwards over leading whitespace
            if (entry.Token.LStrip)
            {
                while (matchStart > 0 && char.IsWhiteSpace(text[matchStart - 1]))
                    matchStart--;
            }

            // RStrip: extend match end forwards over trailing whitespace
            if (entry.Token.RStrip)
            {
                while (matchEnd < text.Length && char.IsWhiteSpace(text[matchEnd]))
                    matchEnd++;
            }

            candidates.Add((pid, matchStart, matchEnd, contentStart, contentEnd));
        }

        // Sort by match start, then by match end (longest first for same start)
        candidates.Sort((a, b) =>
        {
            int cmp = a.matchStart.CompareTo(b.matchStart);
            return cmp != 0 ? cmp : b.matchEnd.CompareTo(a.matchEnd);
        });

        return candidates;
    }

    /// <summary>
    /// 检查字符是否为"词字符"（匹配正则表达式 \w）。
    /// </summary>
    private static bool IsWordChar(char c) => char.IsLetterOrDigit(c) || c == '_';

    /// <summary>
    /// 缓存 token 的标准化内容（用于 normalized=true 的 token）。
    /// 用于阶段 2 的匹配和解码。
    /// </summary>
    private void CacheNormalizedContent(uint id, AddedToken token, INormalizer? normalizer)
    {
        if (!token.Normalized)
        {
            _normalizedContentCache.Remove(id);
            return;
        }

        if (normalizer is not null)
        {
            var ns = new NormalizedString(token.Content);
            normalizer.Normalize(ns);
            var normed = ns.GetSpan();
            if (!normed.SequenceEqual(token.Content.AsSpan()))
            {
                _normalizedContentCache[id] = normed.ToString();
                return;
            }
        }

        // No normalizer or normalization didn't change the content — remove cache entry
        _normalizedContentCache.Remove(id);
    }

    /// <summary>
    /// 在添加 token 或 EncodeSpecialTokens 变更后重建内部 Aho-Corasick 匹配器。
    /// 构建：
    ///   - _splitMatcher：用于非标准化 token（normalized=false）
    ///   - _splitNormalizedMatcher：用于标准化 token（normalized=true）
    /// 同时重建标准化内容 → ID 的反向查找表。
    /// </summary>
    private void RebuildMatcher()
    {
        var nonNormPatterns = new List<string>();
        var nonNormEntries = new List<MatcherEntry>();
        var normPatterns = new List<string>();
        var normEntries = new List<MatcherEntry>();
        _normalizedContentToId.Clear();

        foreach (var kv in _addedTokens)
        {
            var token = kv.Value;

            // Special tokens only participate if EncodeSpecialTokens is true
            if (token.IsSpecial && !_encodeSpecialTokens)
                continue;

            uint id = _addedTokensEncoder[kv.Key];

            if (token.Normalized)
            {
                // Get normalized content (from cache or use original)
                var normalizedContent = _normalizedContentCache.GetValueOrDefault(id, token.Content);
                _normalizedContentToId[normalizedContent] = id;

                normPatterns.Add(normalizedContent);
                normEntries.Add(new MatcherEntry(token, id));
            }
            else
            {
                nonNormPatterns.Add(token.Content);
                nonNormEntries.Add(new MatcherEntry(token, id));
            }
        }

        _splitMatcher = nonNormPatterns.Count > 0
            ? new AhoCorasickMatcher(nonNormPatterns)
            : null;
        _splitEntries = nonNormEntries.ToArray();

        _splitNormalizedMatcher = normPatterns.Count > 0
            ? new AhoCorasickMatcher(normPatterns)
            : null;
        _splitNormalizedEntries = normEntries.ToArray();
    }

    /// <summary>
    /// 获取已添加 token 的下一个可用 ID。
    /// </summary>
    private uint NextAvailableId(IModel model)
    {
        uint baseId = (uint)model.GetVocabSize();
        uint id = baseId;

        while (_addedTokensDecoder.ContainsKey(id))
            id++;

        return id;
    }
}
