using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;
using HuggingFace.Tokenizers.Abstractions;
using HuggingFace.Tokenizers.Internal;
using HuggingFace.Tokenizers.Models.BPE;
using SysEncoding = System.Text.Encoding;

namespace HuggingFace.Tokenizers.Models.WordPiece;

/// <summary>
/// WordPiece 分词模型（BERT 风格）。
/// 实现 BERT、DistilBERT 等模型使用的贪心最长匹配算法。
/// </summary>
[TokenizerComponent("WordPiece")]
public sealed class WordPieceModel : DictionaryVocabModel
{
    private readonly string _continuingSubwordPrefix;
    private readonly int _maxInputCharsPerWord;
    private Utf8Vocab _utf8Vocab;

    private WordPieceModel(
        Dictionary<string, uint> vocab,
        string continuingSubwordPrefix,
        string unkToken,
        int maxInputCharsPerWord) : base(vocab, unkToken)
    {
        _continuingSubwordPrefix = continuingSubwordPrefix;
        _maxInputCharsPerWord = maxInputCharsPerWord;
        _utf8Vocab = new Utf8Vocab(vocab);
    }

    /// <inheritdoc />
    public override List<Token> Tokenize(ReadOnlySpan<char> sequence)
    {
        if (sequence.IsEmpty)
            return [];

        // 快速检查长度：纯 ASCII 时 charLen == sequence.Length
        bool allAscii = true;
        for (int i = 0; i < sequence.Length; i++)
            if (sequence[i] > 0x7F) { allAscii = false; break; }
        int charLen = allAscii ? sequence.Length : CountRunes(sequence);

        if (charLen > _maxInputCharsPerWord)
        {
            if (_vocab.TryGetValue(_unkToken!, out var unkId))
                return [new Token(unkId, _unkToken!, 0, sequence.Length)];
            return [];
        }

        // Greedy longest-match-first algorithm (matches Rust WordPiece exactly)
        var tokens = new List<Token>(sequence.Length / 4 + 1);
        int start = 0;
        int prefixLen = _continuingSubwordPrefix.Length;
        int prefixUtf8Len = prefixLen > 0 ? SysEncoding.UTF8.GetByteCount(_continuingSubwordPrefix) : 0;

        // 一次性编码整个 sequence 并缓存，外循环复用
        int seqUtf8BufSize = SysEncoding.UTF8.GetMaxByteCount(sequence.Length);
        byte[]? seqUtf8Buf = ArrayPool<byte>.Shared.Rent(seqUtf8BufSize);
        int actualSeqUtf8Len = SysEncoding.UTF8.GetBytes(sequence, seqUtf8Buf);

        // UTF-8 查找缓冲区：[prefix | sequence bytes]，每次迭代重新组装
        int totalBufSize = prefixUtf8Len + actualSeqUtf8Len;
        byte[]? utf8Buf = totalBufSize <= seqUtf8BufSize ? seqUtf8Buf : ArrayPool<byte>.Shared.Rent(totalBufSize);

        // 从编码后字节推导每个 char 位置的 UTF-8 字节偏移（避免逐 char GetByteCount）
        // 注意：surrogate pair（high + low）在 UTF-8 中占 4 字节，已在 high surrogate 处计入，
        // low surrogate 不应再贡献字节，否则会导致后续偏移量多算 3 字节。
        Span<int> utf8Offsets = sequence.Length <= 1024 ? stackalloc int[sequence.Length + 1] : new int[sequence.Length + 1];
        utf8Offsets[0] = 0;
        for (int i = 0; i < sequence.Length; i++)
        {
            char c = sequence[i];
            utf8Offsets[i + 1] = utf8Offsets[i] + (c <= 0x7F ? 1 : c <= 0x7FF ? 2 : char.IsHighSurrogate(c) ? 4 : char.IsLowSurrogate(c) ? 0 : 3);
        }

        while (start < sequence.Length)
        {
            int seqLen = sequence.Length - start;
            int actualPrefix = (start > 0 && prefixLen > 0) ? prefixLen : 0;

            // 组装查找字节：[prefix | seqUtf8[start..]]，避免覆盖缓存
            if (actualPrefix > 0)
                SysEncoding.UTF8.GetBytes(_continuingSubwordPrefix.AsSpan(), utf8Buf);
            int remainingSeqUtf8Len = actualSeqUtf8Len - utf8Offsets[start];
            Buffer.BlockCopy(seqUtf8Buf, utf8Offsets[start], utf8Buf, actualPrefix, remainingSeqUtf8Len);

            // 快速路径：尝试整个剩余序列
            int totalUtf8Len = actualPrefix + remainingSeqUtf8Len;

            if (_utf8Vocab.TryGetId(utf8Buf.AsSpan(0, totalUtf8Len), out var fullId))
            {
                // 确认命中后才创建 string
                int totalCharLen = actualPrefix + seqLen;
                char[]? lookupBuf = null;
                Span<char> lookup = totalCharLen <= 256
                    ? stackalloc char[totalCharLen]
                    : (lookupBuf = ArrayPool<char>.Shared.Rent(totalCharLen));
                try
                {
                    if (actualPrefix > 0)
                        _continuingSubwordPrefix.AsSpan().CopyTo(lookup);
                    sequence.Slice(start, seqLen).CopyTo(lookup.Slice(actualPrefix));
                    tokens.Add(new Token(fullId, new string(lookup.Slice(0, totalCharLen)), start, sequence.Length));
                }
                finally
                {
                    if (lookupBuf is not null) ArrayPool<char>.Shared.Return(lookupBuf);
                }
                start = sequence.Length;
                break;
            }

            // 慢速路径：从末尾逐个缩减（复用缓存的 UTF-8 字节）
            int end = sequence.Length;
            Token? foundToken = null;

            while (start < end)
            {
                if (end < sequence.Length)
                {
                    // 使用预计算的 UTF-8 偏移 + 缓存字节，零编码开销
                    int utf8Len = actualPrefix + (utf8Offsets[end] - utf8Offsets[start]);

                    if (_utf8Vocab.TryGetId(utf8Buf.AsSpan(0, utf8Len), out var id))
                    {
                        // 确认命中后才创建 string
                        int keyCharLen = actualPrefix + (end - start);
                        char[]? lookupBuf2 = null;
                        Span<char> lookup2 = keyCharLen <= 256
                            ? stackalloc char[keyCharLen]
                            : (lookupBuf2 = ArrayPool<char>.Shared.Rent(keyCharLen));
                        try
                        {
                            if (actualPrefix > 0)
                                _continuingSubwordPrefix.AsSpan().CopyTo(lookup2);
                            sequence.Slice(start, end - start).CopyTo(lookup2.Slice(actualPrefix));
                            foundToken = new Token(id, new string(lookup2.Slice(0, keyCharLen)), start, end);
                        }
                        finally
                        {
                            if (lookupBuf2 is not null) ArrayPool<char>.Shared.Return(lookupBuf2);
                        }
                        break;
                    }
                }

                int lastCharLen;
                if (end >= 2 && char.IsLowSurrogate(sequence[end - 1]) && char.IsHighSurrogate(sequence[end - 2]))
                    lastCharLen = 2;
                else
                    lastCharLen = 1;
                end -= lastCharLen;
            }

            if (foundToken is null)
            {
                ArrayPool<byte>.Shared.Return(utf8Buf);
                if (utf8Buf != seqUtf8Buf) ArrayPool<byte>.Shared.Return(seqUtf8Buf);
                if (_vocab.TryGetValue(_unkToken!, out var unkId))
                    return [new Token(unkId, _unkToken!, 0, sequence.Length)];
                return [];
            }

            tokens.Add(foundToken.Value);
            start = foundToken.Value.End;
        }

        ArrayPool<byte>.Shared.Return(utf8Buf);
        if (utf8Buf != seqUtf8Buf) ArrayPool<byte>.Shared.Return(seqUtf8Buf);

        return tokens;
    }

    private static int CountRunes(ReadOnlySpan<char> span)
    {
        int count = 0;
        foreach (var _ in span.EnumerateRunes()) count++;
        return count;
    }

    /// <inheritdoc />
    /// <summary>
    /// 轻量级 WordPiece 分词，不分配 string。
    /// 与 Tokenize 相同的贪心最长匹配算法，但只记录 id + offsets。
    /// </summary>
    public List<TokenRef> TokenizeRef(ReadOnlySpan<char> sequence)
    {
        if (sequence.IsEmpty)
            return [];

        bool allAscii = true;
        for (int i = 0; i < sequence.Length; i++)
            if (sequence[i] > 0x7F) { allAscii = false; break; }
        int charLen = allAscii ? sequence.Length : CountRunes(sequence);

        if (charLen > _maxInputCharsPerWord)
        {
            if (_vocab.TryGetValue(_unkToken!, out var unkId))
                return [new TokenRef(unkId, 0, sequence.Length)];
            return [];
        }

        var refs = new List<TokenRef>(sequence.Length / 4 + 1);
        int start = 0;
        int prefixLen = _continuingSubwordPrefix.Length;
        int prefixUtf8Len = prefixLen > 0 ? SysEncoding.UTF8.GetByteCount(_continuingSubwordPrefix) : 0;

        // 一次性编码整个 sequence 并缓存，外循环复用
        int seqUtf8BufSize = SysEncoding.UTF8.GetMaxByteCount(sequence.Length);
        byte[]? seqUtf8Buf = ArrayPool<byte>.Shared.Rent(seqUtf8BufSize);
        int actualSeqUtf8Len = SysEncoding.UTF8.GetBytes(sequence, seqUtf8Buf);

        // UTF-8 查找缓冲区：[prefix | sequence bytes]，每次迭代重新组装
        int totalBufSize = prefixUtf8Len + actualSeqUtf8Len;
        byte[]? utf8Buf = totalBufSize <= seqUtf8BufSize ? seqUtf8Buf : ArrayPool<byte>.Shared.Rent(totalBufSize);

        // 从编码后字节推导每个 char 位置的 UTF-8 字节偏移（避免逐 char GetByteCount）
        // 注意：surrogate pair（high + low）在 UTF-8 中占 4 字节，已在 high surrogate 处计入，
        // low surrogate 不应再贡献字节，否则会导致后续偏移量多算 3 字节。
        Span<int> utf8Offsets = sequence.Length <= 1024 ? stackalloc int[sequence.Length + 1] : new int[sequence.Length + 1];
        utf8Offsets[0] = 0;
        for (int i = 0; i < sequence.Length; i++)
        {
            char c = sequence[i];
            utf8Offsets[i + 1] = utf8Offsets[i] + (c <= 0x7F ? 1 : c <= 0x7FF ? 2 : char.IsHighSurrogate(c) ? 4 : char.IsLowSurrogate(c) ? 0 : 3);
        }

        while (start < sequence.Length)
        {
            int seqLen = sequence.Length - start;
            int actualPrefix = (start > 0 && prefixLen > 0) ? prefixLen : 0;

            // 组装查找字节：[prefix | seqUtf8[start..]]，避免覆盖缓存
            if (actualPrefix > 0)
                SysEncoding.UTF8.GetBytes(_continuingSubwordPrefix.AsSpan(), utf8Buf);
            int remainingSeqUtf8Len = actualSeqUtf8Len - utf8Offsets[start];
            Buffer.BlockCopy(seqUtf8Buf, utf8Offsets[start], utf8Buf, actualPrefix, remainingSeqUtf8Len);

            // 快速路径：整个剩余序列命中
            int totalUtf8Len = actualPrefix + remainingSeqUtf8Len;

            if (_utf8Vocab.TryGetId(utf8Buf.AsSpan(0, totalUtf8Len), out var fullId))
            {
                refs.Add(new TokenRef(fullId, start, sequence.Length));
                start = sequence.Length;
                break;
            }

            // 慢速路径：从末尾逐个缩减（复用缓存的 UTF-8 字节）
            int end = sequence.Length;
            TokenRef? foundRef = null;

            while (start < end)
            {
                if (end < sequence.Length)
                {
                    // 使用预计算的 UTF-8 偏移 + 缓存字节，零编码开销
                    int utf8Len = actualPrefix + (utf8Offsets[end] - utf8Offsets[start]);

                    if (_utf8Vocab.TryGetId(utf8Buf.AsSpan(0, utf8Len), out var id))
                    {
                        foundRef = new TokenRef(id, start, end);
                        break;
                    }
                }

                int lastCharLen;
                if (end >= 2 && char.IsLowSurrogate(sequence[end - 1]) && char.IsHighSurrogate(sequence[end - 2]))
                    lastCharLen = 2;
                else
                    lastCharLen = 1;
                end -= lastCharLen;
            }

            if (foundRef is null)
            {
                ArrayPool<byte>.Shared.Return(utf8Buf);
                if (utf8Buf != seqUtf8Buf) ArrayPool<byte>.Shared.Return(seqUtf8Buf);
                if (_vocab.TryGetValue(_unkToken!, out var unkId))
                    return [new TokenRef(unkId, 0, sequence.Length)];
                return [];
            }

            refs.Add(foundRef.Value);
            start = foundRef.Value.End;
        }

        ArrayPool<byte>.Shared.Return(utf8Buf);
        if (utf8Buf != seqUtf8Buf) ArrayPool<byte>.Shared.Return(seqUtf8Buf);

        return refs;
    }

    /// <summary>
    /// 使用新的词汇表重建模型（供训练器调用）。
    /// 保持原有配置不变。
    /// </summary>
    internal void Rebuild(Dictionary<string, uint> newVocab)
    {
        RebuildVocab(newVocab);
        _utf8Vocab = new Utf8Vocab(newVocab);
    }

    /// <inheritdoc />
    public override IReadOnlyList<string> Save(string folder, string? prefix = null)
    {
        Directory.CreateDirectory(folder);

        var prefixStr = prefix is not null ? $"{prefix}." : "";
        var vocabPath = Path.Combine(folder, $"{prefixStr}vocab.txt");

        using (var writer = new StreamWriter(vocabPath, false, global::System.Text.Encoding.UTF8))
        {
            foreach (var kvp in _vocab.OrderBy(k => k.Value))
            {
                writer.WriteLine(kvp.Key);
            }
        }

        return new List<string> { vocabPath };
    }

    /// <inheritdoc />

    /// <inheritdoc />
    public override string? ContinuingSubwordPrefix => _continuingSubwordPrefix;

    /// <summary>
    /// 从 vocab.txt 文件加载 WordPiece 模型（每行一个 token）。
    /// </summary>
    public static WordPieceModel Load(
        string vocabPath,
        string continuingSubwordPrefix = "##",
        string unkToken = "[UNK]",
        int maxInputCharsPerWord = 100)
    {
        var vocab = new Dictionary<string, uint>(StringComparer.Ordinal);
        uint id = 0;

        foreach (var line in File.ReadLines(vocabPath))
        {
            var token = line.Trim();
            if (!string.IsNullOrEmpty(token))
            {
                vocab[token] = id++;
            }
        }

        return new WordPieceModel(vocab, continuingSubwordPrefix, unkToken, maxInputCharsPerWord);
    }

    /// <summary>
    /// 从 BPE 模型转换为 WordPiece 模型（与 Rust WordPiece::from_bpe 一致）。
    /// 复制词表、unk_token 和 continuing_subword_prefix。
    /// 如果 BPE 词表中不包含 unk_token，则自动补充（与 Rust build_wordpiece_from_bpe 行为一致）。
    /// </summary>
    public static WordPieceModel FromBpe(BpeModel bpe)
    {
        var vocab = new Dictionary<string, uint>(bpe.Vocab, StringComparer.Ordinal);
        var unkToken = bpe.UnkToken ?? "[UNK]";

        // BPE 训练可能不保留初始词表中的 [UNK]，需确保其存在（Rust 端显式检查并补充）
        if (!vocab.ContainsKey(unkToken))
            vocab[unkToken] = (uint)vocab.Count;

        return new WordPieceModel(vocab, bpe.ContinuingSubwordPrefix ?? "##", unkToken, 100);
    }

    /// <summary>
    /// 构建器，用于构建 <see cref="WordPieceModel"/> 实例的工厂方法。
    /// </summary>
    public sealed class WordPieceBuilder
    {
        private Dictionary<string, uint>? _vocab;
        private string _continuingSubwordPrefix = "##";
        private string _unkToken = "[UNK]";
        private int _maxInputCharsPerWord = 100;

        /// <summary>
        /// 设置词表。
        /// </summary>
        public WordPieceBuilder SetVocab(Dictionary<string, uint> vocab)
        {
            _vocab = vocab;
            return this;
        }

        /// <summary>
        /// 设置连续子词前缀。
        /// </summary>
        public WordPieceBuilder SetContinuingSubwordPrefix(string prefix)
        {
            _continuingSubwordPrefix = prefix;
            return this;
        }

        /// <summary>
        /// 设置未知 token。
        /// </summary>
        public WordPieceBuilder SetUnkToken(string unkToken)
        {
            _unkToken = unkToken;
            return this;
        }

        /// <summary>
        /// 设置每个词的最大输入字符数。
        /// </summary>
        public WordPieceBuilder SetMaxInputCharsPerWord(int maxInputCharsPerWord)
        {
            _maxInputCharsPerWord = maxInputCharsPerWord;
            return this;
        }

        /// <summary>
        /// 构建 <see cref="WordPieceModel"/> 实例。
        /// </summary>
        public WordPieceModel Build()
        {
            var vocab = _vocab ?? throw new InvalidOperationException("Vocabulary must be set before building.");
            return new WordPieceModel(vocab, _continuingSubwordPrefix, _unkToken, _maxInputCharsPerWord);
        }
    }
}
