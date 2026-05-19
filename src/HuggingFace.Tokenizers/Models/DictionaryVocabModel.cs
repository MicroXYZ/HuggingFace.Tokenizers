using System.Collections.Frozen;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Models;

/// <summary>
/// 基于 Dictionary 的分词模型基类。
/// 封装词表管理（vocab + vocabR）和通用查询方法，
/// 子类只需实现 Tokenize、Save。
///
/// 内部使用 FrozenDictionary 进行 O(1) 查询（完美哈希），
/// RebuildVocab 时原子替换引用。
///
/// BpeModel、WordPieceModel、WordLevelModel 共享相同的词表结构，
/// 通过此基类消除重复代码。
/// UnigramModel 因词表结构不同（带 log prob），不继承此类。
/// </summary>
public abstract class DictionaryVocabModel : IModel
{
    // 查询用 FrozenDictionary（构建后冻结，完美哈希，2-5x 快于 Dictionary）
    private protected FrozenDictionary<string, uint> _vocab;
    private protected FrozenDictionary<uint, string> _vocabR;

    // 保留 mutable 副本用于 Save/OrderBy/Rebuild
    private protected Dictionary<string, uint> _vocabMutable;

    private protected readonly object _vocabLock = new();
    private protected string? _unkToken;

    /// <summary>
    /// 获取词表（token → ID 映射）。
    /// </summary>
    public IReadOnlyDictionary<string, uint> Vocab => _vocab;

    /// <summary>
    /// 获取反向词表（ID → token 映射）。
    /// </summary>
    public IReadOnlyDictionary<uint, string> VocabR => _vocabR;

    /// <summary>
    /// 获取未知 token。
    /// </summary>
    public string? UnkToken => _unkToken;

    /// <summary>
    /// 初始化词表基类。
    /// </summary>
    protected DictionaryVocabModel(Dictionary<string, uint> vocab, string? unkToken)
    {
        _vocabMutable = vocab;
        _unkToken = unkToken;

        var vocabR = new Dictionary<uint, string>(vocab.Count);
        foreach (var kvp in vocab)
            vocabR[kvp.Value] = kvp.Key;

        // 冻结为 FrozenDictionary（查询用）
        _vocab = vocab.ToFrozenDictionary(StringComparer.Ordinal);
        _vocabR = vocabR.ToFrozenDictionary();
    }

    /// <inheritdoc />
    public uint? TokenToId(string token) =>
        _vocab.TryGetValue(token, out var id) ? id : null;

    /// <inheritdoc />
    public string? IdToToken(uint id) =>
        _vocabR.TryGetValue(id, out var token) ? token : null;

    /// <inheritdoc />
    public IReadOnlyDictionary<string, uint> GetVocab() => _vocab;

    /// <inheritdoc />
    public int GetVocabSize() => _vocab.Count;

    /// <inheritdoc />
    public abstract List<Token> Tokenize(ReadOnlySpan<char> sequence);

    /// <inheritdoc />
    public abstract IReadOnlyList<string> Save(string folder, string? prefix = null);

    /// <inheritdoc />
    public virtual IEnumerable<string> GetMerges() => [];

    /// <inheritdoc />
    public virtual string? ContinuingSubwordPrefix => null;

    /// <inheritdoc />
    public virtual string? EndOfWordSuffix => null;

    /// <summary>
    /// 用新词表重建 vocab 和 vocabR。
    /// BpeModel、WordPieceModel、WordLevelModel 共用此逻辑。
    /// 原子替换 FrozenDictionary 引用（Volatile.Write）。
    /// </summary>
    protected void RebuildVocab(Dictionary<string, uint> newVocab)
    {
        lock (_vocabLock)
        {
            _vocabMutable = newVocab;

            var newVocabR = new Dictionary<uint, string>(newVocab.Count);
            foreach (var kvp in newVocab)
                newVocabR[kvp.Value] = kvp.Key;

            // 原子替换 FrozenDictionary
            Volatile.Write(ref _vocab, newVocab.ToFrozenDictionary(StringComparer.Ordinal));
            Volatile.Write(ref _vocabR, newVocabR.ToFrozenDictionary());
        }
    }
}
