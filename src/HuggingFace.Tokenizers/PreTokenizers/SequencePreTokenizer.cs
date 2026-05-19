using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.PreTokenizers;

/// <summary>
/// 链式组合多个 <see cref="IPreTokenizer"/> 实例的预分词器，
/// 按顺序依次应用。
/// </summary>
[TokenizerComponent("Sequence")]
public sealed class SequencePreTokenizer : IPreTokenizer
{
    private readonly IReadOnlyList<IPreTokenizer> _preTokenizers;

    /// <summary>
    /// 初始化新的 <see cref="SequencePreTokenizer"/>.
    /// </summary>
    /// <param name="preTokenizers">按顺序应用的预分词器序列。</param>
    /// <exception cref="ArgumentNullException"><paramref name="preTokenizers"/> 为 null。</exception>
    /// <exception cref="ArgumentException"><paramref name="preTokenizers"/> 为空。</exception>
    public SequencePreTokenizer(IEnumerable<IPreTokenizer> preTokenizers)
    {
        ArgumentNullException.ThrowIfNull(preTokenizers);
        _preTokenizers = preTokenizers.ToList();

        if (_preTokenizers.Count == 0)
            throw new ArgumentException("At least one pre-tokenizer is required.", nameof(preTokenizers));
    }

    /// <summary>
    /// 使用数组初始化 <see cref="SequencePreTokenizer"/> 的新实例。
    /// </summary>
    /// <param name="preTokenizers">按顺序应用的预分词器。</param>
    public SequencePreTokenizer(params IPreTokenizer[] preTokenizers)
        : this((IEnumerable<IPreTokenizer>)preTokenizers)
    {
    }

    /// <summary>
    /// 获取此序列中的预分词器列表。
    /// </summary>
    public IReadOnlyList<IPreTokenizer> PreTokenizers => _preTokenizers;

    /// <inheritdoc />
    public void PreTokenize(PreTokenizedString pretokenized)
    {
        foreach (var preTokenizer in _preTokenizers)
        {
            preTokenizer.PreTokenize(pretokenized);
        }
    }
}
