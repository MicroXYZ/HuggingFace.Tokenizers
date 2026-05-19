using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.PreTokenizers;

/// <summary>
/// A wrapper that delegates to any <see cref="IPreTokenizer"/> 实例。
/// 适用于序列化/反序列化或动态交换实现。
/// </summary>
[TokenizerComponent("PreTokenizerWrapper")]
public sealed class PreTokenizerWrapper : IPreTokenizer
{
    /// <summary>
    /// 被包装的预分词器实例。
    /// </summary>
    public IPreTokenizer Inner { get; }

    /// <summary>
    /// 初始化新的 <see cref="PreTokenizerWrapper"/>.
    /// </summary>
    /// <param name="inner">The pre-tokenizer to wrap.</param>
    /// <exception cref="ArgumentNullException"><paramref name="inner"/> is null.</exception>
    public PreTokenizerWrapper(IPreTokenizer inner)
    {
        ArgumentNullException.ThrowIfNull(inner);
        Inner = inner;
    }

    /// <inheritdoc />
    public void PreTokenize(PreTokenizedString pretokenized)
    {
        Inner.PreTokenize(pretokenized);
    }
}
