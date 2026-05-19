using System.Diagnostics.CodeAnalysis;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Processors;

/// <summary>
/// 链式组合多个 <see cref="IPostProcessor"/> 实例的后处理器，
/// 按顺序依次应用。前一个处理器的输出作为下一个的输入。
/// </summary>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
[TokenizerComponent("Sequence")]
public sealed class SequenceProcessor : IPostProcessor
{
    private readonly IReadOnlyList<IPostProcessor> _processors;

    /// <summary>此序列中的后处理器列表。</summary>
    public IReadOnlyList<IPostProcessor> Processors => _processors;

    /// <summary>
    /// 从有序处理器列表创建序列处理器。
    /// </summary>
    /// <param name="processors">要链式组合的处理器，按执行顺序排列。</param>
    /// <exception cref="ArgumentException">未提供任何处理器时抛出。</exception>
    public SequenceProcessor(IReadOnlyList<IPostProcessor> processors)
    {
        ArgumentNullException.ThrowIfNull(processors);

        if (processors.Count == 0)
            throw new ArgumentException("At least one processor is required.", nameof(processors));

        _processors = processors;
    }

    /// <summary>
    /// 从参数数组创建序列处理器。
    /// </summary>
    public SequenceProcessor(params IPostProcessor[] processors)
        : this((IReadOnlyList<IPostProcessor>)processors)
    {
    }

    /// <inheritdoc />
    public int AddedTokens(bool isPair)
    {
        int total = 0;
        foreach (var processor in _processors)
            total += processor.AddedTokens(isPair);
        return total;
    }

    /// <inheritdoc />
    public Encoding Process(Encoding encoding, Encoding? pairEncoding, bool addSpecialTokens)
    {
        var current = encoding;
        Encoding? currentPair = pairEncoding;

        foreach (var processor in _processors)
        {
            var result = processor.Process(current, currentPair, addSpecialTokens);
            // 第一个处理器之后，配对编码已被消费/合并
            current = result;
            currentPair = null;
        }

        return current;
    }

    /// <inheritdoc />
    public IReadOnlyList<Encoding> ProcessEncodings(IReadOnlyList<Encoding> encodings, bool addSpecialTokens)
    {
        var current = encodings;

        foreach (var processor in _processors)
            current = processor.ProcessEncodings(current, addSpecialTokens);

        return current;
    }
}
