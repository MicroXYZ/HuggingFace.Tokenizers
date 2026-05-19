using System.Diagnostics.CodeAnalysis;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Processors;

/// <summary>
/// 包装器，将所有 <see cref="IPostProcessor"/> 调用委托给内部处理器。
/// 可作为基类，用于在现有处理器基础上添加行为（日志、指标、缓存）。
/// </summary>
[DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicConstructors)]
[TokenizerComponent("PostProcessorWrapper")]
public class PostProcessorWrapper : IPostProcessor
{
    /// <summary>
    /// 此包装器委托的内部处理器。
    /// </summary>
    protected IPostProcessor Inner { get; }

    /// <summary>
    /// 创建新的包装器，包装给定的处理器。
    /// </summary>
    /// <param name="inner">要委托的处理器。</param>
    /// <exception cref="ArgumentNullException">当 <paramref name="inner"/> 为 null 时抛出。</exception>
    public PostProcessorWrapper(IPostProcessor inner)
    {
        ArgumentNullException.ThrowIfNull(inner);
        Inner = inner;
    }

    /// <inheritdoc />
    public virtual int AddedTokens(bool isPair) => Inner.AddedTokens(isPair);

    /// <inheritdoc />
    public virtual Encoding Process(Encoding encoding, Encoding? pairEncoding, bool addSpecialTokens)
        => Inner.Process(encoding, pairEncoding, addSpecialTokens);

    /// <inheritdoc />
    public virtual IReadOnlyList<Encoding> ProcessEncodings(IReadOnlyList<Encoding> encodings, bool addSpecialTokens)
        => Inner.ProcessEncodings(encodings, addSpecialTokens);
}
