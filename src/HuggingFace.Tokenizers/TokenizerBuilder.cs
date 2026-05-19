namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 流式构建器，用于创建包含所有管道组件的 <see cref="Tokenizer"/>。
/// </summary>
public sealed class TokenizerBuilder
{
    private IModel? _model;
    private INormalizer? _normalizer;
    private IPreTokenizer? _preTokenizer;
    private IPostProcessor? _postProcessor;
    private IDecoder? _decoder;
    private TruncationParams? _truncation;
    private PaddingParams? _padding;
    private readonly List<AddedToken> _addedTokens = [];

    /// <summary>设置分词模型。</summary>
    public TokenizerBuilder WithModel(IModel model)
    {
        _model = model;
        return this;
    }

    /// <summary>设置标准化器。</summary>
    public TokenizerBuilder WithNormalizer(INormalizer? normalizer)
    {
        _normalizer = normalizer;
        return this;
    }

    /// <summary>设置预分词器。</summary>
    public TokenizerBuilder WithPreTokenizer(IPreTokenizer? preTokenizer)
    {
        _preTokenizer = preTokenizer;
        return this;
    }

    /// <summary>设置后处理器。</summary>
    public TokenizerBuilder WithPostProcessor(IPostProcessor? postProcessor)
    {
        _postProcessor = postProcessor;
        return this;
    }

    /// <summary>设置解码器。</summary>
    public TokenizerBuilder WithDecoder(IDecoder? decoder)
    {
        _decoder = decoder;
        return this;
    }

    /// <summary>设置截断参数。</summary>
    public TokenizerBuilder WithTruncation(TruncationParams truncation)
    {
        _truncation = truncation;
        return this;
    }

    /// <summary>设置填充参数。</summary>
    public TokenizerBuilder WithPadding(PaddingParams padding)
    {
        _padding = padding;
        return this;
    }

    /// <summary>添加单个 token。</summary>
    public TokenizerBuilder AddToken(AddedToken token)
    {
        _addedTokens.Add(token);
        return this;
    }

    /// <summary>批量添加 token。</summary>
    public TokenizerBuilder AddTokens(IEnumerable<AddedToken> tokens)
    {
        _addedTokens.AddRange(tokens);
        return this;
    }

    /// <summary>
    /// 构建分词器。如果未配置模型则抛出异常。
    /// </summary>
    /// <returns>构建的分词器。</returns>
    /// <exception cref="InvalidOperationException">未配置模型时抛出。</exception>
    public Tokenizer Build()
    {
        if (_model is null)
            throw new InvalidOperationException("必须配置模型。");

        var tokenizer = new Tokenizer(_model)
        {
            Normalizer = _normalizer,
            PreTokenizer = _preTokenizer,
            PostProcessor = _postProcessor,
            Decoder = _decoder,
            Truncation = _truncation,
            Padding = _padding
        };

        if (_addedTokens.Count > 0)
            tokenizer.AddTokens(_addedTokens);

        return tokenizer;
    }
}
