using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Models;

/// <summary>
/// 模型包装器，委托给任意 <see cref="IModel"/> 实现。
/// 为不同模型类型提供统一接口。
/// </summary>
public sealed class ModelWrapper : IModel
{
    private readonly IModel _inner;
    private readonly ModelType _type;

    /// <summary>获取包装的模型类型。</summary>
    public ModelType Type => _type;

    /// <summary>获取内部模型。</summary>
    public IModel Inner => _inner;

    /// <summary>
    /// 创建新的 model wrapper。
    /// </summary>
    /// <param name="inner">要包装的模型。</param>
    /// <param name="type">模型类型。</param>
    public ModelWrapper(IModel inner, ModelType type)
    {
        _inner = inner ?? throw new ArgumentNullException(nameof(inner));
        _type = type;
    }

    /// <summary>
    /// 创建 BPE 模型的包装器。
    /// </summary>
    public static ModelWrapper CreateBPE(IModel model) =>
        new(model, ModelType.BPE);

    /// <summary>
    /// 创建 WordPiece 模型的包装器。
    /// </summary>
    public static ModelWrapper CreateWordPiece(IModel model) =>
        new(model, ModelType.WordPiece);

    /// <summary>
    /// 创建 WordLevel 模型的包装器。
    /// </summary>
    public static ModelWrapper CreateWordLevel(IModel model) =>
        new(model, ModelType.WordLevel);

    /// <summary>
    /// 创建 Unigram 模型的包装器。
    /// </summary>
    public static ModelWrapper CreateUnigram(IModel model) =>
        new(model, ModelType.Unigram);

    /// <summary>
    /// 自动检测模型类型并创建包装器。
    /// 委托给源代码生成器的 <c>GetModelTypeName</c>，避免重复类型检测逻辑。
    /// </summary>
    /// <param name="model">要包装的模型。</param>
    /// <returns>带有自动检测类型的包装模型。</returns>
    public static ModelWrapper AutoDetect(IModel model)
    {
        var typeName = global::HuggingFace.Tokenizers.Generated.TokenizerComponentFactory.GetModelTypeName(model);
        var type = typeName switch
        {
            "BPE" => ModelType.BPE,
            "WordPiece" => ModelType.WordPiece,
            "WordLevel" => ModelType.WordLevel,
            "Unigram" => ModelType.Unigram,
            _ => throw new ArgumentException($"Unknown model type: {typeName}")
        };

        return new ModelWrapper(model, type);
    }

    /// <inheritdoc />
    public List<Token> Tokenize(ReadOnlySpan<char> sequence) =>
        _inner.Tokenize(sequence);

    /// <inheritdoc />
    public uint? TokenToId(string token) =>
        _inner.TokenToId(token);

    /// <inheritdoc />
    public string? IdToToken(uint id) =>
        _inner.IdToToken(id);

    /// <inheritdoc />
    public IReadOnlyDictionary<string, uint> GetVocab() =>
        _inner.GetVocab();

    /// <inheritdoc />
    public int GetVocabSize() =>
        _inner.GetVocabSize();

    /// <inheritdoc />
    public IReadOnlyList<string> Save(string folder, string? prefix = null) =>
        _inner.Save(folder, prefix);

    /// <summary>
    /// 返回包装器的字符串表示。
    /// </summary>
    public override string ToString() =>
        $"ModelWrapper({Type}, {_inner.GetType().Name})";
}
