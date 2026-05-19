namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 分词模型（BPE、WordPiece、WordLevel、Unigram）。
/// 分词管道的组成部分。
/// </summary>
public interface IModel
{
    /// <summary>
    /// 将给定序列分词为带有 ID 和偏移的 token 列表。
    /// </summary>
    /// <param name="sequence">要分词的文本序列（ReadOnlySpan，string 隐式转换）。</param>
    /// <returns>分词结果。</returns>
    List<Token> Tokenize(ReadOnlySpan<char> sequence);

    /// <summary>
    /// 将给定序列分词为轻量级 TokenRef 列表（不持有 string）。
    /// 默认实现：调用 Tokenize 后转换为 TokenRef。
    /// 模型可重写此方法以避免 string 分配。
    /// </summary>
    /// <param name="sequence">要分词的文本序列。</param>
    /// <returns>轻量级分词结果。</returns>
    List<TokenRef> TokenizeRef(ReadOnlySpan<char> sequence)
    {
        var tokens = Tokenize(sequence);
        var refs = new List<TokenRef>(tokens.Count);
        for (int i = 0; i < tokens.Count; i++)
            refs.Add(new TokenRef(tokens[i].Id, tokens[i].Start, tokens[i].End));
        return refs;
    }

    /// <summary>
    /// 将给定序列分词为带有 ID 和偏移的 token 列表（Memory 重载）。
    /// 默认实现：直接使用 Span 驱动。
    /// </summary>
    /// <param name="sequence">要分词的文本序列。</param>
    /// <returns>分词结果。</returns>
    List<Token> Tokenize(ReadOnlyMemory<char> sequence)
        => Tokenize(sequence.Span);

    /// <summary>
    /// 获取 token 字符串对应的 ID，未找到返回 null。
    /// </summary>
    /// <param name="token">token 字符串。</param>
    /// <returns>token ID 或 null。</returns>
    uint? TokenToId(string token);

    /// <summary>
    /// 获取 ID 对应的 token 字符串，未找到返回 null。
    /// </summary>
    /// <param name="id">token ID。</param>
    /// <returns>token 字符串或 null。</returns>
    string? IdToToken(uint id);

    /// <summary>
    /// 获取完整词表（token → ID 映射）。
    /// </summary>
    IReadOnlyDictionary<string, uint> GetVocab();

    /// <summary>
    /// 获取词表大小。
    /// </summary>
    int GetVocabSize();

    /// <summary>
    /// 将模型保存到指定目录。
    /// </summary>
    /// <param name="folder">保存目录。</param>
    /// <param name="prefix">文件名前缀。</param>
    /// <returns>保存的文件路径列表。</returns>
    IReadOnlyList<string> Save(string folder, string? prefix = null);

    /// <summary>
    /// 获取合并对，格式为 "first second" 字符串。
    /// 不使用合并的模型（WordPiece、WordLevel、Unigram）返回空集合。
    /// </summary>
    IEnumerable<string> GetMerges() => [];

    /// <summary>
    /// 获取连续子词前缀（如 WordPiece 的 "##"、GPT-2 BPE 的 "Ġ"）。
    /// 不使用此特性的模型返回 null。
    /// </summary>
    string? ContinuingSubwordPrefix => null;

    /// <summary>
    /// 获取词尾后缀（如某些 BPE 模型的 "</w>"）。
    /// 不使用此特性的模型返回 null。
    /// </summary>
    string? EndOfWordSuffix => null;
}
