namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 表示分词过程中模型产生的单个 token。
/// 使用 readonly record struct（值类型），消除堆分配。
/// </summary>
/// <param name="Id">token ID。</param>
/// <param name="Value">token 字符串值。</param>
/// <param name="Start">在原始文本中的起始位置（字节或字符偏移，取决于上下文）。</param>
/// <param name="End">在原始文本中的结束位置。</param>
public readonly record struct Token(uint Id, string Value, int Start, int End)
{
    /// <summary>token 的偏移范围 (Start, End)。</summary>
    public (int Start, int End) Offsets => (Start, End);

    public override string ToString() => $"Token({Id}, \"{Value}\", {Start}..{End})";
}

/// <summary>
/// 轻量级 token 引用，不持有 string Value。
/// 通过原始文本 + 偏移按需获取值，消除分词过程中的 string 分配。
/// 适用于内部编码路径（EncodeFast、批量编码等），公共 API 仍使用 Token。
/// </summary>
/// <param name="Id">token ID。</param>
/// <param name="Start">在原始文本中的起始位置（字节或字符偏移，取决于上下文）。</param>
/// <param name="End">在原始文本中的结束位置。</param>
public readonly struct TokenRef(uint Id, int Start, int End) : IEquatable<TokenRef>
{
    /// <summary>token ID。</summary>
    public uint Id { get; } = Id;

    /// <summary>在原始文本中的起始位置。</summary>
    public int Start { get; } = Start;

    /// <summary>在原始文本中的结束位置。</summary>
    public int End { get; } = End;

    /// <summary>token 的偏移范围 (Start, End)。</summary>
    public (int Start, int End) Offsets => (Start, End);

    /// <summary>从原始文本中获取此 token 的字符串值。</summary>
    /// <param name="source">原始文本。</param>
    /// <returns>token 的字符串值。</returns>
    public string GetValue(ReadOnlySpan<char> source) => source.Slice(Start, End - Start).ToString();

    /// <summary>从原始文本中获取此 token 的 Span。</summary>
    /// <param name="source">原始文本。</param>
    /// <returns>token 的字符范围。</returns>
    public ReadOnlySpan<char> GetSpan(ReadOnlySpan<char> source) => source.Slice(Start, End - Start);

    /// <summary>转换为完整的 Token（需要原始文本）。</summary>
    /// <param name="source">原始文本。</param>
    /// <returns>包含 string Value 的 Token。</returns>
    public Token ToToken(ReadOnlySpan<char> source) => new(Id, GetValue(source), Start, End);

    public bool Equals(TokenRef other) => Id == other.Id && Start == other.Start && End == other.End;
    public override bool Equals(object? obj) => obj is TokenRef other && Equals(other);
    public override int GetHashCode() => HashCode.Combine(Id, Start, End);
    public override string ToString() => $"TokenRef({Id}, {Start}..{End})";

    public static bool operator ==(TokenRef left, TokenRef right) => left.Equals(right);
    public static bool operator !=(TokenRef left, TokenRef right) => !left.Equals(right);
}
