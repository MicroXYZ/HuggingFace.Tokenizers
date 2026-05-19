using HuggingFace.Tokenizers.Internal;

namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// PreTokenizedString 中的拆分 — 包含一个 NormalizedString 和可选的 token 列表。
/// </summary>
public sealed class Split
{
    /// <summary>此拆分部分的标准化字符串。</summary>
    public NormalizedString Normalized { get; }

    /// <summary>此拆分部分的 token 列表（分词后填充）。</summary>
    public List<Token>? Tokens { get; internal set; }

    /// <summary>此拆分部分的轻量级 TokenRef 列表（分词后填充，不持有 string）。</summary>
    public List<TokenRef>? TokenRefs { get; internal set; }

    /// <summary>
    /// 初始化 <see cref="Split"/> 的新实例。
    /// </summary>
    /// <param name="normalized">标准化字符串。</param>
    /// <param name="tokens">可选的 token 列表。</param>
    public Split(NormalizedString normalized, List<Token>? tokens = null)
    {
        Normalized = normalized;
        Tokens = tokens;
    }
}
