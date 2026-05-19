namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// Metaspace 前缀方案枚举。
/// 控制何时在 token 前插入替换字符。
/// 由 MetaspaceDecoder 和 MetaspacePreTokenizer 共用。
/// </summary>
public enum PrependScheme
{
    /// <summary>总是在第一个 token 前插入替换字符。</summary>
    Always,

    /// <summary>从不插入替换字符。</summary>
    Never,

    /// <summary>仅在第一个 token 不以它开头时插入替换字符。</summary>
    First
}
