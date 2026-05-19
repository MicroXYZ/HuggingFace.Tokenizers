using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// PrependScheme 解析和 add_prefix_space 兼容逻辑。
/// 从 PreTokenizerResolver 和 DecoderResolver 提取的共享逻辑。
/// </summary>
internal static class PrependSchemeHelper
{
    /// <summary>
    /// 解析 PrependScheme 字符串。
    /// </summary>
    public static PrependScheme Parse(string value) => value.ToLowerInvariant() switch
    {
        "always" => PrependScheme.Always,
        "first" => PrependScheme.First,
        "never" => PrependScheme.Never,
        _ => throw new ArgumentException($"Unsupported prepend scheme: '{value}'.")
    };

    /// <summary>
    /// 解析 prepend_scheme，处理 add_prefix_space 兼容逻辑。
    /// 与 Rust Metaspace 实现一致：
    /// - 有 prepend_scheme 时直接使用
    /// - 仅有 add_prefix_space 时推导（true → Always，false → Never）
    /// - 都没有时默认 Always
    /// - add_prefix_space=false 且 prepend_scheme != Never 时抛出异常
    /// </summary>
    /// <param name="prependSchemeStr">JSON 中的 prepend_scheme 值，可为 null。</param>
    /// <param name="addPrefixSpace">JSON 中的 add_prefix_space 值，可为 null。</param>
    /// <returns>解析后的 PrependScheme。</returns>
    public static PrependScheme ResolveWithCompatibility(string? prependSchemeStr, bool? addPrefixSpace)
    {
        if (prependSchemeStr is not null)
        {
            var scheme = Parse(prependSchemeStr);
            // Rust: add_prefix_space=false 且 prepend_scheme != Never → 报错
            if (addPrefixSpace == false && scheme != PrependScheme.Never)
                throw new ArgumentException(
                    "add_prefix_space=false does not match declared prepend_scheme (must be 'never').");
            return scheme;
        }

        if (addPrefixSpace is not null)
            return addPrefixSpace.Value ? PrependScheme.Always : PrependScheme.Never;

        return PrependScheme.Always; // Rust 默认值
    }
}
