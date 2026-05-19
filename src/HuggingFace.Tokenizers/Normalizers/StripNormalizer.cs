using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Normalizers;

/// <summary>
/// 去除文本的首尾空白。
/// </summary>
[TokenizerComponent("Strip")]
public sealed class StripNormalizer : INormalizer
{
    private readonly bool _stripLeft;
    private readonly bool _stripRight;

    /// <summary>是否去除前导空白。</summary>
    public bool StripLeft => _stripLeft;

    /// <summary>是否去除尾部空白。</summary>
    public bool StripRight => _stripRight;

    /// <summary>
    /// 创建新的 <see cref="StripNormalizer"/>.
    /// </summary>
    /// <param name="stripLeft">是否去除前导空白。</param>
    /// <param name="stripRight">是否去除尾部空白。</param>
    public StripNormalizer(bool stripLeft = true, bool stripRight = true)
    {
        _stripLeft = stripLeft;
        _stripRight = stripRight;
    }

    /// <summary>
    /// 去除标准化字符串的首尾空白。
    /// </summary>
    /// <param name="normalized">要就地标准化的字符串。</param>
    public void Normalize(NormalizedString normalized)
    {
        normalized.Strip(_stripLeft, _stripRight);
    }
}
