namespace HuggingFace.Tokenizers.Abstractions;

/// <summary>
/// 表示已添加到分词器词表中的 token。
/// 特殊 token 如 [CLS]、[SEP]、&lt;s&gt;、&lt;/s&gt; 等。
/// </summary>
public sealed class AddedToken
{
    /// <summary>token 的词汇表 ID。为 null 时由 AddedVocabulary 自动分配。</summary>
    public uint? Id { get; }

    /// <summary>token 内容。</summary>
    public string Content { get; }

    /// <summary>是否为特殊 token。</summary>
    public bool IsSpecial { get; }

    /// <summary>匹配时是否去除左侧空白。</summary>
    public bool LStrip { get; }

    /// <summary>匹配时是否去除右侧空白。</summary>
    public bool RStrip { get; }

    /// <summary>是否仅匹配完整单词。</summary>
    public bool SingleWord { get; }

    /// <summary>
    /// 匹配前是否对 token 进行标准化。
    /// 为 true 时，输入文本在尝试匹配此 token 前会先进行标准化。
    /// </summary>
    public bool Normalized { get; }

    /// <summary>
    /// 创建 AddedToken 实例。
    /// </summary>
    /// <param name="content">token 内容。</param>
    /// <param name="id">token 的词汇表 ID。为 null 时由 AddedVocabulary 自动分配。</param>
    /// <param name="isSpecial">是否为特殊 token。</param>
    /// <param name="lStrip">匹配时是否去除左侧空白。</param>
    /// <param name="rStrip">匹配时是否去除右侧空白。</param>
    /// <param name="singleWord">是否仅匹配完整单词。</param>
    /// <param name="normalized">匹配前是否进行标准化。</param>
    public AddedToken(
        string content,
        uint? id = null,
        bool isSpecial = false,
        bool lStrip = false,
        bool rStrip = false,
        bool singleWord = false,
        bool normalized = false)
    {
        Content = content;
        Id = id;
        IsSpecial = isSpecial;
        LStrip = lStrip;
        RStrip = rStrip;
        SingleWord = singleWord;
        Normalized = normalized;
    }

    /// <summary>
    /// 从内容创建 AddedToken，指定是否为特殊 token。
    /// 特殊 token 默认不进行标准化（normalized = !special）。
    /// </summary>
    /// <param name="content">token 内容。</param>
    /// <param name="isSpecial">是否为特殊 token。</param>
    public static AddedToken From(string content, bool isSpecial = false)
        => new(content, isSpecial: isSpecial, normalized: !isSpecial);

    /// <summary>
    /// 旧版工厂方法。建议使用 <see cref="From"/> 代替。
    /// </summary>
    public static AddedToken FromString(string content, bool isSpecial = false)
        => new(content, isSpecial: isSpecial);

    public override string ToString() => Content;
    public override int GetHashCode() => Content.GetHashCode();
    public override bool Equals(object? obj)
        => obj is AddedToken other && Content == other.Content;
}
