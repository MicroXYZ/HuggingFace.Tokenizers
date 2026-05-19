using System.Globalization;
using System.Text;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.PreTokenizers;

/// <summary>
/// 按 Unicode 脚本边界拆分文本的预分词器。
/// 属于不同 Unicode 脚本的字符被分离到不同的拆分中。
/// 常见的脚本转换（如 Latin → Cyrillic）会导致拆分。
/// 匹配 Rust 的 UnicodeScripts 实现：
/// - Hiragana 和 Katakana 映射为 CJK（fixed_script）
/// - U+30FC (ー) 映射为 CJK
/// - 扩展硬编码范围覆盖更多 Unicode 脚本
/// </summary>
[TokenizerComponent("UnicodeScripts")]
public sealed class UnicodeScriptsPreTokenizer : IPreTokenizer
{
    /// <summary>
    /// 单例实例，方便使用。
    /// </summary>
    public static readonly UnicodeScriptsPreTokenizer Instance = new();

    /// <inheritdoc />
    public void PreTokenize(PreTokenizedString pretokenized)
    {
        pretokenized.Split((_, normalized) =>
        {
            var text = normalized.GetSpan();
            if (text.IsEmpty)
                return Enumerable.Empty<NormalizedString>();

            var parts = new List<NormalizedString>();

            // 使用 Rune 遍历确保正确处理补充平面字符
            var runes = new List<(Rune Rune, int CharStart, int CharLen)>();
            int charIdx = 0;
            foreach (var rune in text.EnumerateRunes())
            {
                int len = rune.Utf16SequenceLength;
                runes.Add((rune, charIdx, len));
                charIdx += len;
            }

            if (runes.Count == 0)
                return Enumerable.Empty<NormalizedString>();

            int start = 0;
            var currentScript = GetScriptCategory(runes[0].Rune);

            for (int i = 1; i < runes.Count; i++)
            {
                var charScript = GetScriptCategory(runes[i].Rune);

                // Split when script changes (unless both are common/unknown)
                if (charScript != currentScript &&
                    !(currentScript == ScriptCategory.Common && charScript == ScriptCategory.Common))
                {
                    // 如果 current is Common, extend to the non-common script
                    if (currentScript == ScriptCategory.Common)
                    {
                        currentScript = charScript;
                        continue;
                    }

                    // 如果 new char is Common (punctuation, digits, etc.), keep with current script
                    if (charScript == ScriptCategory.Common)
                        continue;

                    // Actual script boundary
                    int sliceStart = runes[start].CharStart;
                    int sliceEnd = runes[i].CharStart;
                    if (sliceEnd > sliceStart)
                        parts.Add(normalized.Slice(sliceStart, sliceEnd - sliceStart));
                    start = i;
                    currentScript = charScript;
                }
            }

            // 最后一段
            int finalStart = runes[start].CharStart;
            if (finalStart < text.Length)
                parts.Add(normalized.Slice(finalStart, text.Length - finalStart));

            return parts;
        });
    }

    /// <summary>
    /// 将 Rune 分类到宽泛的脚本类别。
    /// 使用 int code point 确保正确处理补充平面字符。
    /// 匹配 Rust 的 fixed_script 逻辑：Hiragana/Katakana → CJK。
    /// </summary>
    private static ScriptCategory GetScriptCategory(Rune rune)
    {
        var cp = rune.Value;

        // 基本多文种平面的通用类别检查（使用 char 版本）
        if (cp <= 0xFFFF)
        {
            char c = (char)cp;
            var category = CharUnicodeInfo.GetUnicodeCategory(c);

            // Digits, punctuation, symbols, separators → Common
            if (category is UnicodeCategory.DecimalDigitNumber or
                UnicodeCategory.OtherNumber or
                UnicodeCategory.SpaceSeparator or
                UnicodeCategory.LineSeparator or
                UnicodeCategory.ParagraphSeparator or
                UnicodeCategory.Control or
                UnicodeCategory.Format or
                UnicodeCategory.Surrogate or
                UnicodeCategory.PrivateUse or
                UnicodeCategory.ConnectorPunctuation or
                UnicodeCategory.DashPunctuation or
                UnicodeCategory.OpenPunctuation or
                UnicodeCategory.ClosePunctuation or
                UnicodeCategory.InitialQuotePunctuation or
                UnicodeCategory.FinalQuotePunctuation or
                UnicodeCategory.OtherPunctuation or
                UnicodeCategory.MathSymbol or
                UnicodeCategory.CurrencySymbol or
                UnicodeCategory.ModifierSymbol or
                UnicodeCategory.OtherSymbol)
            {
                return ScriptCategory.Common;
            }
        }
        else
        {
            // 补充平面：CJK 扩展 B-E 等
            if (cp >= 0x20000 && cp <= 0x2FA1F) return ScriptCategory.CJK;
        }

        // CJK 范围（包含 CJK 兼容象形文字和扩展）
        if (cp >= 0x4E00 && cp <= 0x9FFF) return ScriptCategory.CJK;
        if (cp >= 0x3400 && cp <= 0x4DBF) return ScriptCategory.CJK;
        if (cp >= 0x3000 && cp <= 0x303F) return ScriptCategory.CJK;
        if (cp >= 0xF900 && cp <= 0xFAFF) return ScriptCategory.CJK;
        if (cp >= 0x2F800 && cp <= 0x2FA1F) return ScriptCategory.CJK;

        // Rust 的 fixed_script: Hiragana 和 Katakana 映射为 CJK（Han）
        if (cp >= 0x3040 && cp <= 0x309F) return ScriptCategory.CJK; // Hiragana → CJK
        if (cp >= 0x30A0 && cp <= 0x30FF) return ScriptCategory.CJK; // Katakana → CJK

        // U+30FC (ー) 长音符号 → CJK
        if (cp == 0x30FC) return ScriptCategory.CJK;

        // 韩文
        if (cp >= 0xAC00 && cp <= 0xD7AF) return ScriptCategory.Hangul;
        if (cp >= 0x1100 && cp <= 0x11FF) return ScriptCategory.Hangul;
        if (cp >= 0x3130 && cp <= 0x318F) return ScriptCategory.Hangul;

        // 西里尔字母
        if (cp >= 0x0400 && cp <= 0x052F) return ScriptCategory.Cyrillic;
        if (cp >= 0x2DE0 && cp <= 0x2DFF) return ScriptCategory.Cyrillic;
        if (cp >= 0xA640 && cp <= 0xA69F) return ScriptCategory.Cyrillic;

        // 阿拉伯字母
        if (cp >= 0x0600 && cp <= 0x06FF) return ScriptCategory.Arabic;
        if (cp >= 0x0750 && cp <= 0x077F) return ScriptCategory.Arabic;
        if (cp >= 0x08A0 && cp <= 0x08FF) return ScriptCategory.Arabic;
        if (cp >= 0xFB50 && cp <= 0xFDFF) return ScriptCategory.Arabic;
        if (cp >= 0xFE70 && cp <= 0xFEFF) return ScriptCategory.Arabic;

        // 天城文
        if (cp >= 0x0900 && cp <= 0x097F) return ScriptCategory.Devanagari;
        if (cp >= 0x0980 && cp <= 0x09FF) return ScriptCategory.Devanagari; // 孟加拉文
        if (cp >= 0x0A00 && cp <= 0x0A7F) return ScriptCategory.Devanagari; // 古木基文
        if (cp >= 0x0A80 && cp <= 0x0AFF) return ScriptCategory.Devanagari; // 古吉拉特文

        // 泰文
        if (cp >= 0x0E00 && cp <= 0x0E7F) return ScriptCategory.Thai;

        // 希伯来文
        if (cp >= 0x0590 && cp <= 0x05FF) return ScriptCategory.Hebrew;

        // 希腊文
        if (cp >= 0x0370 && cp <= 0x03FF) return ScriptCategory.Greek;
        if (cp >= 0x1F00 && cp <= 0x1FFF) return ScriptCategory.Greek;

        // 亚美尼亚文
        if (cp >= 0x0530 && cp <= 0x058F) return ScriptCategory.Armenian;

        // 格鲁吉亚文
        if (cp >= 0x10A0 && cp <= 0x10FF) return ScriptCategory.Georgian;

        // 缅甸文
        if (cp >= 0x1000 && cp <= 0x109F) return ScriptCategory.Myanmar;

        // 高棉文
        if (cp >= 0x1780 && cp <= 0x17FF) return ScriptCategory.Khmer;

        // 老挝文
        if (cp >= 0x0E80 && cp <= 0x0EFF) return ScriptCategory.Lao;

        // 藏文
        if (cp >= 0x0F00 && cp <= 0x0FFF) return ScriptCategory.Tibetan;

        // 拉丁字母（默认用于大多数字母字符）
        if (cp <= 0xFFFF && char.IsLetter((char)cp)) return ScriptCategory.Latin;

        return ScriptCategory.Common;
    }

    private enum ScriptCategory
    {
        Common,
        Latin,
        Cyrillic,
        Arabic,
        CJK,
        Hangul,
        Devanagari,
        Thai,
        Hebrew,
        Greek,
        Armenian,
        Georgian,
        Myanmar,
        Khmer,
        Lao,
        Tibetan
    }
}
