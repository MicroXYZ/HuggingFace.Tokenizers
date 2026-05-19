namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// GPT-2 字节级编码映射表。
/// ByteLevelDecoder、ByteLevelPreTokenizer、ByteLevelNormalizer 共用此映射。
///
/// 映射规则（与 Rust bytes_char() 一致）：
/// - 可打印 ASCII 0x21-0x7E → 自身
/// - Latin-1 Supplement 0xA1-0xAC, 0xAE-0xFF → 自身
/// - 其余字节 → 256 + n（n 为顺序索引）
///
/// 参见 https://github.com/openai/gpt-2/blob/master/src/encoder.py#L9
/// </summary>
internal static class ByteLevelMapping
{
    /// <summary>
    /// 字节 → Unicode 字符映射（256 个条目）。
    /// </summary>
    internal static readonly char[] ByteToChar = BuildByteToCharMap();

    /// <summary>
    /// Unicode 字符 → 字节反向映射。
    /// </summary>
    internal static readonly Dictionary<char, byte> CharToByte = BuildCharToByteMap();

    /// <summary>
    /// Unicode 字符 → 字节直接数组查表（O(1) 替代 Dictionary 哈希）。
    /// 最大映射字符值为 511（256 + 255），数组大小 512 足够。
    /// </summary>
    internal static readonly byte[] CharToByteDirect = BuildCharToByteDirect();

    /// <summary>
    /// 字节 → Unicode 字符的 Dictionary 版本（供 ByteLevelNormalizer 等需要 Dictionary 的场景使用）。
    /// </summary>
    internal static readonly Dictionary<byte, char> ByteToCharDict = BuildByteToCharDict();

    private static Dictionary<byte, char> BuildByteToCharDict()
    {
        var map = new Dictionary<byte, char>(256);
        for (int i = 0; i < 256; i++)
            map[(byte)i] = ByteToChar[i];
        return map;
    }

    private static char[] BuildByteToCharMap()
    {
        var map = new char[256];

        // 阶段 1：自身映射的字节
        // 可打印 ASCII：0x21 ('!') 到 0x7E ('~')
        for (int i = 0x21; i <= 0x7E; i++)
            map[i] = (char)i;

        // Latin-1 Supplement：0xA1 到 0xAC
        for (int i = 0xA1; i <= 0xAC; i++)
            map[i] = (char)i;

        // Latin-1 Supplement：0xAE 到 0xFF
        for (int i = 0xAE; i <= 0xFF; i++)
            map[i] = (char)i;

        // 阶段 2：其余字节映射到 256 + 顺序索引
        int n = 0;
        for (int b = 0; b <= 255; b++)
        {
            if ((b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF))
                continue;

            map[b] = (char)(256 + n);
            n++;
        }

        return map;
    }

    private static Dictionary<char, byte> BuildCharToByteMap()
    {
        var map = new Dictionary<char, byte>(256);
        for (int i = 0; i < 256; i++)
        {
            map[ByteToChar[i]] = (byte)i;
        }
        return map;
    }

    private static byte[] BuildCharToByteDirect()
    {
        // 最大映射字符值为 511（256 + 255）
        var map = new byte[512];
        for (int i = 0; i < 256; i++)
        {
            char c = ByteToChar[i];
            if (c < 512)
                map[c] = (byte)i;
        }
        return map;
    }
}
