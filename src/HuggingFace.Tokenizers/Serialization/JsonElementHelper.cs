using System.Text.Json;

namespace HuggingFace.Tokenizers.Serialization;

/// <summary>
/// 共享的 JSON 元素访问工具方法。
/// 从 5 个 Resolver 中提取的重复 helper 方法，统一维护。
/// </summary>
internal static class JsonElementHelper
{
    // ── Boolean ──────────────────────────────────────────────────────────────

    /// <summary>
    /// 读取布尔值，不存在时返回默认值。
    /// </summary>
    public static bool GetBool(Dictionary<string, JsonElement>? data, string key, bool defaultValue)
    {
        if (data is not null && data.TryGetValue(key, out var element)
            && (element.ValueKind == JsonValueKind.True || element.ValueKind == JsonValueKind.False))
        {
            return element.GetBoolean();
        }
        return defaultValue;
    }

    /// <summary>
    /// 读取可选布尔值（Rust Option&lt;bool&gt;），不存在时返回 null。
    /// </summary>
    public static bool? GetOptionalBool(Dictionary<string, JsonElement>? data, string key)
    {
        if (data is not null && data.TryGetValue(key, out var element)
            && (element.ValueKind == JsonValueKind.True || element.ValueKind == JsonValueKind.False))
        {
            return element.GetBoolean();
        }
        return null;
    }

    // ── String ───────────────────────────────────────────────────────────────

    /// <summary>
    /// 读取可选字符串，不存在时返回 null。
    /// </summary>
    public static string? GetString(Dictionary<string, JsonElement>? data, string key)
    {
        if (data is not null && data.TryGetValue(key, out var element)
            && element.ValueKind == JsonValueKind.String)
        {
            return element.GetString();
        }
        return null;
    }

    /// <summary>
    /// 读取字符串，不存在时返回指定默认值。
    /// </summary>
    public static string GetString(Dictionary<string, JsonElement>? data, string key, string defaultValue)
    {
        if (data is not null && data.TryGetValue(key, out var element)
            && element.ValueKind == JsonValueKind.String)
        {
            return element.GetString()!;
        }
        return defaultValue;
    }

    /// <summary>
    /// 读取必需字符串，不存在时抛出异常。
    /// </summary>
    public static string GetRequiredString(Dictionary<string, JsonElement>? data, string key)
    {
        if (data is not null && data.TryGetValue(key, out var element)
            && element.ValueKind == JsonValueKind.String)
        {
            return element.GetString()!;
        }
        throw new ArgumentException($"Required string '{key}' is missing or invalid in AdditionalData.");
    }

    // ── Number ───────────────────────────────────────────────────────────────

    /// <summary>
    /// 读取可选整数，不存在时返回 null。
    /// </summary>
    public static int? GetInt(Dictionary<string, JsonElement>? data, string key)
    {
        if (data is not null && data.TryGetValue(key, out var element)
            && element.ValueKind == JsonValueKind.Number)
        {
            return element.GetInt32();
        }
        return null;
    }

    /// <summary>
    /// 读取整数，不存在时返回指定默认值。
    /// </summary>
    public static int GetInt(Dictionary<string, JsonElement>? data, string key, int defaultValue)
    {
        if (data is not null && data.TryGetValue(key, out var element)
            && element.ValueKind == JsonValueKind.Number)
        {
            return element.GetInt32();
        }
        return defaultValue;
    }

    /// <summary>
    /// 读取浮点数，不存在时返回指定默认值。
    /// </summary>
    public static float GetFloat(Dictionary<string, JsonElement>? data, string key, float defaultValue)
    {
        if (data is not null && data.TryGetValue(key, out var element)
            && element.ValueKind == JsonValueKind.Number)
        {
            return (float)element.GetDouble();
        }
        return defaultValue;
    }

    /// <summary>
    /// 读取可选浮点数，不存在时返回 null。
    /// </summary>
    public static float? GetOptionalFloat(Dictionary<string, JsonElement>? data, string key)
    {
        if (data is not null && data.TryGetValue(key, out var element)
            && element.ValueKind == JsonValueKind.Number)
        {
            return (float)element.GetDouble();
        }
        return null;
    }

    // ── Char ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// 读取单个字符（从字符串的第一个字符），不存在时返回默认值。
    /// </summary>
    public static char GetChar(Dictionary<string, JsonElement>? data, string key, char defaultValue)
    {
        if (data is not null && data.TryGetValue(key, out var element)
            && element.ValueKind == JsonValueKind.String)
        {
            var str = element.GetString();
            if (str is { Length: > 0 }) return str[0];
        }
        return defaultValue;
    }

    /// <summary>
    /// 宽松读取 uint 值，兼容 JSON number 和 string 两种格式。
    /// 某些 HuggingFace tokenizer.json 中 id/type_id 以字符串形式存储（如 "0" 而非 0），
    /// Rust serde_json 会自动做类型强转，这里提供等价行为。
    /// </summary>
    public static uint GetUInt32Loose(JsonElement element)
    {
        if (element.ValueKind == JsonValueKind.Number)
            return element.GetUInt32();
        if (element.ValueKind == JsonValueKind.String)
        {
            var str = element.GetString();
            if (uint.TryParse(str, out var v))
                return v;
        }
        throw new InvalidOperationException(
            $"The requested operation requires an element of type 'Number', but the target element has type '{element.ValueKind}'.");
    }

    /// <summary>
    /// 宽松读取 int 值，兼容 JSON number 和 string 两种格式。
    /// 与 GetUInt32Loose 对称，处理某些 tokenizer.json 中整数以字符串存储的情况。
    /// </summary>
    public static int GetIntLoose(JsonElement element)
    {
        if (element.ValueKind == JsonValueKind.Number)
            return element.GetInt32();
        if (element.ValueKind == JsonValueKind.String)
        {
            var str = element.GetString();
            if (int.TryParse(str, out var v))
                return v;
        }
        throw new InvalidOperationException(
            $"The requested operation requires an element of type 'Number', but the target element has type '{element.ValueKind}'.");
    }

    // ── Array ────────────────────────────────────────────────────────────────

    /// <summary>
    /// 读取 JSON 数组，不存在时返回 null。
    /// </summary>
    public static JsonElement[]? GetArray(Dictionary<string, JsonElement>? data, string key)
    {
        if (data is not null && data.TryGetValue(key, out var element)
            && element.ValueKind == JsonValueKind.Array)
        {
            return element.EnumerateArray().ToArray();
        }
        return null;
    }

    // ── Pattern (String / Regex) ─────────────────────────────────────────────

    /// <summary>
    /// 读取模式字符串，支持 Rust 对象格式 {"String":"..."} / {"Regex":"..."} 和纯字符串格式。
    /// </summary>
    public static string? GetPatternString(Dictionary<string, JsonElement>? data, string key)
    {
        if (data is null || !data.TryGetValue(key, out var element))
            return null;

        // Object format: {"String":"..."} 或 {"Regex":"..."}
        if (element.ValueKind == JsonValueKind.Object)
        {
            if (element.TryGetProperty("String", out var s) && s.ValueKind == JsonValueKind.String)
                return s.GetString();
            if (element.TryGetProperty("Regex", out var r) && r.ValueKind == JsonValueKind.String)
                return r.GetString();
        }

        // 纯字符串格式
        if (element.ValueKind == JsonValueKind.String)
            return element.GetString();

        return null;
    }

    /// <summary>
    /// 判断模式值是否为正则模式。
    /// </summary>
    public static bool IsRegexPattern(Dictionary<string, JsonElement>? data, string key)
    {
        if (data is null || !data.TryGetValue(key, out var element))
            return false;

        if (element.ValueKind == JsonValueKind.Object && element.TryGetProperty("Regex", out _))
            return true;

        return false;
    }

    // ── Token Pair ───────────────────────────────────────────────────────────

    /// <summary>
    /// 读取 token 对 [token_string, token_id]。
    /// </summary>
    public static (string Token, uint Id) ReadTokenPair(Dictionary<string, JsonElement>? data, string key)
    {
        if (data is null || !data.TryGetValue(key, out var element))
            throw new ArgumentException($"Required '{key}' array is missing in AdditionalData.");

        if (element.ValueKind != JsonValueKind.Array)
            throw new ArgumentException($"Expected '{key}' to be an array, but got {element.ValueKind}.");

        var arr = element.EnumerateArray().ToArray();
        if (arr.Length < 2)
            throw new ArgumentException($"Expected '{key}' to contain at least 2 elements [token, id].");

        var token = arr[0].GetString()
            ?? throw new ArgumentException($"Token string in '{key}' array cannot be null.");
        var id = GetUInt32Loose(arr[1]);

        return (token, id);
    }

    // ── Require / Try ────────────────────────────────────────────────────────

    /// <summary>
    /// 要求属性存在并通过 reader 转换，缺失时抛出异常。
    /// </summary>
    public static T RequireProperty<T>(
        Dictionary<string, JsonElement>? data,
        string key,
        Func<JsonElement, T> reader)
    {
        if (data is null || !data.TryGetValue(key, out var element))
            throw new InvalidOperationException(
                $"Configuration is missing required property '{key}'.");

        return reader(element);
    }

    /// <summary>
    /// 属性存在时通过 reader 转换，否则返回 default。
    /// </summary>
    public static T? TryGetProperty<T>(
        Dictionary<string, JsonElement>? data,
        string key,
        Func<JsonElement, T> reader)
    {
        if (data is not null && data.TryGetValue(key, out var element))
            return reader(element);
        return default;
    }
}
