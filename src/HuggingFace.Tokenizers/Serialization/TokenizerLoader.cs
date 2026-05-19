using System.Text.Json;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Serialization;

/// <summary>
/// 从 HuggingFace <c>tokenizer.json</c> 文件或 JSON 字符串加载 <see cref="Tokenizer"/>。
/// 编排组件解析器以重建完整的分词管道。
/// </summary>
public static class TokenizerLoader
{
    /// <summary>
    /// 异步从 HuggingFace Hub 模型 ID 加载 <see cref="Tokenizer"/>。
    /// 如果本地未缓存，则异步下载 <c>tokenizer.json</c>。
    /// </summary>
    /// <param name="modelId">HuggingFace 模型标识符。</param>
    /// <param name="revision">可选的 git 版本。默认为 <c>"main"</c>。</param>
    /// <param name="token">可选的 HuggingFace 访问令牌。</param>
    /// <param name="cancellationToken">取消令牌。</param>
    /// <returns>完全初始化的 <see cref="Tokenizer"/>。</returns>
    public static async Task<Tokenizer> FromPretrainedAsync(
        string modelId,
        string? revision = null,
        string? token = null,
        CancellationToken cancellationToken = default)
    {
        var path = await TokenizerDownloader.DownloadAsync(modelId, revision, token, cancellationToken)
            .ConfigureAwait(false);
        return FromFile(path);
    }

    /// <summary>
    /// 从磁盘上的 <c>tokenizer.json</c> 文件加载 <see cref="Tokenizer"/>。
    /// </summary>
    /// <param name="path">tokenizer.json 文件路径。</param>
    /// <returns>完全初始化的 <see cref="Tokenizer"/>。</returns>
    /// <exception cref="FileNotFoundException">文件不存在。</exception>
    /// <exception cref="InvalidOperationException">JSON 缺少 model 部分或格式错误。</exception>
    public static Tokenizer FromFile(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"分词器文件未找到: '{path}'。");

        var json = File.ReadAllText(path);
        return FromJson(json);
    }

    /// <summary>
    /// 从包含 tokenizer.json 内容的字节数组加载 <see cref="Tokenizer"/>。
    /// </summary>
    /// <param name="bytes">UTF-8 编码的分词器 JSON 字节。</param>
    /// <returns>完全初始化的 <see cref="Tokenizer"/>。</returns>
    /// <exception cref="ArgumentNullException"><paramref name="bytes"/> 为 <c>null</c>。</exception>
    /// <exception cref="InvalidOperationException">JSON 缺少 model 部分或格式错误。</exception>
    public static Tokenizer FromBytes(byte[] bytes)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        var json = System.Text.Encoding.UTF8.GetString(bytes);
        return FromJson(json);
    }

    /// <summary>
    /// 从 HuggingFace <c>tokenizer.json</c> 格式的 JSON 字符串加载 <see cref="Tokenizer"/>。
    /// </summary>
    /// <param name="json">分词器 JSON 字符串。</param>
    /// <returns>完全初始化的 <see cref="Tokenizer"/>。</returns>
    /// <exception cref="InvalidOperationException">JSON 缺少 model 部分或格式错误。</exception>
    public static Tokenizer FromJson(string json)
    {
        if (string.IsNullOrWhiteSpace(json))
            throw new ArgumentException("JSON 字符串不能为空。", nameof(json));

        var jsonModel = JsonSerializer.Deserialize(json, TokenizerJsonContext.Default.TokenizerJsonModel)
            ?? throw new InvalidOperationException("反序列化分词器 JSON 失败。");

        return FromJsonModel(jsonModel);
    }

    /// <summary>
    /// 从已反序列化的 <see cref="TokenizerJsonModel"/> 构建 <see cref="Tokenizer"/>。
    /// </summary>
    /// <param name="jsonModel">反序列化的 JSON 模型。</param>
    /// <returns>完全初始化的 <see cref="Tokenizer"/>。</returns>
    public static Tokenizer FromJsonModel(TokenizerJsonModel jsonModel)
    {
        ArgumentNullException.ThrowIfNull(jsonModel);

        // 1. 解析模型（必需）
        if (jsonModel.Model is null)
            throw new InvalidOperationException("分词器 JSON 缺少必需的 'model' 部分。");

        var model = ModelResolver.Resolve(jsonModel.Model);

        // 2. 使用模型构建分词器
        var builder = new TokenizerBuilder().WithModel(model);

        // 3. 解析可选组件
        if (jsonModel.Normalizer is not null)
            builder.WithNormalizer(NormalizerResolver.Resolve(jsonModel.Normalizer));

        if (jsonModel.PreTokenizer is not null)
            builder.WithPreTokenizer(PreTokenizerResolver.Resolve(jsonModel.PreTokenizer));

        if (jsonModel.PostProcessor is not null)
            builder.WithPostProcessor(PostProcessorResolver.Resolve(jsonModel.PostProcessor));

        if (jsonModel.Decoder is not null)
            builder.WithDecoder(DecoderResolver.Resolve(jsonModel.Decoder));

        // 4. 解析截断配置
        if (jsonModel.Truncation is not null)
        {
            builder.WithTruncation(new TruncationParams
            {
                Strategy = ParseTruncationStrategy(jsonModel.Truncation.Type),
                MaxLength = jsonModel.Truncation.MaxLength,
                Stride = jsonModel.Truncation.Stride,
                Direction = ParseTruncationDirection(jsonModel.Truncation.Direction)
            });
        }

        // 5. 解析填充配置
        if (jsonModel.Padding is not null)
        {
            var (paddingStrategy, fixedLength) = ParsePaddingStrategy(jsonModel.Padding.Type);
            builder.WithPadding(new PaddingParams
            {
                Strategy = paddingStrategy,
                MaxLength = fixedLength ?? jsonModel.Padding.FixedLength ?? jsonModel.Padding.Length ?? 0,
                PadId = jsonModel.Padding.PadId,
                PadTypeId = jsonModel.Padding.PadTypeId,
                PadToken = jsonModel.Padding.PadToken,
                Direction = ParsePaddingDirection(jsonModel.Padding.Direction),
                PadToMultipleOf = jsonModel.Padding.PadToMultipleOf
            });
        }

        // 6. 构建分词器
        var tokenizer = builder.Build();

        // 7. 添加 token（added_tokens 必须在分词器构建后添加，
        //    因为 AddTokens 需要与模型和标准化器交互）
        if (jsonModel.AddedTokens is { Count: > 0 })
        {
            var tokens = jsonModel.AddedTokens.Select(t => new AddedToken(
                content: t.Content,
                id: t.Id,
                isSpecial: t.Special,
                lStrip: t.LStrip,
                rStrip: t.RStrip,
                singleWord: t.SingleWord,
                normalized: t.Normalized
            )).ToList();

            tokenizer.AddTokens(tokens);
        }

        return tokenizer;
    }

    // ─── 枚举解析器 ───────────────────────────────────────────────────────────────

    private static TruncationStrategy ParseTruncationStrategy(string value) => value switch
    {
        "LongestFirst" => TruncationStrategy.LongestFirst,
        "OnlyFirst" => TruncationStrategy.OnlyFirst,
        "OnlySecond" => TruncationStrategy.OnlySecond,
        _ => TruncationStrategy.LongestFirst
    };

    private static TruncationDirection ParseTruncationDirection(string value) => value switch
    {
        "Left" => TruncationDirection.Left,
        "Right" => TruncationDirection.Right,
        _ => TruncationDirection.Right
    };

    private static (PaddingStrategy Strategy, int? FixedLength) ParsePaddingStrategy(string value)
    {
        // Rust PaddingStrategy 序列化格式：
        // BatchLongest → "BatchLongest"
        // Fixed(size) → {"Fixed": size}
        if (value == "BatchLongest")
            return (PaddingStrategy.BatchLongest, null);
        if (value == "Fixed")
            return (PaddingStrategy.Fixed, null);
        return (PaddingStrategy.BatchLongest, null);
    }

    private static PaddingDirection ParsePaddingDirection(string value) => value switch
    {
        "Left" => PaddingDirection.Left,
        "Right" => PaddingDirection.Right,
        _ => PaddingDirection.Right
    };
}
