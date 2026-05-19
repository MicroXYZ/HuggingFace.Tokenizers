using System.Collections.Concurrent;
using System.Net;
using System.Net.Http.Headers;

namespace HuggingFace.Tokenizers.Serialization;

/// <summary>
/// 从 HuggingFace Hub 下载并缓存 <c>tokenizer.json</c> 文件。
/// 使用临时文件 + 原子重命名防止并发下载竞态。
/// </summary>
public static class TokenizerDownloader
{
    private static readonly HttpClient s_httpClient = new()
    {
        Timeout = TimeSpan.FromMinutes(5)
    };

    // 按 modelId 去重并发下载
    private static readonly ConcurrentDictionary<string, SemaphoreSlim> s_downloadLocks = new();

    /// <summary>
    /// 默认缓存目录：<c>~/.cache/huggingface/tokenizers/</c>
    /// </summary>
    public static string DefaultCacheDir { get; set; } = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
        ".cache", "huggingface", "tokenizers");

    /// <summary>
    /// 从 HuggingFace Hub 下载 <c>tokenizer.json</c> 并返回本地缓存路径。
    /// 如果缓存中已存在该文件，则直接返回缓存路径，不重复下载。
    /// 使用临时文件 + 原子重命名防止并发下载导致文件损坏。
    /// </summary>
    public static async Task<string> DownloadAsync(
        string modelId,
        string? revision = null,
        string? token = null,
        CancellationToken cancellationToken = default)
    {
        if (string.IsNullOrWhiteSpace(modelId))
            throw new ArgumentException("模型 ID 不能为空。", nameof(modelId));

        var cacheDir = Path.Combine(DefaultCacheDir, modelId.Replace("/", "--"));
        var cachePath = Path.Combine(cacheDir, "tokenizer.json");

        // 缓存命中，直接返回
        if (File.Exists(cachePath))
            return cachePath;

        // 按 modelId 去重：同一模型的并发下载只执行一次
        var downloadLock = s_downloadLocks.GetOrAdd(modelId, _ => new SemaphoreSlim(1, 1));
        await downloadLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            // 双重检查：等待锁期间可能已被其他调用完成
            if (File.Exists(cachePath))
                return cachePath;

            // 构建下载 URL
            var resolvedRevision = revision ?? "main";
            var url = $"https://huggingface.co/{modelId}/resolve/{resolvedRevision}/tokenizer.json";

            // 构建 HTTP 请求
            using var request = new HttpRequestMessage(HttpMethod.Get, url);
            if (!string.IsNullOrEmpty(token))
                request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);

            // 执行请求
            using var response = await s_httpClient.SendAsync(request, cancellationToken).ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
            {
                var reason = response.StatusCode switch
                {
                    HttpStatusCode.NotFound => $"未找到模型 '{modelId}'（版本 '{resolvedRevision}'）的分词器。请检查模型 ID 和版本。",
                    HttpStatusCode.Unauthorized => $"访问模型 '{modelId}' 被拒绝。请提供有效的令牌以访问私有/受控模型。",
                    HttpStatusCode.Forbidden => $"访问模型 '{modelId}' 被禁止。您可能缺少必要的权限。",
                    _ => $"下载模型 '{modelId}' 的分词器失败：{(int)response.StatusCode} {response.ReasonPhrase}。"
                };

                throw new HttpRequestException(reason, inner: null, statusCode: response.StatusCode);
            }

            // 先写入临时文件，再原子重命名（防止进程崩溃留下不完整文件）
            Directory.CreateDirectory(cacheDir);
            var tmpPath = cachePath + $".tmp.{Guid.NewGuid():N}";
            try
            {
                var content = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
                await File.WriteAllTextAsync(tmpPath, content, cancellationToken).ConfigureAwait(false);
                File.Move(tmpPath, cachePath, overwrite: true);
            }
            finally
            {
                // 清理残留临时文件
                try { if (File.Exists(tmpPath)) File.Delete(tmpPath); } catch (IOException) { }
            }

            return cachePath;
        }
        finally
        {
            downloadLock.Release();
        }
    }
}
