using System.Collections.Concurrent;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// 后处理器辅助方法。
/// BertProcessing、RobertaProcessing、TemplateProcessing 共用的 MakeSpecialToken/WithTypeId。
/// </summary>
internal static class PostProcessorHelper
{
    // 缓存：单元素 Encoding 的 (Id, TypeId) → Encoding 映射。
    // MakeSpecialToken 创建的 Encoding 是单元素且内容不变，可安全共享。
    // MergeWith 会对共享实例自动 COW，不会污染缓存。
    private static readonly ConcurrentDictionary<(uint Id, uint TypeId), Encoding> _specialTokenCache = new();

    /// <summary>
    /// 创建包含单个特殊 token 的 Encoding。命中缓存时零分配。
    /// 缓存的 Encoding 必须标记为 shared，防止 Merge/MergeWith 原地修改污染缓存。
    /// </summary>
    internal static Encoding MakeSpecialToken((string Token, uint Id) token, uint typeId)
    {
        return _specialTokenCache.GetOrAdd((token.Id, typeId),
            key =>
            {
                var enc = new Encoding(
                    ids: [key.Id],
                    typeIds: [key.TypeId],
                    tokens: [token.Token],
                    words: [null],
                    offsets: [(0, 0)],
                    specialTokensMask: [1],
                    attentionMask: [1]);
                enc.MarkAsShared();
                return enc;
            });
    }

    /// <summary>
    /// 将 encoding 的所有 typeIds 设置为指定值。
    /// 利用 COW 机制：Clone 共享底层数组，只替换 typeIds。
    /// 比原来 6 次 ToArray() 拷贝更高效。
    /// </summary>
    internal static Encoding WithTypeId(Encoding encoding, uint typeId)
    {
        var clone = encoding.Clone(); // COW 浅拷贝，共享底层数组
        var typeIds = new uint[encoding.Length];
        Array.Fill(typeIds, typeId);
        clone.SetTypeIds(typeIds);    // 只修改 typeIds（触发 COW 写时复制）
        return clone;
    }
}
