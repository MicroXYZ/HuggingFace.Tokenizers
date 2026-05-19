using System.Buffers;
using System.IO.Hashing;
using System.Runtime.CompilerServices;
using SysEncoding = System.Text.Encoding;

namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// 基于 UTF-8 字节的开放寻址哈希表。
/// 支持 <see cref="ReadOnlySpan{T}"/> 直接查找，避免 UTF-16 string 分配。
/// 使用 XxHash3（硬件加速）+ 线性探测。
/// </summary>
internal sealed class Utf8Vocab
{
    private byte[][]? _keys;
    private uint[]? _values;
    private int[]? _hashes;
    private int _count;
    private int _capacity;
    private const int InitialCapacity = 16;
    private const double LoadFactor = 0.75;

    // Id → UTF-8 bytes 反向索引，O(1) 查找
    private byte[][]? _idToBytes;

    /// <summary>
    /// 从 string 词表构建 UTF-8 词表（初始化时调用一次）。
    /// </summary>
    public Utf8Vocab(IReadOnlyDictionary<string, uint> vocab)
    {
        _capacity = InitialCapacity;
        while (_capacity < vocab.Count * 2) _capacity <<= 1;
        int mask = _capacity - 1;

        _keys = new byte[_capacity][];
        _values = new uint[_capacity];
        _hashes = new int[_capacity];
        Array.Fill(_hashes, -1);
        _count = 0;

        foreach (var (token, id) in vocab)
        {
            byte[] utf8 = SysEncoding.UTF8.GetBytes(token);
            Insert(utf8, id, mask);
        }

        // 构建 Id → bytes 反向索引
        BuildReverseIndex(vocab.Count);
    }

    /// <summary>
    /// 构建 Id → UTF-8 bytes 反向索引。
    /// </summary>
    private void BuildReverseIndex(int vocabCount)
    {
        // 找到最大 ID，按 maxId+1 分配
        uint maxId = 0;
        for (int i = 0; i < _capacity; i++)
        {
            if (_hashes![i] != -1 && _values![i] > maxId)
                maxId = _values[i];
        }

        _idToBytes = new byte[maxId + 1][];
        for (int i = 0; i < _capacity; i++)
        {
            if (_hashes![i] != -1)
                _idToBytes[_values![i]] = _keys![i];
        }
    }

    private void Insert(byte[] key, uint value, int mask)
    {
        int hash = ComputeHash(key);
        int idx = hash & mask;

        while (_hashes![idx] != -1)
        {
            if (_hashes[idx] == hash && BytesEqual(_keys![idx], key))
            {
                _values![idx] = value; // 覆盖
                return;
            }
            idx = (idx + 1) & mask;
        }

        _hashes[idx] = hash;
        _keys![idx] = key;
        _values![idx] = value;
        _count++;

        if (_count > (int)(_capacity * LoadFactor))
            Resize();
    }

    private void Resize()
    {
        int newCapacity = _capacity << 1;
        int newMask = newCapacity - 1;
        var newKeys = new byte[newCapacity][];
        var newValues = new uint[newCapacity];
        var newHashes = new int[newCapacity];
        Array.Fill(newHashes, -1);

        for (int i = 0; i < _capacity; i++)
        {
            if (_hashes![i] != -1)
            {
                int idx = _hashes[i] & newMask;
                while (newHashes[idx] != -1)
                    idx = (idx + 1) & newMask;
                newHashes[idx] = _hashes[i];
                newKeys[idx] = _keys![i];
                newValues[idx] = _values![i];
            }
        }

        _keys = newKeys;
        _values = newValues;
        _hashes = newHashes;
        _capacity = newCapacity;
    }

    /// <summary>
    /// UTF-8 字节直接查找，零分配。
    /// </summary>
    public bool TryGetId(ReadOnlySpan<byte> utf8Token, out uint id)
    {
        if (_count == 0) { id = 0; return false; }

        int hash = ComputeHash(utf8Token);
        int mask = _capacity - 1;
        int idx = hash & mask;

        while (_hashes![idx] != -1)
        {
            if (_hashes[idx] == hash && _keys![idx].AsSpan().SequenceEqual(utf8Token))
            {
                id = _values![idx];
                return true;
            }
            idx = (idx + 1) & mask;
        }

        id = 0;
        return false;
    }

    /// <summary>
    /// UTF-16 string 查找（兼容路径，内部转 UTF-8 后查找）。
    /// </summary>
    public bool TryGetId(ReadOnlySpan<char> token, out uint id)
    {
        // 预估最大 UTF-8 长度：每个 char 最多 3 字节
        int maxUtf8Len = token.Length * 3;
        byte[]? pooled = null;
        Span<byte> utf8 = maxUtf8Len <= 256
            ? stackalloc byte[maxUtf8Len]
            : (pooled = ArrayPool<byte>.Shared.Rent(maxUtf8Len));
        try
        {
            int written = SysEncoding.UTF8.GetBytes(token, utf8);
            return TryGetId(utf8.Slice(0, written), out id);
        }
        finally
        {
            if (pooled is not null) ArrayPool<byte>.Shared.Return(pooled);
        }
    }

    /// <summary>
    /// UTF-8 字节查找对应的 token bytes。O(1) 反向索引查找。
    /// </summary>
    public bool TryGetTokenBytes(uint id, out byte[]? tokenBytes)
    {
        if (_idToBytes is not null && id < _idToBytes.Length)
        {
            tokenBytes = _idToBytes[id];
            return tokenBytes is not null;
        }
        tokenBytes = null;
        return false;
    }

    /// <summary>
    /// 根据 token ID 获取 token 字符串。O(1) 反向索引查找。
    /// </summary>
    public string GetTokenString(uint id)
    {
        if (_idToBytes is not null && id < _idToBytes.Length && _idToBytes[id] is not null)
            return SysEncoding.UTF8.GetString(_idToBytes[id]);
        return $"<{id}>";
    }

    /// <summary>当前存储的 token 数量。</summary>
    public int Count => _count;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ComputeHash(ReadOnlySpan<byte> data)
    {
        // XxHash3：.NET 10 硬件加速（SSE2/NEON），比 FNV-1a 快 2-4x
        return (int)XxHash3.HashToUInt64(data);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool BytesEqual(byte[] a, ReadOnlySpan<byte> b)
    {
        return a.AsSpan().SequenceEqual(b);
    }
}
