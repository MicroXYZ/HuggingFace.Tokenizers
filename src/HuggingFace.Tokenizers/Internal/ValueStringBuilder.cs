using System.Buffers;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace HuggingFace.Tokenizers.Internal;

/// <summary>
/// 栈优先的可变长度字符串构建器（ref struct，零堆分配）。
/// 参考微软 ML.Tokenizers ValueStringBuilder，精简为项目需要的子集。
///
/// 使用模式：
///   var sb = new ValueStringBuilder(stackalloc char[256]);
///   sb.Append("hello");
///   sb.Append('!');
///   string result = sb.ToString();
///   sb.Dispose();
///
/// 注意：ref struct 不能跨 await，不能装箱，不能存堆上。
/// </summary>
internal ref struct ValueStringBuilder
{
    private char[]? _arrayToReturnToPool;
    private Span<char> _chars;
    private int _pos;

    /// <summary>
    /// 使用调用方提供的栈缓冲区初始化。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ValueStringBuilder(Span<char> initialBuffer)
    {
        _arrayToReturnToPool = null;
        _chars = initialBuffer;
        _pos = 0;
    }

    /// <summary>
    /// 使用指定初始容量从 ArrayPool 租用缓冲区。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ValueStringBuilder(int initialCapacity)
    {
        _arrayToReturnToPool = ArrayPool<char>.Shared.Rent(initialCapacity);
        _chars = _arrayToReturnToPool;
        _pos = 0;
    }

    /// <summary>当前已写入的字符数。</summary>
    public int Length
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        readonly get => _pos;
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        set
        {
            Debug.Assert(value >= 0);
            Debug.Assert(value <= _chars.Length);
            _pos = value;
        }
    }

    /// <summary>底层缓冲区总容量。</summary>
    public readonly int Capacity => _chars.Length;

    /// <summary>返回已写入内容的只读 Span 视图（零分配）。</summary>
    public readonly ReadOnlySpan<char> AsSpan() => _chars.Slice(0, _pos);
    public readonly ReadOnlySpan<char> AsSpan(int start) => _chars.Slice(start, _pos - start);
    public readonly ReadOnlySpan<char> AsSpan(int start, int length) => _chars.Slice(start, length);

    /// <summary>按索引访问字符。</summary>
    public readonly ref char this[int index]
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get
        {
            Debug.Assert(index < _pos);
            return ref _chars[index];
        }
    }

    /// <summary>确保容量足够，不足时从 ArrayPool 扩容。</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void EnsureCapacity(int capacity)
    {
        Debug.Assert(capacity >= 0);
        if ((uint)capacity > (uint)_chars.Length)
            Grow(capacity - _pos);
    }

    /// <summary>追加单个字符。</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Append(char c)
    {
        int pos = _pos;
        Span<char> chars = _chars;
        if ((uint)pos < (uint)chars.Length)
        {
            chars[pos] = c;
            _pos = pos + 1;
        }
        else
        {
            GrowAndAppend(c);
        }
    }

    /// <summary>追加字符串。</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Append(string? s)
    {
        if (s is null) return;

        int pos = _pos;
        if (s.Length == 1 && (uint)pos < (uint)_chars.Length)
        {
            _chars[pos] = s[0];
            _pos = pos + 1;
        }
        else
        {
            AppendSlow(s);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void AppendSlow(string s)
    {
        int pos = _pos;
        if (pos > _chars.Length - s.Length)
            Grow(s.Length);
        s.AsSpan().CopyTo(_chars.Slice(pos));
        _pos += s.Length;
    }

    /// <summary>追加 ReadOnlySpan 字符。</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Append(ReadOnlySpan<char> value)
    {
        int pos = _pos;
        if (pos > _chars.Length - value.Length)
            Grow(value.Length);
        value.CopyTo(_chars.Slice(pos));
        _pos += value.Length;
    }

    /// <summary>追加 char 重复 count 次。</summary>
    public void Append(char c, int count)
    {
        if (_pos > _chars.Length - count)
            Grow(count);
        _chars.Slice(_pos, count).Fill(c);
        _pos += count;
    }

    /// <summary>
    /// 追加 Rune（正确处理补充平面字符）。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Append(Rune rune)
    {
        if (rune.IsBmp)
        {
            Append((char)rune.Value);
        }
        else
        {
            // 补充平面：两个 char（代理对），直接写入 buffer 避免 Span 逃逸
            int pos = _pos;
            if (pos + 2 > _chars.Length)
                Grow(2);
            int cp = rune.Value - 0x10000;
            _chars[pos] = (char)((cp >> 10) + 0xD800);
            _chars[pos + 1] = (char)((cp & 0x3FF) + 0xDC00);
            _pos = pos + 2;
        }
    }

    /// <summary>
    /// 返回可写入 length 个字符的 Span（自动扩容）。
    /// 写入后手动更新 Length。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public Span<char> AppendSpan(int length)
    {
        int origPos = _pos;
        if (origPos > _chars.Length - length)
            Grow(length);
        _pos = origPos + length;
        return _chars.Slice(origPos, length);
    }

    /// <summary>移除最后一个字符。</summary>
    public void RemoveLastChar()
    {
        if (_pos > 0) _pos--;
    }

    /// <summary>移除指定范围。</summary>
    public void Remove(int start, int length)
    {
        if (length > 0 && start + length <= _pos)
        {
            _chars.Slice(start + length, _pos - start - length).CopyTo(_chars.Slice(start));
            _pos -= length;
        }
    }

    /// <summary>
    /// 转换为 string 并释放内部缓冲区。
    /// 调用后不可再使用此实例。
    /// </summary>
    public override string ToString()
    {
        string s = _chars.Slice(0, _pos).ToString();
        Dispose();
        return s;
    }

    /// <summary>
    /// 创建 string 但不释放缓冲区（用于中间检查）。
    /// </summary>
    public readonly string ToStringNoDispose() => _chars.Slice(0, _pos).ToString();

    /// <summary>
    /// 释放从 ArrayPool 租用的缓冲区。
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Dispose()
    {
        char[]? toReturn = _arrayToReturnToPool;
        this = default; // 防止误用已归还的缓冲区
        if (toReturn is not null)
        {
            ArrayPool<char>.Shared.Return(toReturn);
        }
    }

    // ── 内部扩容逻辑 ──

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void Grow(int additionalCapacityBeyondPos)
    {
        Debug.Assert(additionalCapacityBeyondPos > 0);
        Debug.Assert(_pos > _chars.Length - additionalCapacityBeyondPos);

        int newCapacity = (int)Math.Max(
            (uint)(_pos + additionalCapacityBeyondPos),
            Math.Min((uint)_chars.Length * 2, 0x7FFFFFC7));

        char[] poolArray = ArrayPool<char>.Shared.Rent(newCapacity);
        _chars.Slice(0, _pos).CopyTo(poolArray);

        char[]? toReturn = _arrayToReturnToPool;
        _chars = _arrayToReturnToPool = poolArray;
        if (toReturn is not null)
        {
            ArrayPool<char>.Shared.Return(toReturn);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void GrowAndAppend(char c)
    {
        Grow(1);
        Append(c);
    }
}
