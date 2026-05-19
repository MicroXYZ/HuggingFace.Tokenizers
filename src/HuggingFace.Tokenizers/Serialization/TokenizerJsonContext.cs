using System.Text.Json;
using System.Text.Json.Serialization;
using HuggingFace.Tokenizers.Abstractions;

namespace HuggingFace.Tokenizers.Serialization;

/// <summary>
/// AOT-compatible JSON 序列化上下文 for all tokenizer types.
/// </summary>
[JsonSerializable(typeof(TokenizerJsonModel))]
[JsonSerializable(typeof(NormalizerJsonModel))]
[JsonSerializable(typeof(PreTokenizerJsonModel))]
[JsonSerializable(typeof(ModelJsonModel))]
[JsonSerializable(typeof(PostProcessorJsonModel))]
[JsonSerializable(typeof(DecoderJsonModel))]
[JsonSerializable(typeof(AddedTokenJsonModel))]
[JsonSerializable(typeof(TruncationJsonModel))]
[JsonSerializable(typeof(PaddingJsonModel))]
[JsonSerializable(typeof(Dictionary<string, uint>))]
[JsonSerializable(typeof(List<List<string>>))]
[JsonSerializable(typeof(List<string>))]
public partial class TokenizerJsonContext : JsonSerializerContext
{
}
