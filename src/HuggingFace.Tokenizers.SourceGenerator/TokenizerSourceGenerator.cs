using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace HuggingFace.Tokenizers.SourceGenerator;

[Generator]
public class TokenizerSourceGenerator : IIncrementalGenerator
{
    private static readonly string[] TargetInterfaces = new[]
    {
        "INormalizer", "IPreTokenizer", "IModel", "IPostProcessor", "IDecoder"
    };

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var componentClasses = context.SyntaxProvider.CreateSyntaxProvider(
            predicate: static (node, _) => node is ClassDeclarationSyntax { AttributeLists.Count: > 0 },
            transform: static (ctx, _) => GetComponentInfo(ctx))
            .Where(static info => info is not null)
            .Collect();

        context.RegisterSourceOutput(componentClasses, static (spc, classes) =>
        {
            var validClasses = classes.Where(c => c is not null).Cast<ComponentInfo>().ToList();
            if (validClasses.Count > 0)
            {
                GenerateFactory(spc, validClasses);
                GeneratePropertyWriter(spc, validClasses);
            }
        });
    }

    private static ComponentInfo? GetComponentInfo(GeneratorSyntaxContext context)
    {
        var classDecl = (ClassDeclarationSyntax)context.Node;
        var symbol = context.SemanticModel.GetDeclaredSymbol(classDecl, CancellationToken.None) as INamedTypeSymbol;
        if (symbol is null) return null;

        foreach (var attr in symbol.GetAttributes())
        {
            if (attr.AttributeClass?.Name == "TokenizerComponentAttribute" &&
                attr.AttributeClass.ContainingNamespace?.ToDisplayString() == "HuggingFace.Tokenizers.Abstractions")
            {
                if (attr.ConstructorArguments.Length >= 1 && attr.ConstructorArguments[0].Value is string jsonTypeName)
                {
                    var enumNaming = EnumNamingConvention.SnakeCase;
                    if (attr.ConstructorArguments.Length == 2 && attr.ConstructorArguments[1].Value is int enumNamingInt)
                        enumNaming = (EnumNamingConvention)enumNamingInt;
                    else if (attr.NamedArguments.Length > 0)
                    {
                        foreach (var na in attr.NamedArguments)
                        {
                            if (na.Key == "EnumNaming" && na.Value.Value is int nv)
                            {
                                enumNaming = (EnumNamingConvention)nv;
                                break;
                            }
                        }
                    }

                    string? matchedInterface = null;
                    foreach (var iface in symbol.AllInterfaces)
                    {
                        if (TargetInterfaces.Contains(iface.Name))
                        {
                            matchedInterface = iface.Name;
                            break;
                        }
                    }

                    if (matchedInterface is not null)
                    {
                        var properties = new List<PropInfo>();
                        foreach (var member in symbol.GetMembers())
                        {
                            if (member is IPropertySymbol prop
                                && !prop.IsStatic
                                && prop.DeclaredAccessibility == Accessibility.Public
                                && prop.GetMethod is not null)
                            {
                                if (HasAttribute(prop, "SkipSerializationAttribute"))
                                    continue;

                                var typeKind = ClassifyType(prop.Type);
                                if (typeKind is not null)
                                {
                                    string? jsonKey = GetJsonKey(prop);
                                    properties.Add(new PropInfo(prop.Name, typeKind, jsonKey));
                                }
                            }
                        }

                        return new ComponentInfo(
                            symbol.Name,
                            symbol.ContainingNamespace?.ToDisplayString() ?? "Global",
                            matchedInterface,
                            jsonTypeName,
                            symbol.IsAbstract,
                            properties,
                            enumNaming);
                    }
                }
            }
        }

        return null;
    }

    /// <summary>
    /// 检查属性是否具有指定名称的特性。
    /// </summary>
    private static bool HasAttribute(IPropertySymbol prop, string attributeName)
    {
        foreach (var attr in prop.GetAttributes())
        {
            if (attr.AttributeClass?.Name == attributeName)
                return true;
        }
        return false;
    }

    /// <summary>
    /// 从 [JsonKey("...")] 特性获取 JSON 键名，若不存在则返回 null。
    /// </summary>
    private static string? GetJsonKey(IPropertySymbol prop)
    {
        foreach (var attr in prop.GetAttributes())
        {
            if (attr.AttributeClass?.Name == "JsonKeyAttribute" &&
                attr.ConstructorArguments.Length == 1 &&
                attr.ConstructorArguments[0].Value is string key)
            {
                return key;
            }
        }
        return null;
    }

    /// <summary>
    /// 将类型分类为序列化类别，不支持的类型返回 null 以跳过。
    /// </summary>
    private static string? ClassifyType(ITypeSymbol type)
    {
        var display = type.ToDisplayString();

        // 内置简单类型
        switch (display)
        {
            case "bool":
            case "string":
            case "char":
            case "int":
            case "uint":
                return display;
        }

        // 枚举 — 序列化为字符串
        if (type.TypeKind == TypeKind.Enum)
            return "enum";

        // 可空值类型（bool?、int?、char? 等）
        if (type.NullableAnnotation == NullableAnnotation.Annotated && type.TypeKind == TypeKind.Struct)
        {
            var namedType = type as INamedTypeSymbol;
            var underlying = namedType?.TypeArguments.FirstOrDefault();
            if (underlying is not null)
            {
                var innerKind = ClassifyType(underlying);
                if (innerKind is not null)
                    return innerKind + "?";
            }
            return null;
        }

        // 跳过其他类型：集合、元组、复杂对象等
        return null;
    }

    private static void GenerateFactory(SourceProductionContext context, List<ComponentInfo> components)
    {
        var normalizerArms = components.Where(c => c.InterfaceName == "INormalizer" && !c.IsAbstract)
            .Select(c => $"                {c.Namespace}.{c.ClassName} => \"{c.JsonTypeName}\",");
        var preTokenizerArms = components.Where(c => c.InterfaceName == "IPreTokenizer" && !c.IsAbstract)
            .Select(c => $"                {c.Namespace}.{c.ClassName} => \"{c.JsonTypeName}\",");
        var modelArms = components.Where(c => c.InterfaceName == "IModel" && !c.IsAbstract)
            .Select(c => $"                {c.Namespace}.{c.ClassName} => \"{c.JsonTypeName}\",");
        var postProcessorArms = components.Where(c => c.InterfaceName == "IPostProcessor" && !c.IsAbstract)
            .Select(c => $"                {c.Namespace}.{c.ClassName} => \"{c.JsonTypeName}\",");
        var decoderArms = components.Where(c => c.InterfaceName == "IDecoder" && !c.IsAbstract)
            .Select(c => $"                {c.Namespace}.{c.ClassName} => \"{c.JsonTypeName}\",");

        var source = $@"// <auto-generated/>
#nullable enable

namespace HuggingFace.Tokenizers.Generated
{{
    /// <summary>
    /// 编译期类型名称映射，用于序列化。
    /// 从 [TokenizerComponent] 特性生成，AOT 兼容，零反射。
    /// </summary>
    public static class TokenizerComponentFactory
    {{
        public static string GetNormalizerTypeName(HuggingFace.Tokenizers.Abstractions.INormalizer obj)
        {{
            return obj switch
            {{
{string.Join("\n", normalizerArms)}
                _ => obj.GetType().Name
            }};
        }}

        public static string GetPreTokenizerTypeName(HuggingFace.Tokenizers.Abstractions.IPreTokenizer obj)
        {{
            return obj switch
            {{
{string.Join("\n", preTokenizerArms)}
                _ => obj.GetType().Name
            }};
        }}

        public static string GetModelTypeName(HuggingFace.Tokenizers.Abstractions.IModel obj)
        {{
            return obj switch
            {{
{string.Join("\n", modelArms)}
                _ => obj.GetType().Name
            }};
        }}

        public static string GetPostProcessorTypeName(HuggingFace.Tokenizers.Abstractions.IPostProcessor obj)
        {{
            return obj switch
            {{
{string.Join("\n", postProcessorArms)}
                _ => obj.GetType().Name
            }};
        }}

        public static string GetDecoderTypeName(HuggingFace.Tokenizers.Abstractions.IDecoder obj)
        {{
            return obj switch
            {{
{string.Join("\n", decoderArms)}
                _ => obj.GetType().Name
            }};
        }}
    }}
}}
";

        context.AddSource("TokenizerComponentFactory.g.cs", source);
    }

    private static void GeneratePropertyWriter(SourceProductionContext context, List<ComponentInfo> components)
    {
        // 为每个接口构建属性写入方法
        var normalizerMethod = BuildPropertyWriterMethod("WriteNormalizerProperties",
            "HuggingFace.Tokenizers.Abstractions.INormalizer", "INormalizer",
            components.Where(c => c.InterfaceName == "INormalizer" && !c.IsAbstract).ToList());

        var preTokenizerMethod = BuildPropertyWriterMethod("WritePreTokenizerProperties",
            "HuggingFace.Tokenizers.Abstractions.IPreTokenizer", "IPreTokenizer",
            components.Where(c => c.InterfaceName == "IPreTokenizer" && !c.IsAbstract).ToList());

        var modelMethod = BuildPropertyWriterMethod("WriteModelProperties",
            "HuggingFace.Tokenizers.Abstractions.IModel", "IModel",
            components.Where(c => c.InterfaceName == "IModel" && !c.IsAbstract).ToList());

        var postProcessorMethod = BuildPropertyWriterMethod("WritePostProcessorProperties",
            "HuggingFace.Tokenizers.Abstractions.IPostProcessor", "IPostProcessor",
            components.Where(c => c.InterfaceName == "IPostProcessor" && !c.IsAbstract).ToList());

        var decoderMethod = BuildPropertyWriterMethod("WriteDecoderProperties",
            "HuggingFace.Tokenizers.Abstractions.IDecoder", "IDecoder",
            components.Where(c => c.InterfaceName == "IDecoder" && !c.IsAbstract).ToList());

        var source = $@"// <auto-generated/>
#nullable enable
using System.Text.Json;

namespace HuggingFace.Tokenizers.Generated
{{
    /// <summary>
    /// 编译期属性写入器，用于序列化。
    /// 从 [TokenizerComponent] 的公共属性生成，AOT 兼容，零反射。
    /// 处理简单属性（bool、string、char、int、uint、enum、nullable）。
    /// 复杂属性（集合、元组、模式）由序列化器手动处理。
    /// </summary>
    public static class ComponentPropertyWriter
    {{
{normalizerMethod}

{preTokenizerMethod}

{modelMethod}

{postProcessorMethod}

{decoderMethod}
    }}
}}
";

        context.AddSource("ComponentPropertyWriter.g.cs", source);
    }

    private static string BuildPropertyWriterMethod(string methodName, string interfaceFullName,
        string interfaceShortName, List<ComponentInfo> components)
    {
        var arms = new List<string>();
        foreach (var comp in components)
        {
            if (comp.Properties.Count == 0)
            {
                arms.Add($"                case {comp.Namespace}.{comp.ClassName}:");
                arms.Add($"                    // 无可写简单属性");
                arms.Add($"                    break;");
            }
            else
            {
                arms.Add($"                case {comp.Namespace}.{comp.ClassName} x:");
                foreach (var prop in comp.Properties)
                {
                    arms.Add($"                    {GeneratePropertyWriteCode(prop, comp.EnumNaming)}");
                }
                arms.Add($"                    break;");
            }
        }

        var switchBody = arms.Count > 0
            ? string.Join("\n", arms)
            : $"                default: break;";

        return $@"        public static void {methodName}({interfaceFullName} obj, Utf8JsonWriter writer)
        {{
            switch (obj)
            {{
{switchBody}
            }}
        }}";
    }

    private static string GeneratePropertyWriteCode(PropInfo prop, EnumNamingConvention enumNaming)
    {
        var jsonKey = prop.JsonKey ?? ToSnakeCase(prop.Name);

        switch (prop.TypeKind)
        {
            case "bool":
                return $"writer.WriteBoolean(\"{jsonKey}\", x.{prop.Name});";
            case "string":
                return $"writer.WriteString(\"{jsonKey}\", x.{prop.Name});";
            case "char":
                return $"writer.WriteString(\"{jsonKey}\", x.{prop.Name}.ToString());";
            case "int":
                return $"writer.WriteNumber(\"{jsonKey}\", x.{prop.Name});";
            case "uint":
                return $"writer.WriteNumber(\"{jsonKey}\", x.{prop.Name});";
            case "enum":
                return enumNaming == EnumNamingConvention.PascalCase
                    ? $"writer.WriteString(\"{jsonKey}\", x.{prop.Name}.ToString());"
                    : $"writer.WriteString(\"{jsonKey}\", x.{prop.Name}.ToString().ToLowerInvariant());";
            case "bool?":
                return $"if (x.{prop.Name}.HasValue) writer.WriteBoolean(\"{jsonKey}\", x.{prop.Name}.Value);";
            case "char?":
                return $"if (x.{prop.Name}.HasValue) writer.WriteString(\"{jsonKey}\", x.{prop.Name}.Value.ToString());";
            case "int?":
                return $"if (x.{prop.Name}.HasValue) writer.WriteNumber(\"{jsonKey}\", x.{prop.Name}.Value);";
            case "uint?":
                return $"if (x.{prop.Name}.HasValue) writer.WriteNumber(\"{jsonKey}\", x.{prop.Name}.Value);";
            default:
                return $"// 跳过: {prop.Name} ({prop.TypeKind})";
        }
    }

    /// <summary>
    /// 将 PascalCase 转换为 snake_case。
    /// </summary>
    private static string ToSnakeCase(string name)
    {
        var sb = new StringBuilder();
        for (int i = 0; i < name.Length; i++)
        {
            if (char.IsUpper(name[i]) && i > 0)
                sb.Append('_');
            sb.Append(char.ToLower(name[i]));
        }
        return sb.ToString();
    }
}

internal sealed class PropInfo
{
    public string Name { get; }
    public string TypeKind { get; }
    public string? JsonKey { get; }

    public PropInfo(string name, string typeKind, string? jsonKey)
    {
        Name = name;
        TypeKind = typeKind;
        JsonKey = jsonKey;
    }
}

internal sealed class ComponentInfo
{
    public string ClassName { get; }
    public string Namespace { get; }
    public string InterfaceName { get; }
    public string JsonTypeName { get; }
    public bool IsAbstract { get; }
    public IReadOnlyList<PropInfo> Properties { get; }
    public EnumNamingConvention EnumNaming { get; }

    public ComponentInfo(string className, string namespaceName, string interfaceName,
        string jsonTypeName, bool isAbstract, IReadOnlyList<PropInfo> properties,
        EnumNamingConvention enumNaming = EnumNamingConvention.SnakeCase)
    {
        ClassName = className;
        Namespace = namespaceName;
        InterfaceName = interfaceName;
        JsonTypeName = jsonTypeName;
        IsAbstract = isAbstract;
        Properties = properties;
        EnumNaming = enumNaming;
    }
}

internal enum EnumNamingConvention
{
    SnakeCase = 0,
    PascalCase = 1
}
