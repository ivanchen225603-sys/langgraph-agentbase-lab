"""Typed tool declarations; all invocation still goes through ToolExecutor."""

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, get_type_hints

from langchain_core.tools import BaseTool
from langsmith import tracing_context
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError, create_model

from .configuration import AgentPackage, ToolDeclaration
from .contracts import AgentError, ProviderFactory, RunContext, ToolHandler
from .effects import AttemptCommit
from .registry import ComponentRegistry
from .schema import SchemaContract, canonical_json
from .workflows import register_workflows


def _contains_context(annotation):
    # Compiled schemas include dataclasses, TypedDict, aliases and recursive models.
    pending: list[Any] = [TypeAdapter(annotation).core_schema]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, dict):
            model = node.get("cls")
            if isinstance(model, type) and issubclass(model, RunContext):
                return True
            pending.extend(value for value in node.values() if isinstance(value, (dict, list)))
        elif isinstance(node, list):
            pending.extend(node)
    return False


@dataclass(frozen=True)
class ToolBinding:
    declaration: ToolDeclaration
    handler: ToolHandler

    def register(self, registry: ComponentRegistry, *, attempt_commit: AttemptCommit | None = None):
        registry.register_handler(
            self.declaration.handler,
            self.handler,
            version=self.declaration.handler_version or "1",
            attempt_commit=attempt_commit,
        )
        return self.declaration.model_copy(deep=True)


def build_registry(
    package: AgentPackage,
    *,
    tools: Sequence[ToolBinding] = (),
    providers: Mapping[str, ProviderFactory] | None = None,
    workflows: Mapping[str, Callable] | None = None,
) -> ComponentRegistry:
    """Small trusted assembly helper. Advanced version restoration uses the registry directly."""
    registry = ComponentRegistry()
    register_workflows(registry)
    for binding in tools:
        declared = next(
            (tool for tool in package.tools if tool.name == binding.declaration.name), None
        )
        if declared != binding.declaration:
            raise AgentError("tool_adapter", "package tool differs from its typed binding")
        binding.register(registry)
    for name, factory in (providers or {}).items():
        registry.register_provider(name, factory)
    for name, factory in (workflows or {}).items():
        registry.register_workflow(name, factory)
    registry.bind(package)
    return registry


def function_tool(
    function: Callable,
    *,
    name: str,
    scope: str,
    read_only: bool,
    description: str | None = None,
    context_parameter: str | None = None,
    **policy: Any,
) -> ToolBinding:
    """Adapt a native async, fully typed function. Policies are trusted host input.

    context_parameter explicitly injects a RunContext; it is never model input.
    Sync functions are rejected: cancelling a thread cannot stop a write effect.
    Write tools must explicitly supply their existing idempotency/receipt contract.
    """
    if not inspect.iscoroutinefunction(function):
        raise AgentError("tool_adapter", "a native async function is required")
    signature = inspect.signature(function)
    try:
        hints = get_type_hints(function, include_extras=True)
    except (NameError, TypeError) as exc:
        raise AgentError("tool_adapter", "tool annotations cannot be resolved") from exc
    fields: dict[str, Any] = {}
    for key, parameter in signature.parameters.items():
        if parameter.kind not in {parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY}:
            raise AgentError("tool_adapter", "tool parameters must be named arguments")
        annotation = hints.get(key)
        if key == context_parameter:
            if annotation is not RunContext:
                raise AgentError("tool_adapter", "injected context must be RunContext")
            continue
        if _contains_context(annotation):
            raise AgentError("tool_adapter", "RunContext requires explicit server injection")
        if annotation is None or annotation is Any:
            raise AgentError("tool_adapter", "tool parameters require explicit types")
        fields[key] = (
            annotation,
            ... if parameter.default is parameter.empty else parameter.default,
        )
    if context_parameter is not None and context_parameter not in signature.parameters:
        raise AgentError("tool_adapter", "injected context parameter is missing")
    result_type = hints.get("return")
    if result_type is None or result_type is Any:
        raise AgentError("tool_adapter", "tool return type is required")
    if _contains_context(result_type):
        raise AgentError("tool_adapter", "RunContext cannot be returned as model data")
    arguments_type = create_model(
        "ToolArguments", __config__=ConfigDict(extra="forbid", strict=True), **fields
    )
    output = TypeAdapter(result_type)
    declaration = ToolDeclaration(
        name=name,
        handler=name,
        description=description or inspect.getdoc(function) or name,
        scope=scope,
        read_only=read_only,
        retry_safe=policy.pop("retry_safe", read_only),
        input_schema=arguments_type.model_json_schema(),
        output_schema=output.json_schema(),
        **policy,
    )
    result_schema = SchemaContract(declaration.output_schema)

    async def handler(arguments, context):
        try:
            parsed = arguments_type.model_validate_json(canonical_json(arguments), strict=True)
        except ValidationError as exc:
            raise AgentError(
                "tool_arguments", "tool arguments do not match declared types"
            ) from exc
        kwargs = {key: getattr(parsed, key) for key in fields}
        if context_parameter is not None:
            kwargs[context_parameter] = context
        value = await function(**kwargs)
        try:
            checked = output.validate_python(value, strict=True)
            payload = output.dump_python(checked, mode="json", warnings="error")
        except (ValidationError, ValueError, TypeError) as exc:
            raise AgentError("tool_result", "tool result does not match declared type") from exc
        return result_schema.validate(
            payload, code="tool_result", max_bytes=declaration.result_bytes
        )

    return ToolBinding(declaration, handler)


def langchain_tool(
    tool: BaseTool,
    *,
    output_type: Any,
    name: str,
    scope: str,
    read_only: bool,
    **policy: Any,
) -> ToolBinding:
    """Adapt native async LangChain tools, with an explicit JSON output contract.

    This entry supports tools exposing a coroutine, e.g. StructuredTool. It rejects
    implicit thread fallbacks and structured content/artifact response modes.
    """
    if not isinstance(tool, BaseTool) or not inspect.iscoroutinefunction(
        getattr(tool, "coroutine", None)
    ):
        raise AgentError("tool_adapter", "a native async LangChain tool is required")
    if tool.response_format != "content" or output_type is Any:
        raise AgentError("tool_adapter", "an explicit JSON output contract is required")
    if tool.handle_tool_error or tool.handle_validation_error or tool.callbacks:
        raise AgentError("tool_adapter", "tool error suppression and callbacks must be disabled")
    if _contains_context(output_type):
        raise AgentError("tool_adapter", "RunContext cannot be returned as model data")
    input_type = tool.get_input_schema()
    if not issubclass(input_type, BaseModel):
        raise AgentError("tool_adapter", "Pydantic v2 input models are required")
    if _contains_context(input_type):
        raise AgentError("tool_adapter", "RunContext cannot be supplied by a LangChain input")
    output = TypeAdapter(output_type)
    input_schema = input_type.model_json_schema()
    input_schema["additionalProperties"] = False
    declaration = ToolDeclaration(
        name=name,
        handler=name,
        description=tool.description,
        scope=scope,
        read_only=read_only,
        retry_safe=policy.pop("retry_safe", read_only),
        input_schema=input_schema,
        output_schema=output.json_schema(),
        **policy,
    )
    arguments_schema, result_schema = (
        SchemaContract(input_schema),
        SchemaContract(declaration.output_schema),
    )

    async def handler(arguments, context):
        checked = arguments_schema.validate(arguments, code="tool_arguments")
        # No provider callbacks, model metadata or user-controlled RunnableConfig.
        with tracing_context(enabled=False):
            result = await tool.ainvoke(checked, config={"callbacks": []})
        try:
            payload = output.dump_python(
                output.validate_python(result, strict=True), mode="json", warnings="error"
            )
        except (ValidationError, ValueError, TypeError) as exc:
            raise AgentError("tool_result", "tool result does not match declared type") from exc
        return result_schema.validate(
            payload, code="tool_result", max_bytes=declaration.result_bytes
        )

    return ToolBinding(declaration, handler)
