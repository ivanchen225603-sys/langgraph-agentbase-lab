"""LangChain provider 适配层：提供真实的原生流与离线预算估算。"""

import os
from collections.abc import AsyncGenerator, Mapping
from contextlib import aclosing
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langsmith import tracing_context
from pydantic import JsonValue

from .context import ContextProvider
from .contracts import (
    AgentError,
    ModelChunk,
    ModelRequest,
    ModelResponse,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
)
from .schema import canonical_json


class Utf8BudgetEstimator:
    """面向按字节计费 tokenizer 的显式离线上限估算，并非账单依据。

    统计完整规范化请求外加消息框架开销。相比 BPE 分词更严格，但无需隐式
    下载 tokenizer/模型。受信任的 provider 可以改用已固定版本且本地可用的
    tokenizer 来替换本估算器。
    """

    @staticmethod
    def input_tokens(request: ModelRequest) -> int:
        return (
            len(canonical_json(request.model_dump(mode="json"))) + 16 * len(request.messages) + 32
        )

    @staticmethod
    def output_tokens(text: str) -> int:
        return len(text.encode("utf-8"))


class LangChainProvider:
    def __init__(self, model: BaseChatModel, estimator: Utf8BudgetEstimator | None = None):
        self.model = model
        self.estimator = estimator or Utf8BudgetEstimator()
        # 否则 BaseChatModel.astream 会在无原生流时静默返回完整的 ainvoke 结果。
        self.native_stream = (
            type(model)._astream != BaseChatModel._astream and model.disable_streaming is False
        )

    def count_input_tokens(self, request: ModelRequest) -> int:
        return self.estimator.input_tokens(request)

    def count_output_tokens(self, text: str) -> int:
        return self.estimator.output_tokens(text)

    @staticmethod
    def messages(request: ModelRequest) -> list[BaseMessage]:
        remaining = (
            request.messages[1:] if request.messages[0].role == "system" else request.messages
        )
        if not ContextProvider._valid_message_sequence(remaining):
            raise AgentError("model_messages", "invalid model roles or tool observations")
        result: list[BaseMessage] = []
        for message in request.messages:
            if message.role == "system":
                result.append(SystemMessage(content=message.content))
            elif message.role == "user":
                result.append(HumanMessage(content=message.content))
            elif message.role == "tool":
                result.append(
                    ToolMessage(content=message.content, tool_call_id=message.tool_call_id or "")
                )
            else:
                result.append(
                    AIMessage(
                        content=message.content,
                        tool_calls=[
                            {
                                "id": call.call_id,
                                "name": call.name,
                                "args": call.arguments,
                                "type": "tool_call",
                            }
                            for call in message.tool_calls
                        ],
                    )
                )
        return result

    @staticmethod
    def _text(content: str | list) -> str:
        if isinstance(content, str):
            return content
        if all(
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            for block in content
        ):
            return "".join(block["text"] for block in content)
        raise AgentError("model_content", "provider returned unsupported content")

    @staticmethod
    def _usage(message: AIMessage) -> TokenUsage | None:
        usage = message.usage_metadata
        return (
            TokenUsage(input_tokens=usage["input_tokens"], output_tokens=usage["output_tokens"])
            if usage is not None
            else None
        )

    @staticmethod
    def _options(request: ModelRequest) -> dict[str, Any]:
        options: dict[str, Any] = {"max_tokens": request.max_output_tokens}
        if request.tools:
            options["tools"] = request.tools
        if request.output_schema is not None:
            # tool 轮次允许合法地不含最终 JSON 文本。终态 Schema 校验在本地完成；
            # JSON 模式仅适用于不带 tool 的调用。
            if not request.tools:
                options["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "agent_output",
                        "schema": request.output_schema,
                        "strict": False,
                    },
                }
        return options

    async def complete(self, request: ModelRequest) -> ModelResponse:
        try:
            with tracing_context(enabled=False):
                message = await self.model.ainvoke(
                    self.messages(request),
                    config=RunnableConfig(callbacks=[]),
                    **self._options(request),
                )
            if not isinstance(message, AIMessage) or message.invalid_tool_calls:
                raise AgentError("model_tool_calls", "provider returned invalid tool calls")
            raw_calls = message.additional_kwargs.get("tool_calls")
            if raw_calls is not None:
                from .model_calls import strict_json

                canonical_json(raw_calls)
                if not isinstance(raw_calls, list):
                    raise AgentError(
                        "model_tool_calls", "provider returned invalid raw tool calls", 502
                    )
                calls = []
                for call in raw_calls:
                    arguments = strict_json(call["function"]["arguments"])
                    if not isinstance(arguments, dict):
                        raise AgentError(
                            "model_tool_calls", "tool arguments must be an object", 502
                        )
                    calls.append(
                        ToolCall(
                            call_id=call["id"],
                            name=call["function"]["name"],
                            arguments=arguments,
                        )
                    )
            else:
                calls = [
                    ToolCall(
                        call_id=call.get("id") or "",
                        name=call["name"],
                        arguments=call["args"],
                    )
                    for call in message.tool_calls
                ]
            return ModelResponse(
                text=self._text(message.content),
                tool_calls=calls,
                usage=self._usage(message),
                finish_reason=message.response_metadata.get("finish_reason"),
            )
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError("model_provider", "model provider request failed", 502) from exc

    async def stream(self, request: ModelRequest) -> AsyncGenerator[ModelChunk, None]:
        if not self.native_stream:
            raise AgentError("native_stream", "model has no enabled native async stream", 409)
        try:
            with tracing_context(enabled=False):
                async with aclosing(
                    cast(
                        Any,
                        self.model.astream(
                            self.messages(request),
                            config=RunnableConfig(callbacks=[]),
                            **self._options(request),
                        ),
                    )
                ) as chunks:
                    async for message in chunks:
                        if not isinstance(message, AIMessageChunk):
                            raise AgentError(
                                "model_chunk", "provider returned an invalid stream chunk"
                            )
                        # LC 的部分 JSON 解析绝不能变成可执行的 tool 调用。
                        if any(call.get("index") is None for call in message.tool_call_chunks):
                            raise AgentError(
                                "model_tool_calls", "provider omitted tool fragment index", 502
                            )
                        deltas = [
                            ToolCallDelta(
                                index=cast(int, call.get("index")),
                                call_id=call.get("id"),
                                name=call.get("name"),
                                arguments_delta=call.get("args") or "",
                            )
                            for call in message.tool_call_chunks
                        ]
                        yield ModelChunk(
                            text_delta=self._text(message.content),
                            tool_call_deltas=deltas,
                            usage=self._usage(message),
                            finish_reason=message.response_metadata.get("finish_reason"),
                        )
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError("model_provider", "model provider stream failed", 502) from exc


def openai_factory(configuration: Mapping[str, JsonValue]) -> LangChainProvider:
    """受信任的启动期绑定。构造过程不会发起任何 provider/tokenizer 请求。"""
    from langchain_openai import ChatOpenAI

    options = configuration.get("options") or {}
    if not isinstance(options, dict) or set(options) - {"temperature", "base_url"}:
        raise AgentError("model_options", "unsupported OpenAI provider options")
    temperature = options.get("temperature")
    if temperature is not None and (
        not isinstance(temperature, (int, float))
        or isinstance(temperature, bool)
        or not 0 <= temperature <= 2
    ):
        raise AgentError("model_options", "invalid model temperature")
    base_url = options.get("base_url")
    if base_url is not None and (
        not isinstance(base_url, str) or not base_url.startswith(("https://", "http://"))
    ):
        raise AgentError("model_options", "invalid provider base URL")
    credential_env = configuration.get("credential_env")
    key = os.environ.get(credential_env, "") if isinstance(credential_env, str) else ""
    if not key:
        raise AgentError("model_credentials", "configured model credential is unavailable", 503)
    try:
        return LangChainProvider(
            ChatOpenAI(
                model=str(configuration["model"]),
                api_key=key,
                temperature=temperature,
                base_url=base_url,
                max_retries=0,
                timeout=30.0,
                streaming=True,
                stream_usage=True,
            )
        )
    except Exception as exc:
        raise AgentError("model_configuration", "model provider configuration is invalid") from exc
