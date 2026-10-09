"""单一预算内的 provider 调用路径：分块原生流、完整调用与终态校验。"""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from contextlib import aclosing

from pydantic import JsonValue

from .budgets import BudgetLedger
from .configuration import BudgetPolicy
from .contracts import (
    AgentError,
    ModelChunk,
    ModelProvider,
    ModelRequest,
    ModelResponse,
    RunContext,
    TokenUsage,
    ToolCall,
)
from .schema import SchemaContract, canonical_json

ChunkSink = Callable[[ModelChunk], Awaitable[None]]


def strict_json(raw: str) -> JsonValue:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate object key")
            result[key] = value
        return result

    try:
        result = json.loads(raw, object_pairs_hook=unique)
        canonical_json(result)
        return result
    except (ValueError, TypeError, RecursionError) as exc:
        raise AgentError("model_json", "model returned invalid complete JSON", 502) from exc


class ChunkAccumulator:
    def __init__(self, max_chunks: int = 4096):
        self.text = ""
        self.calls: dict[int, dict[str, str]] = {}
        self.usage: TokenUsage | None = None
        self.finish_reason: str | None = None
        self.max_chunks, self.chunk_count = max_chunks, 0

    def add(self, chunk: ModelChunk):
        self.chunk_count += 1
        if self.chunk_count > self.max_chunks:
            raise AgentError("model_chunk_limit", "provider stream chunk limit exceeded", 502)
        if self.finish_reason is not None and (chunk.text_delta or chunk.tool_call_deltas):
            raise AgentError("model_stream", "provider continued output after finish", 502)
        self.text += chunk.text_delta
        if len(self.text.encode()) > 262144:
            raise AgentError("model_output_limit", "model text byte limit exceeded", 502)
        for delta in chunk.tool_call_deltas:
            call = self.calls.setdefault(delta.index, {"id": "", "name": "", "arguments": ""})
            if delta.call_id:
                if call["id"] and call["id"] != delta.call_id:
                    raise AgentError("model_tool_calls", "provider changed a tool call ID", 502)
                call["id"] = delta.call_id
            if delta.name:
                if call["name"] and call["name"] != delta.name:
                    raise AgentError("model_tool_calls", "provider changed a tool name", 502)
                call["name"] = delta.name
            call["arguments"] += delta.arguments_delta
            if len(call["arguments"].encode()) > 65536:
                raise AgentError("model_output_limit", "tool arguments byte limit exceeded", 502)
        canonical_json(self.calls_as_json(), max_bytes=262144)
        if chunk.usage is not None:
            if self.usage is not None and (
                chunk.usage.input_tokens < self.usage.input_tokens
                or chunk.usage.output_tokens < self.usage.output_tokens
            ):
                raise AgentError("model_usage", "provider usage snapshots decreased", 502)
            self.usage = chunk.usage
        if chunk.finish_reason is not None:
            if self.finish_reason is not None and chunk.finish_reason != self.finish_reason:
                raise AgentError("model_stream", "provider changed finish reason", 502)
            self.finish_reason = chunk.finish_reason

    def calls_as_json(self):
        return {str(index): call for index, call in self.calls.items()}

    def output_text(self) -> str:
        return self.text + "".join(call["name"] + call["arguments"] for call in self.calls.values())

    def response(self) -> ModelResponse:
        if self.finish_reason not in {"stop", "tool_calls", "function_call"}:
            raise AgentError("model_finish", "model stream did not finish successfully", 502)
        if self.calls and self.finish_reason not in {"tool_calls", "function_call"}:
            raise AgentError("model_tool_calls", "tool calls require a successful tool finish", 502)
        if not self.calls and self.finish_reason in {"tool_calls", "function_call"}:
            raise AgentError("model_tool_calls", "provider omitted complete tool calls", 502)
        calls = []
        for index in sorted(self.calls):
            call = self.calls[index]
            arguments = strict_json(call["arguments"])
            if not isinstance(arguments, dict):
                raise AgentError("model_tool_calls", "tool arguments must be an object", 502)
            try:
                calls.append(ToolCall(call_id=call["id"], name=call["name"], arguments=arguments))
            except ValueError as exc:
                raise AgentError("model_tool_calls", "incomplete model tool call", 502) from exc
        try:
            return ModelResponse(
                text=self.text, tool_calls=calls, usage=self.usage, finish_reason=self.finish_reason
            )
        except ValueError as exc:
            raise AgentError(
                "model_tool_calls", "provider returned conflicting tool calls", 502
            ) from exc


class ModelCaller:
    def __init__(
        self,
        ledger: BudgetLedger,
        *,
        control_guard: Callable[[RunContext], bool] | None = None,
        authorization_guard: Callable[[RunContext, ModelRequest], Awaitable[bool]] | None = None,
        request_control_guard: Callable[[RunContext, ModelRequest], bool] | None = None,
    ):
        self.ledger = ledger
        self.control_guard = control_guard
        self.authorization_guard = authorization_guard
        self.request_control_guard = request_control_guard

    async def call(
        self,
        provider: ModelProvider,
        context: RunContext,
        policy: BudgetPolicy,
        request: ModelRequest,
        *,
        sink: ChunkSink | None = None,
        provisional: bool = True,
        streaming: bool = True,
        request_guard: Callable[[], bool] | None = None,
    ) -> ModelResponse:
        if request.max_output_tokens != policy.max_output_tokens:
            raise AgentError("model_budget_binding", "request output limit differs from run policy")
        if streaming and not provider.native_stream:
            raise AgentError("native_stream", "provider has no native stream", 409)
        if len(canonical_json(request.model_dump(mode="json")).decode()) > policy.max_input_chars:
            raise AgentError("prompt_limit", "complete model request exceeds character limit", 409)
        if self.authorization_guard is not None:
            try:
                async with asyncio.timeout(max(0.0, context.deadline - time.time())):
                    if await self.authorization_guard(context, request) is not True:
                        raise AgentError("source_revoked", "model request is not authorized", 403)
            except TimeoutError as exc:
                raise AgentError(
                    "model_timeout", "model authorization deadline expired", 504
                ) from exc
        try:
            estimate = provider.count_input_tokens(request)
        except Exception as exc:
            raise AgentError(
                "token_count", "complete model input cannot be estimated", 409
            ) from exc
        reservation = self.ledger.reserve(context, policy, estimate)
        try:
            async with asyncio.timeout(max(0.0, context.deadline - time.time())):
                # After the final identity/source awaits and token/budget preparation, before
                # any data crosses the provider boundary. The guard must be synchronous.
                if (
                    (self.control_guard is not None and self.control_guard(context) is not True)
                    or (
                        self.request_control_guard is not None
                        and self.request_control_guard(context, request) is not True
                    )
                    or (request_guard is not None and request_guard() is not True)
                ):
                    raise AgentError("source_revoked", "model request authorization changed", 403)
                if streaming:
                    accumulated = ChunkAccumulator(policy.max_stream_chunks)
                    async with aclosing(provider.stream(request)) as stream:
                        async for chunk in stream:
                            if not isinstance(chunk, ModelChunk):
                                raise AgentError("model_chunk", "invalid provider chunk", 502)
                            accumulated.add(chunk)
                            if (
                                self._output_estimate(provider, accumulated.output_text())
                                > policy.max_output_tokens
                            ):
                                raise AgentError(
                                    "model_output_budget", "model output exceeds budget", 502
                                )
                            if sink is not None and provisional and chunk.text_delta:
                                # 绝不下发未成型的 tool 参数片段。
                                await sink(ModelChunk(text_delta=chunk.text_delta))
                    response = accumulated.response()
                else:
                    response = await provider.complete(request)
                    self._successful_finish(response)
                    output = response.text + "".join(
                        call.name + canonical_json(call.arguments).decode()
                        for call in response.tool_calls
                    )
                    if self._output_estimate(provider, output) > policy.max_output_tokens:
                        raise AgentError("model_output_budget", "model output exceeds budget", 502)
                if not self.ledger.settle(reservation, response.usage):
                    raise AgentError("model_usage_overrun", "provider exceeded reserved usage", 502)
                return response
        except asyncio.CancelledError:
            raise
        except TimeoutError as exc:
            raise AgentError("model_timeout", "model execution deadline expired", 504) from exc
        except AgentError:
            raise
        except Exception as exc:
            raise AgentError("model_provider", "model call failed", 502) from exc

    @staticmethod
    def _output_estimate(provider: ModelProvider, text: str) -> int:
        value = provider.count_output_tokens(text)
        if type(value) is not int or value < 0:
            raise AgentError("token_count", "output estimate must be a nonnegative integer", 502)
        return value

    @staticmethod
    def _successful_finish(response: ModelResponse):
        if response.finish_reason not in {"stop", "tool_calls", "function_call"}:
            raise AgentError("model_finish", "model response did not finish successfully", 502)
        if response.tool_calls and response.finish_reason not in {"tool_calls", "function_call"}:
            raise AgentError("model_tool_calls", "invalid complete tool finish", 502)
        if not response.tool_calls and response.finish_reason in {"tool_calls", "function_call"}:
            raise AgentError("model_tool_calls", "provider omitted complete tool calls", 502)

    @staticmethod
    def final_value(response: ModelResponse, output_schema: dict | None) -> JsonValue:
        if response.tool_calls:
            raise AgentError(
                "model_pending_tools", "model response still requires tool observations", 409
            )
        ModelCaller._successful_finish(response)
        value = strict_json(response.text) if output_schema is not None else response.text
        if output_schema is not None:
            SchemaContract(output_schema).validate(value, code="agent_output")
        return value
