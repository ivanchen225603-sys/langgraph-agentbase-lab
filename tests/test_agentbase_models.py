"""Offline model protocol, LangChain stream, reservations and final-output tests."""

import asyncio
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

from agentbase.budgets import SqliteBudgetLedger
from agentbase.configuration import BudgetPolicy
from agentbase.contracts import (
    AgentError,
    Identity,
    Message,
    ModelChunk,
    ModelRequest,
    ModelResponse,
    RunContext,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
)
from agentbase.model_calls import ChunkAccumulator, ModelCaller, strict_json
from agentbase.providers import LangChainProvider, Utf8BudgetEstimator, openai_factory
from agentbase.storage import SqliteDatabase


@pytest.fixture
def ledger(tmp_path):
    database = SqliteDatabase(tmp_path / "model.sqlite")
    result = SqliteBudgetLedger(database.connection)
    assert not database.path.exists()
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        result.install(db)
    return result


def context(run="run1", *, timeout=10.0):
    return RunContext(
        identity=Identity(owner="user"),
        agent_id="alpha",
        run_id=run,
        conversation_id="same",
        deadline=time.time() + timeout,
        config_digest="sha256:" + "0" * 64,
    )


def request(**changes):
    return ModelRequest(
        messages=[Message(role="system", content="Policy"), Message(role="user", content="Input")],
        **changes,
    )


class NativeChat(BaseChatModel):
    _release: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _closed: bool = PrivateAttr(default=False)
    _closed_event: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _seen: list = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self):
        return "offline-native-test"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._seen.append((messages, kwargs))
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="complete", response_metadata={"finish_reason": "stop"}
                    )
                )
            ]
        )

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        self._seen.append((messages, kwargs))
        try:
            yield ChatGenerationChunk(message=AIMessageChunk(content="first"))
            await self._release.wait()
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=" second", response_metadata={"finish_reason": "stop"}
                )
            )
        finally:
            self._closed = True
            self._closed_event.set()


class InvokeOnly(BaseChatModel):
    @property
    def _llm_type(self):
        return "offline-invoke-only"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="completed", response_metadata={"finish_reason": "stop"}
                    )
                )
            ]
        )


class ScriptProvider:
    native_stream = True

    def __init__(self, chunks):
        self.chunks, self.closed = chunks, False

    count_input_tokens = staticmethod(Utf8BudgetEstimator.input_tokens)
    count_output_tokens = staticmethod(Utf8BudgetEstimator.output_tokens)

    async def complete(self, request):
        return ModelResponse(text="complete", finish_reason="stop")

    async def stream(self, request):
        try:
            for chunk in self.chunks:
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
        finally:
            self.closed = True


async def test_langchain_native_delta_received_before_provider_finishes(ledger):
    chat = NativeChat()
    provider = LangChainProvider(chat)
    assert provider.native_stream
    received = asyncio.Event()
    data = []

    async def sink(chunk):
        data.append(chunk.text_delta)
        received.set()

    pending = asyncio.create_task(
        ModelCaller(ledger).call(provider, context(), BudgetPolicy(), request(), sink=sink)
    )
    try:
        await asyncio.wait_for(received.wait(), 2)
        assert data == ["first"] and not pending.done() and not chat._release.is_set()
        chat._release.set()
        response = await pending
        assert response.text == "first second" and chat._closed
        assert response.usage is None  # Unknown usage is never represented as zero.
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)


async def test_invoke_only_model_cannot_masquerade_as_native_stream(ledger):
    provider = LangChainProvider(InvokeOnly())
    assert not provider.native_stream
    with pytest.raises(AgentError) as error:
        await ModelCaller(ledger).call(provider, context(), BudgetPolicy(), request())
    assert error.value.code == "native_stream" and ledger.snapshot("run1")["calls"] == 0
    response = await ModelCaller(ledger).call(
        provider, context(), BudgetPolicy(), request(), streaming=False
    )
    assert response.text == "completed"


def test_message_conversion_preserves_actual_tool_observations_and_rejects_orphans():
    messages = [
        Message(role="system", content="policy"),
        Message(role="user", content="input"),
        Message(
            role="assistant",
            tool_calls=[ToolCall(call_id="call1", name="lookup", arguments={"x": 1})],
        ),
        Message(role="tool", tool_call_id="call1", content='{"observed":true}'),
    ]
    converted = LangChainProvider.messages(ModelRequest(messages=messages))
    assert converted[2].tool_calls[0]["id"] == "call1"
    assert isinstance(converted[3], ToolMessage) and converted[3].tool_call_id == "call1"
    for invalid in (
        [Message(role="tool", tool_call_id="invented", content="fake")],
        [Message(role="user", content="input"), Message(role="system", content="hijack")],
    ):
        with pytest.raises(AgentError):
            LangChainProvider.messages(ModelRequest(messages=invalid))


def test_complete_calls_only_after_successful_finish_and_interleaved_json_fragments():
    accumulator = ChunkAccumulator()
    accumulator.add(
        ModelChunk(
            tool_call_deltas=[
                ToolCallDelta(index=0, call_id="a", name="lookup", arguments_delta='{"x":'),
                ToolCallDelta(index=1, call_id="b", name="lookup", arguments_delta='{"x":2'),
            ]
        )
    )
    with pytest.raises(AgentError, match="finish"):
        accumulator.response()
    accumulator.add(
        ModelChunk(
            tool_call_deltas=[
                ToolCallDelta(index=1, arguments_delta="}"),
                ToolCallDelta(index=0, arguments_delta="1}"),
            ],
            finish_reason="tool_calls",
        )
    )
    response = accumulator.response()
    assert [(c.call_id, c.arguments) for c in response.tool_calls] == [
        ("a", {"x": 1}),
        ("b", {"x": 2}),
    ]
    with pytest.raises(AgentError, match="observations"):
        ModelCaller.final_value(response, {"type": "object"})


@pytest.mark.parametrize(
    "arguments", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', "[1]", '{"x":']
)
def test_invalid_or_incomplete_argument_json_never_creates_executable_call(arguments):
    accumulator = ChunkAccumulator()
    accumulator.add(
        ModelChunk(
            tool_call_deltas=[
                ToolCallDelta(index=0, call_id="a", name="lookup", arguments_delta=arguments)
            ],
            finish_reason="tool_calls",
        )
    )
    with pytest.raises(AgentError):
        accumulator.response()


@pytest.mark.parametrize("finish", [None, "length", "content_filter", "error"])
def test_unsuccessful_finish_never_produces_final_or_tool_call(finish):
    accumulator = ChunkAccumulator()
    accumulator.add(ModelChunk(text_delta='{"x":1}', finish_reason=finish))
    with pytest.raises(AgentError):
        accumulator.response()


def test_tool_fragment_conflicts_sizes_and_usage_snapshot_contract():
    accumulator = ChunkAccumulator()
    accumulator.add(
        ModelChunk(
            tool_call_deltas=[
                ToolCallDelta(index=0, call_id="a", name="lookup", arguments_delta="{")
            ]
        )
    )
    with pytest.raises(AgentError, match="ID"):
        accumulator.add(ModelChunk(tool_call_deltas=[ToolCallDelta(index=0, call_id="b")]))
    with pytest.raises(AgentError, match="name"):
        accumulator.add(ModelChunk(tool_call_deltas=[ToolCallDelta(index=0, name="other")]))
    oversized = ChunkAccumulator()
    with pytest.raises(AgentError, match="byte"):
        oversized.add(
            ModelChunk(tool_call_deltas=[ToolCallDelta(index=0, arguments_delta="中" * 30000)])
        )
    usage = ChunkAccumulator()
    usage.add(ModelChunk(text_delta="a", usage=TokenUsage(input_tokens=10, output_tokens=1)))
    usage.add(
        ModelChunk(
            text_delta="b", usage=TokenUsage(input_tokens=10, output_tokens=2), finish_reason="stop"
        )
    )
    assert usage.response().usage == TokenUsage(input_tokens=10, output_tokens=2)
    with pytest.raises(AgentError, match="decreased"):
        usage.add(ModelChunk(usage=TokenUsage(input_tokens=9, output_tokens=1)))


def test_persistent_atomic_run_reservations_and_unknown_or_overrun_usage(ledger):
    trusted_context = context()
    policy = BudgetPolicy(max_model_calls=2, max_run_tokens=1024, max_output_tokens=128)
    with ThreadPoolExecutor(max_workers=3) as workers:

        def reserve(_):
            try:
                return ledger.reserve(trusted_context, policy, 100)
            except AgentError as exc:
                return exc.code

        results = list(workers.map(reserve, range(3)))
    assert results.count("model_run_budget") == 1
    assert ledger.snapshot("run1") == {"calls": 2, "tokens": 456, "overrun": 0}
    restarted = SqliteBudgetLedger(ledger.connection)
    with pytest.raises(AgentError, match="exhausted"):
        restarted.reserve(trusted_context, policy, 100)
    receipts = [result for result in results if result != "model_run_budget"]
    assert restarted.settle(receipts[0], None)
    assert not restarted.settle(receipts[1], TokenUsage(input_tokens=101, output_tokens=1))
    assert restarted.snapshot("run1")["overrun"] == 1
    with pytest.raises(AgentError, match="changed"):
        restarted.settle(receipts[0], TokenUsage(input_tokens=1))
    with pytest.raises(AgentError, match="binding"):
        restarted.reserve(trusted_context.model_copy(update={"agent_id": "beta"}), policy, 100)
    paused = context("paused")
    restarted.reserve(paused, policy, 100)
    resumed = paused.model_copy(update={"deadline": paused.deadline + 3600})
    restarted.reserve(resumed, policy, 100)
    with pytest.raises(AgentError, match="exhausted"):
        restarted.reserve(resumed, policy, 100)
    assert restarted.snapshot("paused")["calls"] == 2


async def test_full_request_including_tools_and_output_schema_is_counted(ledger):
    provider = ScriptProvider([ModelChunk(text_delta="ok", finish_reason="stop")])
    caller = ModelCaller(ledger)
    tiny = request()
    policy = BudgetPolicy(max_input_tokens=Utf8BudgetEstimator.input_tokens(tiny) + 1)
    for data in (
        request(tools=[{"description": "long tool description " * 100}]),
        request(output_schema={"type": "object", "description": "long output contract " * 100}),
    ):
        with pytest.raises(AgentError) as error:
            await caller.call(provider, context(), policy, data)
        assert error.value.code == "model_input_budget"
    assert ledger.snapshot("run1")["calls"] == 0


async def test_failed_stream_keeps_allowance_and_hides_provider_exception(ledger):
    provider = ScriptProvider([RuntimeError("private-secret and raw payload")])
    trusted_context, policy = context(), BudgetPolicy(max_model_calls=1)
    with pytest.raises(AgentError) as error:
        await ModelCaller(ledger).call(provider, trusted_context, policy, request())
    assert "private-secret" not in str(error.value) and provider.closed
    assert ledger.snapshot("run1")["calls"] == 1
    with pytest.raises(AgentError, match="exhausted"):
        await ModelCaller(SqliteBudgetLedger(ledger.connection)).call(
            provider, trusted_context, policy, request()
        )


async def test_final_only_never_emits_raw_and_cancel_or_slow_sink_closes_stream(ledger):
    received = []

    async def sink(chunk):
        received.append(chunk.text_delta)

    provider = ScriptProvider([ModelChunk(text_delta='{"x":1}', finish_reason="stop")])
    response = await ModelCaller(ledger).call(
        provider, context(), BudgetPolicy(), request(), sink=sink, provisional=False
    )
    assert received == []
    assert ModelCaller.final_value(response, {"type": "object", "required": ["x"]}) == {"x": 1}
    with pytest.raises(AgentError):
        ModelCaller.final_value(
            ModelResponse(text='{"wrong":1}', finish_reason="stop"),
            {"type": "object", "required": ["x"]},
        )
    for action in ("timeout", "cancel"):
        chat = NativeChat()
        entered = asyncio.Event()

        async def slow_sink(chunk, entered=entered):
            entered.set()
            await asyncio.Event().wait()

        pending = asyncio.create_task(
            ModelCaller(ledger).call(
                LangChainProvider(chat),
                context(action, timeout=0.2),
                BudgetPolicy(),
                request(),
                sink=slow_sink,
            )
        )
        await asyncio.wait_for(entered.wait(), 2)
        if action == "cancel":
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        else:
            with pytest.raises(AgentError) as error:
                await pending
            assert error.value.code == "model_timeout"

        await asyncio.wait_for(chat._closed_event.wait(), 0.5)
        assert chat._closed


def test_openai_startup_and_counting_are_offline_options_and_secrets_are_bounded(monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("startup attempted network access")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setenv("TEST_MODEL_KEY", "private-startup-secret")
    provider = openai_factory({"model": "configured-model", "credential_env": "TEST_MODEL_KEY"})
    assert provider.native_stream and provider.model.max_retries == 0
    assert provider.count_input_tokens(request()) > 0
    for configuration in (
        {"model": "configured-model", "credential_env": "MISSING_TEST_KEY"},
        {
            "model": "configured-model",
            "credential_env": "TEST_MODEL_KEY",
            "options": {"max_retries": 5},
        },
    ):
        with pytest.raises(AgentError) as error:
            openai_factory(configuration)
        assert "private-startup-secret" not in str(error.value)


def test_strict_final_json_rejects_nonfinite_and_duplicate_objects():
    for value in ('{"x":NaN}', '{"x":1,"x":2}'):
        with pytest.raises(AgentError):
            strict_json(value)


def test_empty_tool_finish_and_excess_empty_chunks_are_rejected():
    accumulator = ChunkAccumulator(max_chunks=2)
    accumulator.add(ModelChunk())
    accumulator.add(ModelChunk())
    with pytest.raises(AgentError, match="chunk limit"):
        accumulator.add(ModelChunk())
    empty_tools = ChunkAccumulator()
    empty_tools.add(ModelChunk(finish_reason="tool_calls"))
    with pytest.raises(AgentError, match="omitted"):
        empty_tools.response()


@pytest.mark.parametrize("invalid", [-1, True, "unknown"])
async def test_invalid_output_estimator_never_bypasses_budget(ledger, invalid):
    provider = ScriptProvider([ModelChunk(text_delta="x", finish_reason="stop")])
    provider.count_output_tokens = lambda text: invalid
    with pytest.raises(AgentError) as error:
        await ModelCaller(ledger).call(provider, context(), BudgetPolicy(), request())
    assert error.value.code == "token_count" and provider.closed


@pytest.mark.parametrize("arguments", ['{"x":1,"x":2}', '{"x":NaN}', "[1]"])
async def test_complete_langchain_adapter_validates_original_raw_tool_json(arguments):
    class RawChat(InvokeOnly):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            additional_kwargs={
                                "tool_calls": [
                                    {
                                        "id": "raw",
                                        "type": "function",
                                        "function": {"name": "lookup", "arguments": arguments},
                                    }
                                ]
                            },
                            response_metadata={"finish_reason": "tool_calls"},
                        )
                    )
                ]
            )

    with pytest.raises(AgentError):
        await LangChainProvider(RawChat()).complete(request())


async def test_openai_wire_stream_and_fragmented_tool_arguments_with_mock_transport(ledger):
    release, observed = asyncio.Event(), asyncio.Event()
    sent = []

    class WireStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            def data(delta, finish=None):
                return (
                    "data: "
                    + json.dumps(
                        {
                            "id": "test-chat",
                            "model": "gpt-4o-mini",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                        }
                    )
                    + "\n\n"
                ).encode()

            yield data({"role": "assistant", "content": "first"})
            await release.wait()
            yield data(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "wire-call",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"x":'},
                        }
                    ]
                }
            )
            yield data(
                {"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}, "tool_calls"
            )
            yield b'data: {"id":"test-chat","model":"gpt-4o-mini","object":"chat.completion.chunk","created":1,"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":8,"total_tokens":18}}\n\n'
            yield b"data: [DONE]\n\n"

    async def handler(http_request):
        sent.append(json.loads(http_request.content))
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=WireStream()
        )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Synthetic lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            },
        }
    ]
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        chat = ChatOpenAI(
            model="gpt-4o-mini",
            api_key="offline-test-key",
            base_url="https://offline.invalid/v1",
            max_retries=0,
            streaming=True,
            stream_usage=True,
            http_async_client=http_client,
        )

        async def sink(chunk):
            assert not chunk.tool_call_deltas
            observed.set()

        pending = asyncio.create_task(
            ModelCaller(ledger).call(
                LangChainProvider(chat), context(), BudgetPolicy(), request(tools=tools), sink=sink
            )
        )
        try:
            await asyncio.wait_for(observed.wait(), 2)
            assert not pending.done() and not release.is_set()
            assert sent[0]["stream"] is True and sent[0]["tools"] == tools
            release.set()
            response = await pending
            assert response.tool_calls == [
                ToolCall(call_id="wire-call", name="lookup", arguments={"x": 1})
            ]
            assert response.usage == TokenUsage(input_tokens=10, output_tokens=8)
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
