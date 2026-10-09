"""与业务无关的消息、provider 流、Tool 以及可信的 Run 上下文。"""

from collections.abc import AsyncGenerator, Mapping
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
ComponentName = Annotated[str, Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]{0,63}$")]
Version = Annotated[str, Field(pattern=r"^[A-Za-z0-9_.-]{1,64}$")]


class BaseContract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class AgentError(Exception):
    """对外暴露的错误码（public error code）；message 绝不能包含凭据或原始模型数据。"""

    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


class Identity(BaseContract):
    """由服务端认证创建，绝不采信来自用户消息或配置的输入。"""

    owner: Identifier
    scopes: frozenset[str] = frozenset()
    permission_version: int = Field(1, ge=1)


class ToolCall(BaseContract):
    """一次完整的调用，而非片段；执行仍然需要通过 Schema 校验与授权检查。"""

    call_id: str = Field(min_length=1, max_length=128)
    name: ComponentName
    arguments: dict[str, JsonValue]


class Message(BaseContract):
    role: Literal["system", "user", "assistant", "tool"]
    content: str = Field("", max_length=262144)
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=16)
    tool_call_id: str | None = Field(None, max_length=128)
    sources: list[dict[str, JsonValue]] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def role_contract(self):
        if self.role != "assistant" and self.tool_calls:
            raise ValueError("only assistant messages can declare tool calls")
        if (self.role == "tool") != (self.tool_call_id is not None):
            raise ValueError("tool messages require a call ID; other roles cannot supply one")
        return self


class TokenUsage(BaseContract):
    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)


class ToolCallDelta(BaseContract):
    """provider 返回的片段必须先组装完整，才能构造可执行的 ToolCall。"""

    index: int = Field(ge=0, le=15)
    call_id: str | None = Field(None, max_length=128)
    name: str | None = Field(None, max_length=64)
    arguments_delta: str = Field("", max_length=65536)


class ModelChunk(BaseContract):
    text_delta: str = Field("", max_length=65536)
    tool_call_deltas: list[ToolCallDelta] = Field(default_factory=list, max_length=16)
    usage: TokenUsage | None = None  # 累计快照，绝非可加的增量值。
    finish_reason: str | None = Field(None, max_length=64)


class ModelResponse(BaseContract):
    text: str = Field("", max_length=262144)
    tool_calls: list[ToolCall] = Field(default_factory=list, max_length=16)
    usage: TokenUsage | None = None  # usage 缺失表示未知，而非声称零成本。
    finish_reason: str | None = Field(None, max_length=64)

    @model_validator(mode="after")
    def unique_calls(self):
        ids = [call.call_id for call in self.tool_calls]
        if len(ids) != len(set(ids)):
            raise ValueError("model tool call IDs must be unique")
        return self


class ModelRequest(BaseContract):
    messages: list[Message] = Field(min_length=1, max_length=100)
    tools: list[dict[str, JsonValue]] = Field(default_factory=list, max_length=64)
    output_schema: dict[str, JsonValue] | None = None
    max_output_tokens: int = Field(512, ge=1, le=32768)


class RunContext(BaseContract):
    identity: Identity
    agent_id: Identifier
    run_id: Identifier
    conversation_id: Identifier
    epoch: int = Field(0, ge=0)
    deadline: float
    config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    operation_key: str | None = Field(None, max_length=256)
    attempt_token: str | None = Field(None, max_length=64)
    capabilities: dict[str, JsonValue] = Field(default_factory=dict)
    dependency_results: dict[str, JsonValue] = Field(default_factory=dict)
    scope_ceiling: frozenset[str] | None = Field(None, exclude_if=lambda value: value is None)

    @property
    def effective_scopes(self) -> frozenset[str]:
        return (
            self.identity.scopes
            if self.scope_ceiling is None
            else self.identity.scopes & self.scope_ceiling
        )

    @property
    def access_identity(self) -> Identity:
        return self.identity.model_copy(update={"scopes": self.effective_scopes})


class RunEvent(BaseContract):
    run_id: Identifier
    agent_id: Identifier
    seq: int = Field(ge=1)
    kind: ComponentName
    created_at: float
    data: JsonValue = None
    provisional: bool = False


class ModelProvider(Protocol):
    native_stream: bool

    async def complete(self, request: ModelRequest) -> ModelResponse: ...

    def stream(self, request: ModelRequest) -> AsyncGenerator[ModelChunk, None]: ...

    def count_input_tokens(self, request: ModelRequest) -> int: ...

    def count_output_tokens(self, text: str) -> int: ...


class ToolHandler(Protocol):
    async def __call__(self, arguments: dict[str, JsonValue], context: RunContext) -> JsonValue: ...


class ProviderFactory(Protocol):
    def __call__(self, configuration: Mapping[str, JsonValue]) -> ModelProvider: ...


class MemoryRecord(BaseContract):
    id: Identifier
    owner: Identifier
    agent_id: Identifier
    conversation_id: str = Field("", max_length=64)
    content: str = Field(min_length=1, max_length=2000)
    version: int = Field(1, ge=1)
    importance: float = Field(0.5, ge=0, le=1)
    enabled: bool = True
    valid_until: float | None = None
    created_at: float
    updated_at: float
    sources: list[dict[str, JsonValue]] = Field(default_factory=list, max_length=100)
