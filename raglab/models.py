import operator
import re
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator

ID_PATTERN = r"^[A-Za-z0-9_-]{1,64}$"
Identifier = Annotated[str, Field(pattern=ID_PATTERN)]
Question = Annotated[str, Field(min_length=1, max_length=1000)]


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid", str_strip_whitespace=True, strict=True, allow_inf_nan=False
    )
    schema_version: Literal[1] = 1


class Principal(BaseModel):
    owner: Identifier
    scopes: frozenset[str] = frozenset({"read", "memory:write", "index:write"})
    permission_version: int = 1


class DocumentRecord(StrictModel):
    document_id: str
    source: str
    title: str
    content: str = Field(max_length=262144)
    digest: str
    owner: str = "public"
    valid_until: float | None = None


class Chunk(StrictModel):
    chunk_id: str
    document_id: str
    source: str
    title: str
    text: str = Field(max_length=4000)
    start: int
    end: int
    digest: str
    index_version: str = "candidate"


class IndexManifest(StrictModel):
    owner: str = "public"
    version: str
    created_at: float
    embedding: str
    dimensions: int
    splitter: str = "recursive-char-v1"
    chunk_size: int
    overlap: int
    documents: int
    chunks: int
    reused_vectors: int
    content_digest: str


class RetrievalHit(StrictModel):
    chunk: Chunk
    vector_score: float | None = None
    keyword_score: float | None = None
    fusion_score: float = 0
    rerank_score: float = 0
    relevance: float = 0
    channels: list[str] = Field(default_factory=list)


class Citation(StrictModel):
    chunk_id: str
    document_id: str
    source: str
    index_version: str
    quote: str = Field(max_length=2000)


class AnswerResult(StrictModel):
    status: Literal["answered", "refused", "clarify"]
    answer: str
    question: str
    resolved_question: str
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[RetrievalHit] = Field(default_factory=list)
    degradation: list[str] = Field(default_factory=list)
    reason: str | None = None
    trace_id: str
    index_version: str
    stages_ms: dict[str, float] = Field(default_factory=dict)
    validation: str = "exact-extractive"
    model_usage: dict[str, Any] = Field(default_factory=dict)
    context_documents: list[str] = Field(default_factory=list, max_length=80)


class MemoryInput(StrictModel):
    content: str = Field(min_length=1, max_length=2000)
    kind: Literal["fact", "preference", "goal", "episode"] = "fact"
    importance: float = Field(0.5, ge=0, le=1)
    valid_until: float | None = None


class MemoryPatch(StrictModel):
    content: str | None = Field(None, min_length=1, max_length=2000)
    enabled: bool | None = None
    importance: float | None = Field(None, ge=0, le=1)
    valid_until: float | None = None
    expected_version: int = Field(ge=1)


class MemoryRecord(MemoryInput):
    memory_id: str
    owner: str
    source: str = "explicit-user"
    created_at: float
    updated_at: float
    enabled: bool = True
    version: int = 1


class RetrievalParams(StrictModel):
    top_k: int = Field(4, ge=1, le=12)
    candidate_pool: int = Field(24, ge=4, le=100)
    min_relevance: float = Field(0.16, ge=0, le=1)
    rrf_k: int = Field(60, ge=1, le=100)
    vector_weight: float = Field(0.5, ge=0, le=1)
    mmr_lambda: float | None = Field(None, ge=0, le=1)


class QueryInput(StrictModel):
    question: Question
    conversation_id: Identifier = "default"
    params: RetrievalParams = Field(default_factory=RetrievalParams)
    timeout_seconds: float = Field(20, ge=0.1, le=120)


class BuildInput(StrictModel):
    paths: list[str] | None = Field(None, max_length=100)
    chunk_size: int = Field(480, ge=120, le=2000)
    overlap: int = Field(80, ge=0, le=1000)
    visibility: Literal["public", "private"] = "public"
    timeout_seconds: float = Field(60, ge=1, le=120)

    @model_validator(mode="after")
    def overlap_limit(self):
        if self.overlap > self.chunk_size // 2:
            raise ValueError("overlap must not exceed half chunk_size")
        return self


class TaskSpec(StrictModel):
    id: Identifier
    tool: str = Field(min_length=1, max_length=64)
    arguments: dict[str, Any]
    depends_on: list[Identifier] = Field(default_factory=list, max_length=8)
    use_dependencies: bool = False


class RunInput(StrictModel):
    message: str = Field("", max_length=4000)
    tasks: list[TaskSpec] | None = Field(None, min_length=1, max_length=8)
    conversation_id: Identifier = "default"
    switch_topic: bool = False
    pause_for_review: bool = False
    timeout_seconds: float = Field(30, ge=0.1, le=120)
    params: RetrievalParams = Field(default_factory=RetrievalParams)

    @model_validator(mode="after")
    def has_input(self):
        if not self.tasks and not self.message:
            raise ValueError("message or tasks required")
        return self


class ToolResult(StrictModel):
    task_id: str
    tool: str
    status: Literal["succeeded", "failed", "cancelled", "needs_reconciliation", "skipped"]
    value: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    retryable: bool = False
    operation_key: str | None = None


class RunView(StrictModel):
    run_id: str
    conversation_id: str
    topic_epoch: int
    thread_id: str
    checkpoint_id: str | None
    status: str
    deadline: float
    tasks: list[TaskSpec]
    results: dict[str, ToolResult]
    unfinished: list[str]
    decisions: list[str]
    recoverable: bool
    resume_reason: str | None = None
    interrupts: list[Any] = Field(default_factory=list)


def merge_by_key(left: dict | None, right: dict | None) -> dict:
    """superstep 与恢复（resume）共用的幂等（idempotent）合并：每个任务 ID 只保留一份不可变结果。"""
    result = dict(left or {})
    for key, value in (right or {}).items():
        if key in result and result[key] != value:
            raise ValueError(f"conflicting parallel result: {key}")
        result[key] = value
    return result


class AgentState(TypedDict, total=False):
    state_version: int
    graph_version: str
    principal: str
    permission_version: int
    conversation_id: str
    topic_epoch: int
    run_id: str
    deadline: float
    index_version: str
    message: str
    params: dict
    tasks: list[dict]
    task: dict
    dependency_results: dict
    results: Annotated[dict[str, dict], merge_by_key]
    events: Annotated[dict[str, str], merge_by_key]
    waves: int
    history: list[dict]
    status: str
    approval_required: bool
    approved: bool


class RagState(TypedDict, total=False):
    model_scope: str
    principal: str
    question: str
    resolved_question: str
    index_version: str
    deadline: float
    params: dict
    history: list[dict]
    trace_id: str
    branches: Annotated[dict[str, dict], merge_by_key]
    degradation: Annotated[list[str], operator.add]
    evidence: list[dict]
    answer: dict


def simple_plan(message: str) -> list[TaskSpec]:
    parts = [x.strip() for x in re.split(r"[；;\n]+", message) if x.strip()]
    if not 1 <= len(parts) <= 8:
        raise ValueError("plan must contain 1..8 explicit independent questions")
    return [
        TaskSpec(id=f"q{i + 1}", tool="rag.answer", arguments={"question": p})
        for i, p in enumerate(parts)
    ]
