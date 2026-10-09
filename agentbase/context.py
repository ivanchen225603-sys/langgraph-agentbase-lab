"""有界、来源感知的上下文与显式记忆，独立于具体业务字段。"""

import copy
import sqlite3
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, cast

from pydantic import Field

from .artifacts import ArtifactRecord, ArtifactReference, ArtifactScope, ArtifactStore
from .configuration import AgentDefinition
from .contracts import AgentError, BaseContract, Identity, MemoryRecord, Message, RunContext
from .schema import canonical_json


class ContextStore(Protocol):
    def epoch(self, owner: str, agent_id: str, conversation_id: str) -> int: ...

    def turns(
        self, owner: str, agent_id: str, conversation_id: str, epoch: int, limit: int
    ) -> list[list[Message]]: ...

    def append_turn(
        self,
        owner: str,
        agent_id: str,
        conversation_id: str,
        epoch: int,
        turn_id: str,
        messages: list[Message],
        *,
        guard: Callable[[sqlite3.Connection], bool],
    ) -> bool: ...


class MemoryStore(Protocol):
    def memories(
        self, owner: str, agent_id: str, conversation_id: str, limit: int
    ) -> list[MemoryRecord]: ...

    def memory(
        self, owner: str, agent_id: str, conversation_id: str, memory_id: str
    ) -> MemoryRecord: ...


class ContextBundle(BaseContract):
    history: list[Message] = Field(default_factory=list)
    memories: list[MemoryRecord] = Field(default_factory=list)
    dropped_turns: int = 0


SourceChecker = Callable[[Identity, str, dict], Awaitable[bool]]


@dataclass(frozen=True)
class SourceFrame:
    identity: Identity
    agent_id: str
    memory_agents: frozenset[str]
    memory_conversation: str
    artifact_namespace: str | None
    run_id: str | None = None


@dataclass(frozen=True)
class SourceExpansion:
    key: tuple
    frame: SourceFrame
    sources: list[dict]
    stamp: str


SourceExpander = Callable[[SourceFrame, dict], SourceExpansion | None]
FrameAuthorizer = Callable[[SourceFrame], Awaitable[bool]]


class ContextProvider:
    def __init__(
        self,
        history: ContextStore,
        memories: MemoryStore,
        source_checker: SourceChecker | None = None,
        *,
        artifacts: ArtifactStore | None = None,
        source_expander: SourceExpander | None = None,
        frame_authorizer: FrameAuthorizer | None = None,
    ):
        self.history, self.memory_store, self.source_checker = history, memories, source_checker
        self.artifacts = artifacts
        self.source_expander = source_expander
        self.frame_authorizer = frame_authorizer
        self._runtime_frame_authorizer: FrameAuthorizer | None = None

    @staticmethod
    def _valid_message_sequence(messages: list[Message]) -> bool:
        pending: set[str] = set()
        for message in messages:
            if message.role == "system":
                return False
            if message.role == "assistant" and message.tool_calls:
                if pending:
                    return False
                ids = [call.call_id for call in message.tool_calls]
                if len(ids) != len(set(ids)):
                    return False
                pending.update(ids)
            elif message.role == "tool":
                if message.tool_call_id not in pending:
                    return False
                pending.remove(message.tool_call_id)
            elif pending:
                return False
        return not pending

    async def authorized_sources(
        self,
        identity: Identity,
        agent_id: str,
        sources: list[dict],
        *,
        memory_agents: frozenset[str] | None = None,
        memory_conversation: str = "",
        artifact_namespace: str | None = None,
        depth: int = 0,
        ancestors: frozenset[tuple] = frozenset(),
        run_id: str | None = None,
    ) -> bool:
        allowed = memory_agents if memory_agents is not None else frozenset({agent_id})
        artifacts = self.artifacts
        observed: dict[tuple, MemoryRecord | ArtifactRecord] = {}
        expanded: dict[tuple, tuple[SourceFrame, dict, SourceExpansion]] = {}
        checks = 0
        frames: set[SourceFrame] = set()

        async def authorize_frame(frame):
            frames.add(frame)
            for policy in (self.frame_authorizer, self._runtime_frame_authorizer):
                if policy is not None and await policy(frame) is not True:
                    return False
            return True

        def read_local(key):
            kind, owner, source_agent, namespace, identifier = key
            if kind == "memory":
                return self.memory_store.memory(owner, source_agent, namespace, identifier)
            assert artifacts is not None
            return artifacts.read(
                ArtifactScope(owner=owner, agent_id=source_agent, namespace=namespace), identifier
            )

        async def check(items, level, seen, frame):
            nonlocal checks
            if level > 16:
                return False
            if not await authorize_frame(frame):
                return False
            for source in items:
                checks += 1
                if checks > 1024 or not isinstance(source, dict):
                    return False
                kind = source.get("kind")
                if kind in {"child_receipt", "task_input"}:
                    if self.source_expander is None:
                        return False
                    try:
                        expansion = self.source_expander(frame, source)
                    except (AgentError, ValueError, KeyError):
                        return False
                    if expansion is None or expansion.key in seen:
                        return False
                    observation_key = (expansion.key, frame)
                    if observation_key in expanded and expanded[observation_key][2] != expansion:
                        return False
                    expanded[observation_key] = (frame, copy.deepcopy(source), expansion)
                    if expansion.sources and not await check(
                        expansion.sources, level + 1, seen | {expansion.key}, expansion.frame
                    ):
                        return False
                    if not expansion.sources and not await authorize_frame(expansion.frame):
                        return False
                    continue
                if kind == "memory":
                    parts = (
                        source.get("agent_id"),
                        source.get("conversation_id", ""),
                        source.get("id"),
                    )
                    if not all(isinstance(part, str) for part in parts):
                        return False
                    parts = cast(tuple[str, str, str], parts)
                    if parts[0] not in frame.memory_agents or parts[1] != frame.memory_conversation:
                        return False
                    key = ("memory", frame.identity.owner, *parts)
                elif kind == "artifact":
                    if artifacts is None or frame.artifact_namespace is None:
                        return False
                    try:
                        reference = ArtifactReference.model_validate(source)
                    except ValueError:
                        return False
                    if (
                        reference.agent_id != frame.agent_id
                        or reference.namespace != frame.artifact_namespace
                    ):
                        return False
                    key = (
                        "artifact",
                        frame.identity.owner,
                        reference.agent_id,
                        reference.namespace,
                        reference.id,
                    )
                else:
                    if self.source_checker is None or not await self.source_checker(
                        frame.identity, frame.agent_id, source
                    ):
                        return False
                    continue
                if key in seen:
                    return False
                try:
                    record = read_local(key)
                except (AgentError, KeyError):
                    return False
                namespace = (
                    record.conversation_id if isinstance(record, MemoryRecord) else record.namespace
                )
                if (
                    (record.owner, record.agent_id, namespace, record.id) != key[1:]
                    or not record.enabled
                    or record.version != source.get("version")
                    or (
                        isinstance(record, ArtifactRecord) and record.digest != source.get("digest")
                    )
                    or (record.valid_until is not None and record.valid_until <= time.time())
                    or (key in observed and observed[key] != record)
                ):
                    return False
                observed[key] = record
                if record.sources and not await check(
                    record.sources, level + 1, seen | {key}, frame
                ):
                    return False
            return True

        frame = SourceFrame(
            identity, agent_id, allowed, memory_conversation, artifact_namespace, run_id
        )
        if not await check(sources, depth, ancestors, frame):
            return False
        for checked_frame in frames:
            if not await authorize_frame(checked_frame):
                return False
        for original_frame, reference, expansion in expanded.values():
            try:
                if (
                    self.source_expander is None
                    or self.source_expander(original_frame, reference) != expansion
                ):
                    return False
            except (AgentError, ValueError, KeyError):
                return False
        # Recheck the entire local closure after all external awaits, including earlier siblings.
        # Runtime publication additionally validates it in the final SQL transaction.
        for key, record in observed.items():
            try:
                current = read_local(key)
            except AgentError:
                return False
            if (
                current != record
                or not current.enabled
                or (current.valid_until is not None and current.valid_until <= time.time())
            ):
                return False
        return True

    async def load(self, definition: AgentDefinition, context: RunContext) -> ContextBundle:
        if definition.id != context.agent_id:
            raise AgentError("context_agent", "context belongs to a different Agent", 403)
        policy, owner = definition.context, context.identity.owner
        memory_agents = (
            frozenset([context.agent_id, *policy.shared_memory_agents])
            if policy.memory_scope != "none"
            else frozenset()
        )
        memory_conversation = (
            context.conversation_id if policy.memory_scope == "conversation" else ""
        )
        if self.history.epoch(owner, context.agent_id, context.conversation_id) != context.epoch:
            raise AgentError("context_epoch", "conversation changed", 409)
        selected: list[list[Message]] = []
        dropped, chars, count = 0, 0, 0
        max_messages = (
            policy.management.candidate_messages if policy.management else policy.max_messages
        )
        max_chars = policy.management.candidate_chars if policy.management else policy.max_chars
        turns = self.history.turns(
            owner, context.agent_id, context.conversation_id, context.epoch, max_messages
        )
        for turn in reversed(turns):
            if not self._valid_message_sequence(turn):
                dropped += 1
                continue
            if not all(
                [
                    await self.authorized_sources(
                        context.access_identity,
                        context.agent_id,
                        message.sources,
                        memory_agents=memory_agents,
                        memory_conversation=memory_conversation,
                        artifact_namespace=context.conversation_id,
                        run_id=context.run_id,
                    )
                    for message in turn
                ]
            ):
                dropped += 1
                continue
            size = len(
                canonical_json([message.model_dump(mode="json") for message in turn]).decode()
            )
            if count + len(turn) > max_messages or chars + size > max_chars:
                dropped += 1
                continue
            selected.append(turn)
            count += len(turn)
            chars += size
        records: list[MemoryRecord] = []
        if policy.memory_scope != "none" and policy.max_memories:
            # 会话级（conversation-scoped）的显式记忆在 epoch 变更后依然保留；历史记录不会。
            conversation = context.conversation_id if policy.memory_scope == "conversation" else ""
            for agent_id in [context.agent_id, *policy.shared_memory_agents]:
                for record in self.memory_store.memories(
                    owner, agent_id, conversation, policy.max_memories
                ):
                    if await self.authorized_sources(
                        context.access_identity,
                        context.agent_id,
                        [self.memory_reference(record)],
                        memory_agents=memory_agents,
                        memory_conversation=memory_conversation,
                        artifact_namespace=context.conversation_id,
                        run_id=context.run_id,
                    ):
                        records.append(record)
            records.sort(key=lambda record: (record.importance, record.updated_at), reverse=True)
            bounded = []
            for record in records:
                size = len(record.content)
                if chars + size <= max_chars:
                    bounded.append(record)
                    chars += size
                if len(bounded) == policy.max_memories:
                    break
            records = bounded
        return ContextBundle(
            history=[message for turn in reversed(selected) for message in turn],
            memories=records,
            dropped_turns=dropped,
        )

    @staticmethod
    def memory_reference(record: MemoryRecord) -> dict:
        return {
            "kind": "memory",
            "id": record.id,
            "agent_id": record.agent_id,
            "conversation_id": record.conversation_id,
            "version": record.version,
        }

    async def revalidate(
        self, bundle: ContextBundle, definition: AgentDefinition, context: RunContext
    ) -> bool:
        """运行时在调用模型之前立即调用此方法；它不是数据库提交守卫（commit guard）。"""
        if definition.id != context.agent_id:
            raise AgentError("context_agent", "context belongs to a different Agent", 403)
        policy = definition.context
        allowed = (
            frozenset([context.agent_id, *policy.shared_memory_agents])
            if policy.memory_scope != "none"
            else frozenset()
        )
        conversation = context.conversation_id if policy.memory_scope == "conversation" else ""
        if (
            self.history.epoch(context.identity.owner, context.agent_id, context.conversation_id)
            != context.epoch
        ):
            return False
        sources = [source for message in bundle.history for source in message.sources]
        sources.extend(self.memory_reference(record) for record in bundle.memories)
        valid = await self.authorized_sources(
            context.access_identity,
            context.agent_id,
            sources,
            memory_agents=allowed,
            memory_conversation=conversation,
            artifact_namespace=context.conversation_id,
            run_id=context.run_id,
        )
        return (
            valid
            and self.history.epoch(
                context.identity.owner, context.agent_id, context.conversation_id
            )
            == context.epoch
        )

    def commit_final(
        self,
        context: RunContext,
        messages: list[Message],
        *,
        guard: Callable[[sqlite3.Connection], bool],
    ) -> bool:
        if not self._valid_message_sequence(messages):
            raise AgentError(
                "history_messages", "history contains invalid roles or tool observations"
            )
        return self.history.append_turn(
            context.identity.owner,
            context.agent_id,
            context.conversation_id,
            context.epoch,
            context.run_id,
            messages,
            guard=guard,
        )
