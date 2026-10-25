import asyncio
import time

import pytest

from raglab.errors import LabError
from raglab.models import BuildInput, MemoryInput, MemoryPatch, Principal, QueryInput
from raglab.retrieval import SnapshotRetriever
from tests.conftest import LOCAL


async def test_local_answer_citations(service):
    result = await service.answer(LOCAL, QueryInput(question="RAG 为什么要把文档分块？"))
    assert result["status"] == "answered"
    assert "分块" in result["answer"]
    for cite in result["citations"]:
        hit = next(x for x in result["evidence"] if x["chunk"]["chunk_id"] == cite["chunk_id"])
        assert cite["quote"] in hit["chunk"]["text"]
        assert cite["index_version"] == result["index_version"]


async def test_no_answer_and_pronoun_clarification(service):
    assert (await service.answer(LOCAL, QueryInput(question="月球酒店的电话号码？")))[
        "status"
    ] == "refused"
    assert (
        await service.answer(
            LOCAL, QueryInput(question="它的超时是多少？", conversation_id="empty")
        )
    )["status"] == "clarify"


async def test_actual_history_consumption(service):
    first = await service.answer(
        LOCAL, QueryInput(question="Orion 服务如何备份？", conversation_id="multi")
    )
    assert "Orion" in first["answer"]
    second = await service.answer(
        LOCAL, QueryInput(question="它的超时时间是多少？", conversation_id="multi")
    )
    assert second["status"] == "answered"
    assert second["resolved_question"] == "Orion的超时时间是多少？"
    assert "8 秒" in second["answer"] and "服务端" not in second["answer"]
    bob = await service.answer(
        Principal(owner="bob"), QueryInput(question="它的超时时间是多少？", conversation_id="multi")
    )
    assert bob["status"] == "clarify"


async def test_index_reuse_and_failure_preserves_active(service, monkeypatch):
    original = service.index_version()
    rebuilt = await service.build("local", BuildInput())
    assert rebuilt["manifest"]["reused_vectors"] == rebuilt["manifest"]["chunks"]
    assert rebuilt["manifest"]["version"] != original
    active = service.index_version()

    async def broken(_):
        raise ValueError("synthetic embedding failure")

    monkeypatch.setattr(service.embeddings, "aembed_documents", broken)
    with pytest.raises(ValueError):
        await service.build("local", BuildInput(chunk_size=120))
    assert service.index_version() == active


@pytest.mark.parametrize("failed", ["vector", "keyword", "both"])
async def test_channel_failures(service, monkeypatch, failed):
    original = SnapshotRetriever._aget_relevant_documents

    async def call(self, query, **kwargs):
        if failed == "both" or self.channel == failed:
            raise ValueError("synthetic channel failure")
        return await original(self, query, **kwargs)

    monkeypatch.setattr(SnapshotRetriever, "_aget_relevant_documents", call)
    result = await service.answer(LOCAL, QueryInput(question="RAG 为什么要把文档分块？"))
    assert result["status"] == ("refused" if failed == "both" else "answered")
    assert result["degradation"]


async def test_rerank_failure_fallback(service):
    def broken(*args):
        raise ValueError()

    service.rag.reranker = broken
    result = await service.answer(LOCAL, QueryInput(question="文档分块上下文成本"))
    assert result["status"] == "answered"
    assert "rerank:failed_using_fusion" in result["degradation"]


@pytest.mark.parametrize("failure", ["exception", "fake", "false_claim"])
async def test_generator_failure_or_false_claim_extracts(service, failure):
    class FakeGenerator:
        async def generate(self, question, hits):
            if failure == "exception":
                raise ValueError("secret_error_marker")
            chunk = hits[0]["chunk"]
            return {
                "claims": [
                    {
                        "chunk_id": "fake" if failure == "fake" else chunk["chunk_id"],
                        "quote": "not real",
                        "text": "未证实的事实",
                    }
                ]
            }, {}

    service.generator = FakeGenerator()
    result = await service.answer(LOCAL, QueryInput(question="文档分块上下文成本"))
    assert result["status"] == "answered" and result["validation"] == "exact-extractive"
    assert result["degradation"]
    assert "secret_error_marker" not in str(result)


async def test_revocation_inflight_and_history(service):
    entered, release = asyncio.Event(), asyncio.Event()

    class FakeGenerator:
        async def generate(self, question, hits):
            entered.set()
            await release.wait()
            raise ValueError()

    service.generator = FakeGenerator()
    pending = asyncio.create_task(service.answer(LOCAL, QueryInput(question="文档分块上下文成本")))
    await entered.wait()
    rows = service.store.snapshot(service.index_version(), "local")
    document = next(c["document_id"] for c, _ in rows if c["source"] == "01-rag.md")
    service.store.revoke("local", document)
    release.set()
    result = await pending
    assert result["status"] == "refused" and not result["citations"] and not result["evidence"]
    assert not service.store.history("local", "default", 0, 4, 3000)


async def test_memory_owner_dedup_correction_disable_delete(service):
    memories = service.memories
    original = memories.create("alice", MemoryInput(content="偏好 Python 示例", kind="preference"))
    assert (
        memories.create("alice", MemoryInput(content="偏好 Python 示例", kind="preference"))[
            "memory_id"
        ]
        == original["memory_id"]
    )
    assert not memories.search("bob", "Python 示例")
    changed = memories.patch(
        "alice", original["memory_id"], MemoryPatch(content="偏好 Java 示例", expected_version=1)
    )
    assert changed["version"] == 2
    assert not service.store.memory_current("alice", original["memory_id"], 1)
    with pytest.raises(LabError, match="changed"):
        memories.patch(
            "alice", original["memory_id"], MemoryPatch(enabled=False, expected_version=1)
        )
    memories.patch("alice", original["memory_id"], MemoryPatch(enabled=False, expected_version=2))
    assert not memories.search("alice", "Java 示例")
    memories.delete("alice", original["memory_id"])
    assert not memories.list("alice")


async def test_expired_memories_filtered(service):
    service.memories.create(
        "local", MemoryInput(content="Python 偏好", valid_until=time.time() - 1)
    )
    assert not service.memories.search("local", "Python 偏好")


@pytest.mark.parametrize(
    "source,quote",
    [
        ("生产环境禁止执行任意 shell。", "执行任意 shell"),
        ("生产环境不允许外部访问。", "允许外部访问"),
    ],
)
async def test_model_cannot_clip_negation_from_citation(service, tmp_path, source, quote):
    root = tmp_path / "conditions"
    root.mkdir()
    (root / "conditions.md").write_text("# 生产环境\n\n" + source, encoding="utf-8")
    service.settings.knowledge_root = root
    await service.build("local", BuildInput())

    class ClippedGenerator:
        async def generate(self, question, hits):
            return {
                "claims": [
                    {"chunk_id": hits[0]["chunk"]["chunk_id"], "quote": quote, "text": quote}
                ]
            }, {}

    service.generator = ClippedGenerator()
    result = await service.answer(LOCAL, QueryInput(question="生产环境 " + quote + "？"))
    assert result["validation"] == "exact-extractive"
    assert result["status"] == "answered" and result["answer"] == source
    assert result["degradation"]


async def test_revocation_before_generation_never_sends_document(service):
    original = service.rag.reranker
    calls = []

    class RecordingGenerator:
        async def generate(self, question, hits):
            calls.append(hits)
            raise AssertionError("revoked data must not reach model")

    service.generator = RecordingGenerator()

    def revoke_during_fusion(hits, question):
        for hit in hits:
            service.store.revoke("local", hit["chunk"]["document_id"])
        return original(hits, question)

    service.rag.reranker = revoke_during_fusion
    result = await service.answer(LOCAL, QueryInput(question="文档分块上下文成本"))
    assert result["status"] == "refused" and not calls and not result["evidence"]
