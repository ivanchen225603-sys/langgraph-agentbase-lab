import asyncio
import copy
import time

import pytest
from pydantic import ValidationError

from raglab.config import Settings
from raglab.embedding import LexicalHashEmbeddings, cosine
from raglab.errors import LabError, TransientError
from raglab.generation import validate_claims
from raglab.ingestion import load_documents, split_documents
from raglab.models import (
    BuildInput,
    Principal,
    QueryInput,
    RetrievalParams,
    RunInput,
    TaskSpec,
    merge_by_key,
)
from raglab.persistence import digest
from raglab.resilience import AnswerCache, Circuit, guarded
from raglab.retrieval import fuse


def test_embeddings_deterministic():
    emb = LexicalHashEmbeddings()
    one, two = emb.embed_documents(["RAG 文档分块", "RAG 文档分块"])
    assert one == two == emb.embed_query("RAG 文档分块")
    assert cosine(one, two) == pytest.approx(1)
    assert emb.embed_query("") == [0] * 384


def test_reducer_replay_safe():
    a, b = {"a": {"x": 1}}, {"b": {"x": 2}}
    assert merge_by_key(a, b) == merge_by_key(b, a)
    assert merge_by_key(a, a) == a
    with pytest.raises(ValueError):
        merge_by_key(a, {"a": {"x": 3}})


@pytest.mark.parametrize(
    "data", [{"chunk_size": 100}, {"overlap": 400}, {"chunk_size": 480, "overlap": 241}]
)
def test_split_schema_limits(data):
    with pytest.raises(ValidationError):
        BuildInput(**data)


def test_split_offsets_stability(tmp_path):
    path = tmp_path / "a.md"
    path.write_text("# Guide\n\n" + "Document boundaries and evidence. " * 50, encoding="utf-8")
    req = BuildInput(chunk_size=120, overlap=30)
    docs = load_documents(tmp_path, req, "local", Settings())
    first = split_documents(docs, req, 100)
    assert first == split_documents(docs, req, 100)
    assert len(first) > 1
    for chunk in first:
        assert docs[0]["content"][chunk["start"] : chunk["end"]] == chunk["text"]


@pytest.mark.parametrize(
    "file_name,content,build_data",
    [
        ("bad.bin", b"hello", {"paths": ["bad.bin"]}),
        ("bad.md", b"\xff\xfe", {"paths": ["bad.md"]}),
        ("empty.txt", b"", {"paths": ["empty.txt"]}),
        ("big.txt", b"a" * 100, {"paths": ["big.txt"]}),
    ],
)
def test_loader_rejects_bad_inputs(tmp_path, file_name, content, build_data):
    (tmp_path / file_name).write_bytes(content)
    with pytest.raises(LabError):
        load_documents(tmp_path, BuildInput(**build_data), "local", Settings(max_file_bytes=50))


def test_loader_path_escape(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.md").write_text("untrusted", encoding="utf-8")
    with pytest.raises(LabError, match="escapes"):
        load_documents(root, BuildInput(paths=["../outside.md"]), "local", Settings())


def test_rrf_does_not_compare_raw_scores():
    c1 = {"chunk_id": "a", "text": "RAG", "document_id": "d"}
    c2 = {"chunk_id": "b", "text": "RAG", "document_id": "d"}
    branches = {
        "vector": {"documents": [{"chunk": c1, "score": 0.9}, {"chunk": c2, "score": 0.8}]},
        "keyword": {"documents": [{"chunk": c2, "score": 9000}, {"chunk": c1, "score": 1}]},
    }
    result = fuse(branches, "RAG", RetrievalParams())
    assert len(result) == 2
    assert result[0]["fusion_score"] == result[1]["fusion_score"]
    assert result[0]["channels"] == ["vector", "keyword"]


def test_citation_and_negation_validation():
    chunk = {
        "chunk_id": "c",
        "document_id": "d",
        "source": "a",
        "index_version": "v",
        "text": "生产未批准外网访问，超时 8 秒。",
    }
    candidate = {"claims": [{"chunk_id": "c", "quote": chunk["text"], "text": chunk["text"]}]}
    assert (
        validate_claims(candidate, [{"chunk": chunk}], "v", extractive=True)[1][0]["chunk_id"]
        == "c"
    )
    for text in ("生产批准外网访问，超时 8 秒。", "生产未批准外网访问，超时 80 秒。"):
        broken = copy.deepcopy(candidate)
        broken["claims"][0]["text"] = text
        with pytest.raises(ValueError):
            validate_claims(broken, [{"chunk": chunk}], "v", extractive=False)
    with pytest.raises(ValueError):
        validate_claims(candidate, [{"chunk": chunk}], "wrong_version", extractive=True)


@pytest.mark.parametrize(
    "tasks",
    [
        [
            TaskSpec(id="a", tool="rag.answer", arguments={"question": "q"}),
            TaskSpec(id="a", tool="rag.answer", arguments={"question": "q"}),
        ],
        [TaskSpec(id="a", tool="unknown", arguments={"question": "q"})],
        [TaskSpec(id="a", tool="rag.answer", arguments={"question": "q"}, depends_on=["unknown"])],
        [TaskSpec(id="a", tool="rag.answer", arguments={"question": "q"}, depends_on=["a"])],
        [
            TaskSpec(id="a", tool="rag.answer", arguments={"question": "q"}, depends_on=["b"]),
            TaskSpec(id="b", tool="rag.answer", arguments={"question": "q"}, depends_on=["a"]),
        ],
    ],
)
async def test_plan_validation(service, tasks):
    with pytest.raises(LabError):
        await service.submit(Principal(owner="local"), "bad", RunInput(tasks=tasks))
    assert not service.store.requests()


async def test_retry_classification():
    calls = 0

    async def call():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TransientError()
        return "ok"

    assert await guarded(call, time.time() + 1, Circuit()) == "ok"
    assert calls == 2
    calls = 0

    async def bad():
        nonlocal calls
        calls += 1
        raise LabError("forbidden", "denied", 403)

    with pytest.raises(LabError):
        await guarded(bad, time.time() + 1, Circuit())
    assert calls == 1


async def test_circuit_half_open_one_probe():
    circuit = Circuit(threshold=1, cooldown=0)
    circuit.fail()
    circuit.enter()
    with pytest.raises(LabError, match="open"):
        circuit.enter()
    circuit.success()
    circuit.enter()


async def test_cache_singleflight_cleanup():
    cache, calls, entered, release = AnswerCache(2), 0, asyncio.Event(), asyncio.Event()

    async def call():
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return {"ok": 1}

    first = asyncio.create_task(cache.get_or_compute("x", call))
    await entered.wait()
    other = asyncio.create_task(cache.get_or_compute("x", call))
    release.set()
    assert await first == await other
    assert calls == 1 and not cache.flights

    async def fail():
        raise ValueError()

    with pytest.raises(ValueError):
        await cache.get_or_compute("failure", fail)
    assert not cache.flights


async def test_cache_key_isolation(service):
    q = QueryInput(question="RAG")
    a = service.cache_key(Principal(owner="alice"), q, "v", 0, [])
    b = service.cache_key(Principal(owner="bob"), q, "v", 0, [])
    assert a != b
    for version, epoch, history in [("v2", 0, []), ("v", 1, []), ("v", 0, [{"q": "x"}])]:
        assert a != service.cache_key(Principal(owner="alice"), q, version, epoch, history)
    assert digest("x") != digest("y")
