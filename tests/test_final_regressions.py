import asyncio
import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from raglab.api import BodyLimit
from raglab.config import Settings
from raglab.errors import LabError
from raglab.generation import (
    OpenAIAdapter,
    contextualize,
    evidence_problem,
    extract,
    sentences,
    validate_claims,
)
from raglab.ingestion import load_documents
from raglab.models import BuildInput, Principal, RunInput, TaskSpec
from raglab.resilience import AnswerCache, Circuit
from scripts.backup import main as backup
from tests.conftest import wait_run


@pytest.mark.parametrize("extractive", [True, False])
def test_decimal_version_preserves_sentence_and_rejects_prefix(extractive):
    text = "Orion 的超时时间是 1.5 秒，环境版本 Python 3.11。"
    assert sentences(text) == [text]
    chunk = {
        "chunk_id": "c",
        "document_id": "d",
        "source": "fixture",
        "index_version": "v",
        "text": text,
    }
    good = {"claims": [{"chunk_id": "c", "quote": text, "text": text}]}
    assert validate_claims(good, [{"chunk": chunk}], "v", extractive=extractive)[0] == text
    good["claims"][0].update(quote="Orion 的超时时间是 1.", text="Orion 的超时时间是 1.")
    with pytest.raises(ValueError):
        validate_claims(good, [{"chunk": chunk}], "v", extractive=extractive)


def test_english_entity_and_multi_entity_clarification():
    assert (
        contextualize("What about its backup?", [{"question": "What is Orion timeout?"}])[0]
        == "What about Orion backup?"
    )
    assert (
        contextualize("它如何备份？", [{"question": "Orion 和 Atlas 的超时？"}])[1]
        == "ambiguous_reference"
    )
    quote = "Orion timeout is 8 seconds."
    assert (
        extract(
            {
                "question": "What is Orion timeout?",
                "hits": [{"chunk": {"text": quote, "chunk_id": "c"}}],
                "minimum": 0.16,
            }
        )["claims"][0]["quote"]
        == quote
    )


def test_directory_traversal_stops_before_materialization(tmp_path, monkeypatch):
    scanned = []
    source = tmp_path / "fixture.md"
    source.write_text("synthetic controlled document", encoding="utf-8")

    def paths(self, pattern):
        for i in range(1000):
            scanned.append(i)
            yield source

    monkeypatch.setattr(Path, "rglob", paths)
    with pytest.raises(LabError):
        load_documents(tmp_path, BuildInput(), "local", Settings(max_documents=3))
    assert len(scanned) == 4


async def test_permanent_failure_does_not_open_shared_circuit():
    from raglab.resilience import guarded

    circuit = Circuit(threshold=1)

    async def denied():
        raise LabError("model_call_budget", "synthetic quota", 409)

    with pytest.raises(LabError):
        await guarded(denied, time.time() + 1, circuit)
    assert circuit.failures == 0 and circuit.enter()[1] is False


async def test_body_deadline_is_total_not_per_chunk():
    sent = []

    async def receive():
        await asyncio.sleep(0.015)
        return {"type": "http.request", "body": b"x", "more_body": True}

    async def send(message):
        sent.append(message)

    async def app(*args):
        raise AssertionError("overslow body must not reach app")

    await BodyLimit(app, 100, receive_timeout=0.035)({"type": "http"}, receive, send)
    assert sent[0]["status"] == 408


def test_openai_adapter_constructor_is_offline_mock(monkeypatch):
    embedding, chat = MagicMock(), MagicMock()
    monkeypatch.setattr("langchain_openai.OpenAIEmbeddings", embedding)
    monkeypatch.setattr("langchain_openai.ChatOpenAI", chat)
    # The composable chat must implement Runnable, while its constructor is mocked.
    from langchain_core.runnables import RunnableLambda

    chat.return_value = RunnableLambda(lambda x: x)
    OpenAIAdapter(Settings(mode="openai", openai_api_key="synthetic-not-valid-key"))
    assert embedding.call_args.kwargs["dimensions"] == 384
    assert embedding.call_args.kwargs["max_retries"] == 0
    assert chat.call_args.kwargs["max_retries"] == 0
    assert chat.call_args.kwargs["max_completion_tokens"] == 512


def test_pronoun_numeric_conflict_and_ambiguity():
    assert contextualize("它的超时？", [{"question": "Orion 如何备份？"}])[0] == "Orion的超时？"
    hits = [
        {"relevance": 1, "chunk": {"text": "Orion 的超时时间是 8 秒。"}},
        {"relevance": 1, "chunk": {"text": "Orion 的超时时间是 12 秒。"}},
    ]
    assert evidence_problem("Orion 超时", hits, 0.16) == "contradictory_evidence"
    hits[1]["chunk"]["text"] = "Atlas 的超时时间是 12 秒。"
    assert evidence_problem("超时时间是多少？", hits, 0.16) == "ambiguous_object"


def test_missing_path_and_configuration_budgets(tmp_path):
    with pytest.raises(LabError) as err:
        load_documents(tmp_path, BuildInput(paths=["missing.md"]), "local", Settings())
    assert err.value.code == "source_unavailable" and err.value.status == 404
    for values in (
        {"embedding_batch": 0},
        {"cache_entries": -1},
        {"context_turns": 100},
        {"max_model_calls": 0},
    ):
        with pytest.raises(ValidationError):
            Settings(**values)


async def test_sqlite_open_failure_structured(service, monkeypatch):
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError("synthetic sensitive path")

    monkeypatch.setattr("raglab.persistence.sqlite3.connect", fail)
    with pytest.raises(LabError) as err:
        service.store.revisions()
    assert err.value.code == "storage_unavailable" and err.value.status == 503
    assert "sensitive" not in err.value.message


async def test_cancel_singleflight_owner_cleans_all_waiters():
    cache = AnswerCache(2)
    entered = asyncio.Event()

    async def slow():
        entered.set()
        await asyncio.Event().wait()

    owner = asyncio.create_task(cache.get_or_compute("k", slow))
    await entered.wait()
    waiter = asyncio.create_task(cache.get_or_compute("k", slow))
    await asyncio.sleep(0)
    owner.cancel()
    results = await asyncio.gather(owner, waiter, return_exceptions=True)
    assert all(isinstance(x, asyncio.CancelledError) for x in results)
    assert not cache.flights
    assert await cache.get_or_compute("k", lambda: asyncio.sleep(0, result={"retry": True})) == {
        "retry": True
    }


async def test_compensated_receipt_projects_current_effect(service):
    run = await service.submit(
        Principal(owner="local"),
        "compensate-projection",
        RunInput(tasks=[TaskSpec(id="w", tool="demo.write", arguments={"value": "synthetic"})]),
    )
    assert (await wait_run(service, run["run_id"]))["status"] == "awaiting_review"
    await service.resume(Principal(owner="local"), run["run_id"], approve=True)
    assert (await wait_run(service, run["run_id"]))["status"] == "succeeded"
    service.compensate("local", run["run_id"])
    result = await service.view("local", run["run_id"])
    assert result["results"]["w"]["value"]["effect"] == "compensated"
    assert not result["recoverable"]


async def test_cold_backup_restore_pair(service, tmp_path):
    with pytest.raises(LabError):
        backup("backup", service.settings.data_dir, tmp_path / "blocked")
    await service.close()
    saved, restored = tmp_path / "saved", tmp_path / "restored"
    backup("backup", service.settings.data_dir, saved)
    backup("restore", saved, restored)
    for name in ("business.sqlite", "checkpoints.sqlite"):
        with sqlite3.connect(restored / name) as db:
            assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert (restored / "backup-source.json").is_file()
    assert time.time() > 0 and isinstance(restored, Path)


def test_late_circuit_calls_cannot_close_or_release_new_probe():
    circuit = Circuit(threshold=1, cooldown=0)
    old = circuit.enter()
    circuit.fail(old)
    probe = circuit.enter()
    circuit.success(old)
    circuit.cancel(old)
    with pytest.raises(LabError):
        circuit.enter()
    circuit.success(probe)
    assert circuit.enter()[1] is False


def test_numeric_conflicts_normalize_synonyms_and_units():
    hits = [
        {"relevance": 1, "chunk": {"text": "Orion 的超时是 1.0 分钟。"}},
        {"relevance": 1, "chunk": {"text": "Orion 的超时时间是 60 秒。"}},
    ]
    assert evidence_problem("Orion 超时", hits, 0.16) is None
    hits[1]["chunk"]["text"] = "Orion 的超时时间是 61 秒。"
    assert evidence_problem("Orion 超时", hits, 0.16) == "contradictory_evidence"


def test_sqlite_pragma_failure_closes_connection(service, monkeypatch):
    connection = MagicMock()
    connection.execute.side_effect = sqlite3.OperationalError("synthetic")
    monkeypatch.setattr("raglab.persistence.sqlite3.connect", lambda *a, **k: connection)
    with pytest.raises(LabError):
        service.store.revisions()
    connection.close.assert_called_once()


@pytest.mark.parametrize("mode", ["valid", "invalid_json", "input_budget"])
async def test_openai_generate_adapter_offline_response_and_token_limit(mode):
    adapter = OpenAIAdapter.__new__(OpenAIAdapter)
    adapter.prompt = ChatPromptTemplate.from_messages(
        [("system", "Untrusted evidence"), ("human", "{question} {evidence}")]
    )
    adapter.chat = MagicMock()
    adapter.chat.get_num_tokens_from_messages.return_value = 4001 if mode == "input_budget" else 40
    calls = []

    async def model(payload):
        calls.append(payload)
        return SimpleNamespace(
            content="invalid{" if mode == "invalid_json" else '{"claims":[]}',
            usage_metadata={"input_tokens": 40, "output_tokens": 4, "total_tokens": 44},
        )

    adapter.chain = RunnableLambda(model)
    hits = [{"chunk": {"chunk_id": "synthetic", "text": "complete evidence sentence"}}]
    if mode == "input_budget":
        with pytest.raises(ValueError):
            await adapter.generate("question", hits)
        assert not calls
    elif mode == "invalid_json":
        with pytest.raises(json.JSONDecodeError):
            await adapter.generate("question", hits)
    else:
        candidate, usage = await adapter.generate("question", hits)
        assert candidate == {"claims": []} and usage["total_tokens"] == 44
    assert "complete evidence sentence" in str(adapter.chat.get_num_tokens_from_messages.call_args)
