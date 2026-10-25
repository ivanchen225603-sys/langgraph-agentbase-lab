"""Retained storage bounds, CAS updates, maintenance and concurrent admission."""

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from pydantic import ValidationError

from agentbase import MemoryLimits
from agentbase.contracts import AgentError
from agentbase.schema import canonical_json
from agentbase.storage import SqliteContextStore, SqliteDatabase


@pytest.mark.parametrize("value", [0, -1, True, "10"])
def test_limit_configuration_is_strict(value):
    with pytest.raises(ValidationError):
        MemoryLimits(max_owner_records=value)


def test_sources_and_owner_bytes_are_counted_across_namespaces(setup_store, monkeypatch):
    database, make = setup_store
    monkeypatch.setattr(time, "time", lambda: 1234.0)
    source = [{"value": "中" * 100}]
    unlimited = SqliteContextStore(database.connection)
    first = unlimited.create_memory("a", "worker", "one", "中", sources=source)
    size = len(canonical_json(first.model_dump(mode="json")))
    unlimited.delete_memory("a", "worker", "one", first.id)
    store = make(max_owner_bytes=size)
    exact = store.create_memory("a", "worker", "one", "中", sources=source)
    assert len(canonical_json(exact.model_dump(mode="json"))) == size
    with pytest.raises(AgentError) as error:
        store.create_memory("a", "other", "two", "中")
    assert error.value.code == "memory_quota"
    assert store.create_memory("b", "worker", "one", "中", sources=source).owner == "b"


@pytest.fixture
def setup_store(tmp_path):
    database = SqliteDatabase(tmp_path / "memory.sqlite")
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        SqliteContextStore.install(db)

    def make(**values):
        return SqliteContextStore(database.connection, memory_limits=MemoryLimits(**values))

    return database, make


def test_disabled_limits_preserve_existing_store(setup_store):
    database, _ = setup_store
    store = SqliteContextStore(database.connection)
    for _ in range(25):
        store.create_memory("a", "worker", "", "hello")
    assert len(store.memories("a", "worker", "", 30)) == 25
    assert len(store.memories("a", "worker", "", 5)) == 5


def test_namespace_and_owner_counts_include_disabled_and_expired(setup_store):
    _, make = setup_store
    store = make(max_namespace_records=1, max_owner_records=2)
    first = store.create_memory("a", "worker", "one", "hello", valid_until=time.time() - 1)
    store.update_memory("a", "worker", "one", first.id, first.version, enabled=False)
    with pytest.raises(AgentError, match="storage limit") as error:
        store.create_memory("a", "worker", "one", "another")
    assert error.value.code == "memory_quota"
    store.create_memory("a", "other", "two", "hello")
    with pytest.raises(AgentError):
        store.create_memory("a", "third", "three", "hello")
    assert store.create_memory("b", "worker", "one", "hello").owner == "b"
    store.delete_memory("a", "worker", "one", first.id)
    store.create_memory("a", "worker", "one", "replacement")


@pytest.mark.parametrize("field", ["max_namespace_bytes", "max_owner_bytes"])
def test_utf8_payload_byte_quota_and_failed_update_are_atomic(setup_store, field):
    database, make = setup_store
    original = SqliteContextStore(database.connection).create_memory("a", "worker", "", "中")
    size = len(canonical_json(original.model_dump(mode="json")))
    with database.connection() as db:
        assert (
            db.execute("SELECT length(CAST(payload AS BLOB)) FROM agent_memories").fetchone()[0]
            == size
        )
    store = make(**{field: size + 20})
    with pytest.raises(AgentError) as error:
        store.update_memory("a", "worker", "", original.id, 1, content="中" * 100)
    assert error.value.code == "memory_quota"
    assert store.memory("a", "worker", "", original.id) == original
    with pytest.raises(AgentError):
        store.create_memory("a", "worker", "", "new")
    with database.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM agent_memories").fetchone()[0] == 1


def test_oversized_existing_data_allows_maintenance_and_delete(setup_store):
    database, make = setup_store
    original = SqliteContextStore(database.connection).create_memory("a", "worker", "", "x" * 100)
    store = make(max_namespace_bytes=1, max_owner_bytes=1)
    smaller = store.update_memory("a", "worker", "", original.id, 1, content="x")
    disabled = store.update_memory("a", "worker", "", original.id, smaller.version, enabled=False)
    assert disabled.content == "x" and not disabled.enabled
    with pytest.raises(AgentError):
        store.update_memory(
            "a", "worker", "", original.id, disabled.version, content="longer", enabled=False
        )
    with pytest.raises(AgentError) as conflict:
        store.update_memory("a", "worker", "", original.id, 1, content="x")
    assert conflict.value.code == "memory_conflict"
    store.delete_memory("a", "worker", "", original.id)
    with database.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM agent_memories").fetchone()[0] == 0


def test_disabling_at_exact_byte_capacity_still_counts_retained_record(setup_store):
    database, make = setup_store
    original = SqliteContextStore(database.connection).create_memory("a", "worker", "", "hello")
    size = len(canonical_json(original.model_dump(mode="json")))
    store = make(max_namespace_bytes=size, max_owner_bytes=size)
    disabled = store.update_memory("a", "worker", "", original.id, 1, enabled=False)
    assert not disabled.enabled
    with pytest.raises(AgentError):
        store.create_memory("a", "worker", "", "replacement")


@pytest.mark.parametrize("replacement", ["\x00" * 99, '"' * 99, "\\" * 99])
@pytest.mark.parametrize("enabled", [None, False])
def test_json_escaping_cannot_disguise_content_growth(setup_store, replacement, enabled):
    database, make = setup_store
    original = SqliteContextStore(database.connection).create_memory("a", "worker", "", "x" * 100)
    size = len(canonical_json(original.model_dump(mode="json")))
    store = make(max_namespace_bytes=size, max_owner_bytes=size)
    with pytest.raises(AgentError) as error:
        store.update_memory("a", "worker", "", original.id, 1, content=replacement, enabled=enabled)
    assert error.value.code == "memory_quota"
    assert store.memory("a", "worker", "", original.id) == original


@pytest.mark.parametrize("kind", ["create", "update"])
def test_competing_writes_cannot_spend_same_capacity(setup_store, kind):
    database, make = setup_store
    unlimited = SqliteContextStore(database.connection)
    originals = (
        [unlimited.create_memory("a", "worker", "", "x") for _ in range(2)]
        if kind == "update"
        else []
    )
    if kind == "update":
        with database.connection() as db:
            size = db.execute(
                "SELECT SUM(length(CAST(payload AS BLOB))) FROM agent_memories"
            ).fetchone()[0]
        store = make(max_namespace_bytes=size + 110)
    else:
        store = make(max_namespace_records=1)
    barrier = Barrier(2)

    def write(index):
        barrier.wait(timeout=5)
        try:
            if kind == "update":
                store.update_memory("a", "worker", "", originals[index].id, 1, content="x" * 100)
            else:
                store.create_memory("a", "worker", "", "hello")
            return "ok"
        except AgentError as error:
            return error.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(write, range(2)))
    assert sorted(results) == ["memory_quota", "ok"]


def test_limits_reject_mixed_database_encoding_but_none_remains_compatible(tmp_path):
    path = tmp_path / "utf16.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA encoding='UTF-16'")
        SqliteContextStore.install(db)
    database = SqliteDatabase(path)
    SqliteContextStore(database.connection).create_memory("a", "worker", "", "hello")
    store = SqliteContextStore(database.connection, memory_limits=MemoryLimits())
    with pytest.raises(AgentError) as error:
        store.create_memory("a", "worker", "", "hello")
    assert error.value.code == "memory_encoding"
