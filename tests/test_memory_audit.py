"""Synthetic databases only; this module is self-contained in a tracked export."""

import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from agentbase.configuration import AgentDefinition
from agentbase.context import ContextProvider
from agentbase.contracts import AgentError, Identity, MemoryRecord, Message, RunContext
from agentbase.schema import canonical_json, content_digest
from agentbase.storage import SqliteContextStore, SqliteDatabase

ACTOR = Identity(owner="tenant_a_user", scopes=frozenset({"read"}), permission_version=3)


@pytest.fixture
def store(tmp_path):
    database = SqliteDatabase(tmp_path / "synthetic.sqlite")
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        SqliteContextStore.install(db)
    return SqliteContextStore(database.connection)


def create(store, **changes):
    return store.create_memory(ACTOR.owner, "worker", "same", "SYNTHETIC_PRIVATE_BODY", **changes)


def audit(store, record, **changes):
    return store.memory_audit(ACTOR, "worker", "same", record.id, guard=lambda db: True, **changes)


def snapshot(store):
    with store.connection() as db:
        return {
            name: [tuple(row) for row in db.execute(f"SELECT * FROM {name}")]
            for name in ("agent_memories", "agent_memory_audit")
        }


def legacy_database(tmp_path):
    database = SqliteDatabase(tmp_path / "legacy.sqlite")
    record = MemoryRecord(
        id="legacy",
        owner=ACTOR.owner,
        agent_id="worker",
        conversation_id="same",
        content="SYNTHETIC_LEGACY_BODY",
        version=7,
        created_at=1.0,
        updated_at=2.0,
    )
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        for sql in (
            "CREATE TABLE agent_context_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)",
            "INSERT INTO agent_context_meta VALUES('schema','1')",
            "CREATE TABLE agent_conversations(owner TEXT,agent_id TEXT,conversation_id TEXT,epoch INTEGER NOT NULL,PRIMARY KEY(owner,agent_id,conversation_id))",
            "CREATE TABLE agent_history(owner TEXT,agent_id TEXT,conversation_id TEXT,epoch INTEGER,turn_id TEXT,payload TEXT NOT NULL,created_at REAL NOT NULL,PRIMARY KEY(owner,agent_id,conversation_id,epoch,turn_id))",
            "CREATE TABLE agent_memories(owner TEXT,agent_id TEXT,conversation_id TEXT,id TEXT,payload TEXT NOT NULL,version INTEGER NOT NULL,enabled INTEGER NOT NULL,valid_until REAL,updated_at REAL NOT NULL,PRIMARY KEY(owner,agent_id,conversation_id,id))",
        ):
            db.execute(sql)
        db.execute(
            "INSERT INTO agent_memories VALUES(?,?,?,?,?,?,?,?,?)",
            (
                record.owner,
                record.agent_id,
                record.conversation_id,
                record.id,
                canonical_json(record.model_dump(mode="json")).decode(),
                7,
                1,
                None,
                2.0,
            ),
        )
    return database, record


def test_create_correct_disable_enable_delete_are_linked_without_plaintext(store):
    sources = [{"kind": "external", "id": "SYNTHETIC_SOURCE_SECRET", "token": "SYNTHETIC_TOKEN"}]
    record = create(store, actor=ACTOR, sources=sources)
    versions = [record]
    versions.append(
        store.update_memory(
            ACTOR.owner, "worker", "same", record.id, 1, content="SYNTHETIC_CORRECTION", actor=ACTOR
        )
    )
    versions.append(
        store.update_memory(ACTOR.owner, "worker", "same", record.id, 2, enabled=False, actor=ACTOR)
    )
    versions.append(
        store.update_memory(ACTOR.owner, "worker", "same", record.id, 3, enabled=True, actor=ACTOR)
    )
    store.delete_memory(ACTOR.owner, "worker", "same", record.id, actor=ACTOR)
    rows = audit(store, record)
    assert [r.operation for r in rows] == ["create", "update", "disable", "enable", "delete"]
    assert [(r.old_version, r.new_version) for r in rows] == [
        (None, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, None),
    ]
    assert rows[0].before_digest is None and rows[-1].after_digest is None
    for i, version in enumerate(versions):
        assert rows[i].after_digest == content_digest(version.model_dump(mode="json"))
        assert rows[i + 1].before_digest == rows[i].after_digest
    for row in rows:
        assert row.actor_context == "host-identity" and row.actor_owner == ACTOR.owner
        assert row.actor_permission_version == 3
        assert row.actor_identity_digest == content_digest(ACTOR.model_dump(mode="json"))
        assert row.source_fingerprints == [content_digest(sources[0])]
    encoded = canonical_json([r.model_dump(mode="json") for r in rows]).decode()
    assert all(
        secret not in encoded
        for secret in (
            "SYNTHETIC_PRIVATE_BODY",
            "SYNTHETIC_CORRECTION",
            "SYNTHETIC_SOURCE_SECRET",
            "SYNTHETIC_TOKEN",
        )
    )
    assert snapshot(store)["agent_memories"] == []


def test_legacy_call_contract_does_not_invent_authenticated_identity(store):
    record = create(store)
    store.update_memory(ACTOR.owner, "worker", "same", record.id, 1, enabled=False)
    store.delete_memory(ACTOR.owner, "worker", "same", record.id)
    rows = audit(store, record)
    assert len(rows) == 3
    assert all(r.actor_context == "legacy-owner" and r.actor_owner == ACTOR.owner for r in rows)
    assert all(r.actor_permission_version is None and r.actor_identity_digest is None for r in rows)


def test_identity_fingerprint_sorts_scopes_without_storing_their_values(store):
    actor = ACTOR.model_copy(update={"scopes": frozenset({"read", "SYNTHETIC_SCOPE_SECRET"})})
    record = create(store, actor=actor)
    (row,) = audit(store, record)
    expected = content_digest(actor.model_dump(mode="json") | {"scopes": sorted(actor.scopes)})
    assert row.actor_identity_digest == expected
    assert "SYNTHETIC_SCOPE_SECRET" not in row.model_dump_json()


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
def test_audit_write_failure_rolls_back_actual_mutation(store, operation):
    record = create(store)
    before = snapshot(store)
    with store.connection() as db:
        db.execute(
            "CREATE TRIGGER synthetic_audit_failure BEFORE INSERT ON agent_memory_audit BEGIN SELECT RAISE(ABORT,'synthetic failure'); END"
        )
    with pytest.raises(AgentError) as error:
        if operation == "create":
            create(store)
        elif operation == "update":
            store.update_memory(ACTOR.owner, "worker", "same", record.id, 1, enabled=False)
        else:
            store.delete_memory(ACTOR.owner, "worker", "same", record.id)
    assert error.value.code == "storage_unavailable" and snapshot(store) == before


@pytest.mark.parametrize("operation", ["create", "update", "delete"])
def test_mismatched_actor_cannot_commit_or_leave_a_success_log(store, operation):
    record = create(store)
    before = snapshot(store)
    other = Identity(owner="tenant_b_user")
    with pytest.raises(AgentError) as error:
        if operation == "create":
            create(store, actor=other)
        elif operation == "update":
            store.update_memory(
                ACTOR.owner, "worker", "same", record.id, 1, content="hijack", actor=other
            )
        else:
            store.delete_memory(ACTOR.owner, "worker", "same", record.id, actor=other)
    assert error.value.code == "memory_actor" and snapshot(store) == before


def test_version_conflict_and_wrong_namespace_do_not_append_audit(store):
    record = create(store)
    store.update_memory(ACTOR.owner, "worker", "same", record.id, 1, content="corrected")
    before = snapshot(store)
    for owner, agent, conversation, version in (
        (ACTOR.owner, "worker", "same", 1),
        ("tenant_b_user", "worker", "same", 2),
        (ACTOR.owner, "other", "same", 2),
        (ACTOR.owner, "worker", "other", 2),
    ):
        with pytest.raises(AgentError):
            store.update_memory(owner, agent, conversation, record.id, version, content="hijack")
    assert snapshot(store) == before


def test_two_connections_competing_for_one_version_log_only_the_winner(store):
    record = create(store)
    barrier = Barrier(2)

    def correct(text):
        barrier.wait(timeout=5)
        try:
            store.update_memory(ACTOR.owner, "worker", "same", record.id, 1, content=text)
            return "updated"
        except AgentError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(correct, ["one", "two"])) == ["memory_conflict", "updated"]
    assert [r.operation for r in audit(store, record)] == ["create", "update"]


def test_migrate_populated_v1_and_repeat_install_without_fabricated_history(tmp_path):
    database, legacy = legacy_database(tmp_path)
    store = SqliteContextStore(database.connection)
    before = store.memory(ACTOR.owner, "worker", "same", legacy.id)
    for _ in range(2):
        with database.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            store.install(db)
            assert (
                db.execute("SELECT value FROM agent_context_meta WHERE key='schema'").fetchone()[0]
                == "2"
            )
    assert store.memory(ACTOR.owner, "worker", "same", legacy.id) == before
    assert audit(store, legacy) == []
    store.update_memory(ACTOR.owner, "worker", "same", legacy.id, 7, content="corrected")
    (event,) = audit(store, legacy)
    assert (event.old_version, event.new_version) == (7, 8)


def test_failed_migration_rolls_back_schema_and_is_retryable(tmp_path):
    database, record = legacy_database(tmp_path)
    with pytest.raises(AgentError):
        with database.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.set_authorizer(
                lambda action, *args: (
                    sqlite3.SQLITE_DENY
                    if action == sqlite3.SQLITE_CREATE_INDEX
                    else sqlite3.SQLITE_OK
                )
            )
            SqliteContextStore.install(db)
    with database.connection() as db:
        assert (
            db.execute("SELECT value FROM agent_context_meta WHERE key='schema'").fetchone()[0]
            == "1"
        )
        assert not db.execute(
            "SELECT name FROM sqlite_master WHERE name='agent_memory_audit'"
        ).fetchone()
        db.execute("BEGIN IMMEDIATE")
        SqliteContextStore.install(db)
    store = SqliteContextStore(database.connection)
    assert store.memory(ACTOR.owner, "worker", "same", record.id) == record


def test_legacy_install_without_begin_preserves_caller_transaction_control(tmp_path):
    database = SqliteDatabase(tmp_path / "legacy-installer.sqlite")
    with database.connection() as db:
        assert not db.in_transaction
        SqliteContextStore.install(db)
        assert db.in_transaction
        db.rollback()
        assert not db.execute(
            "SELECT name FROM sqlite_master WHERE name='agent_context_meta'"
        ).fetchone()
        SqliteContextStore.install(db)
    with database.connection() as db:
        assert (
            db.execute("SELECT value FROM agent_context_meta WHERE key='schema'").fetchone()[0]
            == "2"
        )


def test_unknown_schema_is_rejected_before_new_audit_ddl(tmp_path):
    database, _ = legacy_database(tmp_path)
    with database.connection() as db:
        db.execute("UPDATE agent_context_meta SET value='99' WHERE key='schema'")
    with pytest.raises(AgentError, match="migration"):
        with database.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            SqliteContextStore.install(db)
    with database.connection() as db:
        assert not db.execute(
            "SELECT name FROM sqlite_master WHERE name='agent_memory_audit'"
        ).fetchone()


@pytest.mark.parametrize("guard_result", [False, None, 1])
def test_audit_read_requires_fresh_explicit_host_authorization(store, guard_result):
    record = create(store)
    with pytest.raises(AgentError) as error:
        store.memory_audit(ACTOR, "worker", "same", record.id, guard=lambda db: guard_result)
    assert error.value.code == "memory_audit_forbidden"


def test_audit_read_rechecks_policy_and_rejects_async_or_client_identity(store):
    record = create(store)
    decisions = iter([True, False])
    with pytest.raises(AgentError):
        store.memory_audit(ACTOR, "worker", "same", record.id, guard=lambda db: next(decisions))

    async def async_guard(db):
        return True

    with pytest.raises(AgentError):
        store.memory_audit(ACTOR, "worker", "same", record.id, guard=async_guard)
    with pytest.raises(AgentError):
        store.memory_audit(
            {"owner": ACTOR.owner}, "worker", "same", record.id, guard=lambda db: True
        )


def test_audit_reads_are_owner_namespace_scoped_and_bounded(store):
    record = create(store)
    other = store.create_memory("tenant_b_user", "worker", "same", "other")
    assert not store.memory_audit(
        Identity(owner="tenant_b_user"), "worker", "same", record.id, guard=lambda db: True
    )
    assert not audit(store, other)
    assert not store.memory_audit(ACTOR, "other", "same", record.id, guard=lambda db: True)
    assert not store.memory_audit(ACTOR, "worker", "other", record.id, guard=lambda db: True)
    store.update_memory(ACTOR.owner, "worker", "same", record.id, 1, enabled=False)
    (first,) = audit(store, record, limit=1)
    (second,) = audit(store, record, after=first.sequence, limit=1)
    assert second.sequence > first.sequence and second.operation == "disable"


@pytest.mark.parametrize(
    "page",
    [
        {"limit": 0},
        {"limit": 201},
        {"limit": True},
        {"after": -1},
        {"after": True},
        {"after": 1 << 63},
    ],
)
def test_invalid_audit_pages_are_rejected(store, page):
    with pytest.raises(AgentError, match="page"):
        audit(store, create(store), **page)


@pytest.mark.parametrize("payload", ["not-json", '{"owner":"tenant_b_user"}'])
def test_corrupt_audit_payload_fails_closed(store, payload):
    record = create(store)
    with store.connection() as db:
        db.execute("UPDATE agent_memory_audit SET payload=?", (payload,))
    with pytest.raises(AgentError) as error:
        audit(store, record)
    assert error.value.code == "memory_audit_storage"


@pytest.mark.parametrize("change", ["revoke", "disable", "delete", "expire"])
async def test_audit_does_not_revive_derived_memory_or_replace_source_authority(
    store, monkeypatch, change
):
    allowed = [True]

    async def source_checker(identity, agent, source):
        return allowed[0] and source == {"kind": "external", "id": "synthetic"}

    provider = ContextProvider(store, store, source_checker)
    until = time.time() + 60
    root = create(store, sources=[{"kind": "external", "id": "synthetic"}], valid_until=until)
    derived = store.create_memory(
        ACTOR.owner, "worker", "same", "derived", sources=[provider.memory_reference(root)]
    )
    context = RunContext(
        identity=ACTOR,
        agent_id="worker",
        run_id="synthetic",
        conversation_id="same",
        epoch=0,
        deadline=until,
        config_digest="sha256:" + "0" * 64,
    )
    definition = AgentDefinition.model_validate(
        {
            "id": "worker",
            "description": "synthetic",
            "model": "unused",
            "prompt": {"system_template": "policy"},
            "input_schema": {"type": "string"},
            "context": {"memory_scope": "conversation"},
        }
    )
    provider.commit_final(
        context,
        [
            Message(
                role="assistant", content="derived", sources=[provider.memory_reference(derived)]
            )
        ],
        guard=lambda db: True,
    )
    assert (await provider.load(definition, context)).history
    if change == "revoke":
        allowed[0] = False
    elif change == "disable":
        store.update_memory(ACTOR.owner, "worker", "same", root.id, 1, enabled=False)
    elif change == "delete":
        store.delete_memory(ACTOR.owner, "worker", "same", root.id)
    else:
        monkeypatch.setattr(time, "time", lambda: until + 1)
    rows = audit(store, root)
    assert rows  # Historical evidence remains readable only through the host's audit policy.
    expected = 2 if change in {"disable", "delete"} else 1
    assert len(rows) == expected  # Reads, external revocation and expiry invent no mutation.
    bundle = await provider.load(definition, context)
    assert not bundle.history and derived.id not in {m.id for m in bundle.memories}
    assert not await provider.authorized_sources(
        ACTOR, "worker", [rows[0].model_dump(mode="json")], memory_conversation="same"
    )


def test_audit_payload_scope_corruption_is_not_returned_as_another_owner(store):
    record = create(store)
    with store.connection() as db:
        payload = json.loads(db.execute("SELECT payload FROM agent_memory_audit").fetchone()[0])
        payload["owner"] = "tenant_b_user"
        db.execute("UPDATE agent_memory_audit SET payload=?", (json.dumps(payload),))
    with pytest.raises(AgentError, match="namespace"):
        audit(store, record)
