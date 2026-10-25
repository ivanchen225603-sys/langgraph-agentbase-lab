"""Controlled artifact storage and source authorization, with real SQLite transactions."""

import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
from pydantic import ValidationError

from agentbase.artifacts import (
    ArtifactLimits,
    ArtifactOperation,
    ArtifactReference,
    ArtifactScope,
    SqliteArtifactStore,
)
from agentbase.context import ContextProvider
from agentbase.contracts import AgentError, Identity, Message
from agentbase.storage import SqliteContextStore, SqliteDatabase
from tests.test_agentbase_context import context, definition

SCOPE = ArtifactScope(owner="user", agent_id="alpha", namespace="same")


def operation(key, *, run="run", binding="0"):
    return ArtifactOperation(run_id=run, key=key, config_digest="sha256:" + binding * 64)


def allow(db):
    return True


@pytest.fixture
def stores(tmp_path):
    database = SqliteDatabase(tmp_path / "state.sqlite")
    artifacts = SqliteArtifactStore(database.connection)
    memories = SqliteContextStore(database.connection)
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        artifacts.install(db)
        memories.install(db)
    return artifacts, memories


def create(store, key="create", value="original", *, scope=SCOPE, sources=None, expiry=None):
    return store.create(
        scope, operation(key), value, sources=sources or [], valid_until=expiry, guard=allow
    )


def test_constructor_and_explicit_install_do_not_hide_migrations(tmp_path):
    database = SqliteDatabase(tmp_path / "new.sqlite")
    SqliteArtifactStore(database.connection)
    assert not database.path.exists()
    with database.connection() as db:
        with pytest.raises(AgentError) as error:
            SqliteArtifactStore.install(db)
        assert error.value.code == "artifact_schema"
        assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
        db.execute("BEGIN IMMEDIATE")
        db.execute("CREATE TABLE agent_artifact_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        db.execute("INSERT INTO agent_artifact_meta VALUES('schema','999')")
    with database.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        with pytest.raises(AgentError):
            SqliteArtifactStore.install(db)
        assert [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")] == [
            "agent_artifact_meta"
        ]


def test_bound_receipt_lookup_requires_guard_and_current_source_reauthorization(stores):
    store, memories = stores
    original = create(store)
    assert store.receipt(SCOPE, operation("create"), guard=allow) == original
    assert store.receipt(SCOPE, operation("missing"), guard=allow) is None
    with pytest.raises(AgentError) as error:
        store.receipt(SCOPE, operation("create", binding="1"), guard=allow)
    assert error.value.code == "artifact_operation_conflict"
    with pytest.raises(AgentError) as error:
        store.receipt(SCOPE, operation("create"), guard=lambda db: False)
    assert error.value.code == "artifact_guard"
    store.revoke(SCOPE, operation("revoke"), original.id, 1, guard=allow)
    assert store.receipt(SCOPE, operation("create"), guard=allow) == original
    assert not store.read(SCOPE, original.id).enabled


def test_receipt_lookup_migration_retains_old_data_and_rolls_back(stores):
    store, _ = stores
    original = create(store)
    with store.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute("ALTER TABLE agent_artifact_mutations DROP COLUMN config_digest")
        db.execute("UPDATE agent_artifact_meta SET value='1' WHERE key='schema'")
        before = tuple(db.execute("SELECT * FROM agent_artifact_mutations").fetchone())
    with pytest.raises(RuntimeError), store.connection() as db:
        db.execute("BEGIN IMMEDIATE")
        store.install(db)
        raise RuntimeError("migration interrupted")
    with store.connection() as db:
        assert db.execute("SELECT value FROM agent_artifact_meta").fetchone()[0] == "1"
        assert tuple(db.execute("SELECT * FROM agent_artifact_mutations").fetchone()) == before
        db.execute("BEGIN IMMEDIATE")
        store.install(db)
    with store.connection() as db:
        migrated = tuple(db.execute("SELECT * FROM agent_artifact_mutations").fetchone())
        assert migrated[:-1] == before and migrated[-1] is None
    assert create(store) == original
    with pytest.raises(AgentError) as error:
        store.receipt(SCOPE, operation("create"), guard=allow)
    assert error.value.code == "artifact_operation_conflict"


def test_revisions_cas_receipts_revoke_restore_and_reopen(stores):
    store, _ = stores
    original = create(store, sources=[{"kind": "external", "id": "source"}])
    second = store.update(
        SCOPE, operation("update"), original.id, 1, {"items": [2]}, sources=[], guard=allow
    )
    assert second.version == 2 and second.digest != original.digest
    replay = create(store, sources=original.sources)
    assert replay == original
    with pytest.raises(AgentError) as error:
        create(store, value="changed")
    assert error.value.code == "artifact_operation_conflict"
    with pytest.raises(AgentError) as error:
        store.update(SCOPE, operation("stale"), original.id, 1, "stale", sources=[], guard=allow)
    assert error.value.code == "artifact_conflict"
    revoked = store.revoke(SCOPE, operation("revoke"), original.id, 2, guard=allow)
    assert not revoked.enabled and revoked.version == 2
    with pytest.raises(AgentError) as error:
        store.update(
            SCOPE, operation("disabled-update"), original.id, 2, "value", sources=[], guard=allow
        )
    assert error.value.code == "artifact_revoked"
    restored = store.restore(SCOPE, operation("restore"), original.id, 2, 1, guard=allow)
    assert restored.version == 3 and restored.value == original.value
    assert restored.sources == original.sources and restored.enabled
    historic = store.read(SCOPE, original.id, version=1)
    assert historic.model_dump(exclude={"enabled"}) == original.model_dump(exclude={"enabled"})
    assert not historic.enabled
    assert (
        not SqliteArtifactStore(store.connection)
        .revoke(SCOPE, operation("revoke"), original.id, 2, guard=allow)
        .enabled
    )  # Replays the original mutation receipt; does not revoke v3.
    assert store.read(SCOPE, original.id).version == 3 and store.read(SCOPE, original.id).enabled
    with pytest.raises(AgentError) as error:
        store.restore(SCOPE, operation("restore"), original.id, 2, 2, guard=allow)
    assert error.value.code == "artifact_operation_conflict"


@pytest.mark.parametrize(
    "field,value", [("owner", "other"), ("agent_id", "beta"), ("namespace", "other")]
)
def test_all_operations_are_scoped_and_receipts_cannot_cross_namespaces(stores, field, value):
    store, _ = stores
    record = create(store)
    foreign = ArtifactScope.model_validate({**SCOPE.model_dump(), field: value})
    assert store.list(foreign) == []
    for call in [
        lambda: store.read(foreign, record.id),
        lambda: store.update(
            foreign, operation("update"), record.id, 1, "other", sources=[], guard=allow
        ),
        lambda: store.revoke(foreign, operation("revoke"), record.id, 1, guard=allow),
        lambda: store.restore(foreign, operation("restore"), record.id, 1, 1, guard=allow),
    ]:
        with pytest.raises(AgentError) as error:
            call()
        assert error.value.code == "artifact_not_found"
    another = create(store, scope=foreign)
    assert another.id != record.id  # Identical operation keys are isolated too.


def test_guard_and_receipt_failure_roll_back_every_mutation(stores):
    store, _ = stores
    with pytest.raises(AgentError) as error:
        store.create(SCOPE, operation("denied"), "value", sources=[], guard=lambda db: False)
    assert error.value.code == "artifact_guard" and store.list(SCOPE) == []
    original = create(store)
    with store.connection() as db:
        db.execute(
            "CREATE TRIGGER reject_receipt BEFORE INSERT ON agent_artifact_mutations BEGIN SELECT RAISE(ABORT,'injected receipt failure'); END"
        )
    with pytest.raises(AgentError) as error:
        store.update(
            SCOPE, operation("interrupted"), original.id, 1, "new", sources=[], guard=allow
        )
    assert error.value.code == "storage_unavailable"
    assert store.read(SCOPE, original.id) == original
    with store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM agent_artifact_revisions").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM agent_artifact_mutations").fetchone()[0] == 1


def test_concurrent_cas_has_one_winner_and_no_lost_revision(stores):
    store, _ = stores
    original = create(store)
    barrier = Barrier(2)

    def update(index):
        barrier.wait(timeout=3)
        try:
            return store.update(
                SCOPE, operation(f"update{index}"), original.id, 1, index, sources=[], guard=allow
            )
        except AgentError as exc:
            return exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(update, [1, 2]))
    assert results.count("artifact_conflict") == 1
    assert store.read(SCOPE, original.id).version == 2
    with store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM agent_artifact_revisions").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM agent_artifact_mutations").fetchone()[0] == 2


def test_retained_revision_owner_and_mutation_limits_keep_revocation_available(stores):
    store, _ = stores
    limited = SqliteArtifactStore(
        store.connection,
        ArtifactLimits(
            max_namespace_artifacts=1,
            max_owner_artifacts=1,
            max_namespace_mutations=1,
            max_owner_mutations=1,
            max_versions=1,
        ),
    )
    record = create(limited)
    assert create(limited) == record  # Replay is available even at every limit.
    assert not limited.revoke(SCOPE, operation("revoke"), record.id, 1, guard=allow).enabled
    for scope in [SCOPE, SCOPE.model_copy(update={"namespace": "another"})]:
        with pytest.raises(AgentError) as error:
            create(limited, key="another", scope=scope)
        assert error.value.code == "artifact_quota"
    with pytest.raises(AgentError) as error:
        limited.restore(SCOPE, operation("restore"), record.id, 1, 1, guard=allow)
    assert error.value.code == "artifact_quota"
    with store.connection() as db:
        retained = db.execute("SELECT SUM(size_bytes) FROM agent_artifact_revisions").fetchone()[0]
    byte_limited = SqliteArtifactStore(
        store.connection, ArtifactLimits(max_namespace_bytes=retained)
    )
    with pytest.raises(AgentError) as error:
        byte_limited.restore(SCOPE, operation("restore-byte"), record.id, 1, 1, guard=allow)
    assert error.value.code == "artifact_quota"


def test_json_byte_bounds_and_returned_values_do_not_mutate_revisions(stores):
    store, _ = stores
    tiny = SqliteArtifactStore(store.connection, ArtifactLimits(max_content_bytes=5))
    with pytest.raises(AgentError, match="byte"):
        create(tiny, value="\U0001f600")  # UTF-8 JSON is six bytes, not three characters.
    value, sources = {"items": [1]}, [{"kind": "external", "ids": ["one"]}]
    original = create(store, value=value, sources=sources)
    value["items"].append(2)
    sources[0]["ids"].append("two")
    original.value["items"].append(3)
    original.sources[0]["ids"].append("three")
    assert store.read(SCOPE, original.id).value == {"items": [1]}
    assert store.read(SCOPE, original.id).sources == [{"kind": "external", "ids": ["one"]}]
    single = SqliteArtifactStore(store.connection, ArtifactLimits(max_content_bytes=1))
    one = create(single, key="one", value=1)
    assert not single.revoke(SCOPE, operation("one-revoke"), one.id, 1, guard=allow).enabled


async def test_reference_scope_version_digest_expiry_and_fail_closed(stores):
    store, memories = stores

    async def permissive(identity, agent, source):
        return True

    provider = ContextProvider(memories, memories, permissive, artifacts=store)
    original = create(store)

    async def authorized(reference, *, owner="user", namespace="same"):
        return await provider.authorized_sources(
            Identity(owner=owner), "alpha", [reference], artifact_namespace=namespace
        )

    assert await authorized(original.reference())
    assert not await authorized(original.reference(), owner="other")
    assert not await authorized(original.reference(), namespace="other")
    assert not await ContextProvider(memories, memories, permissive).authorized_sources(
        Identity(owner="user"), "alpha", [original.reference()], artifact_namespace="same"
    )
    for invalid in [
        {**original.reference(), "owner": "user"},
        {**original.reference(), "version": True},
        {**original.reference(), "digest": "sha256:" + "f" * 64},
        {k: v for k, v in original.reference().items() if k != "digest"},
    ]:
        assert not await authorized(invalid)
    with pytest.raises(ValidationError):
        ArtifactReference.model_validate({**original.reference(), "owner": "user"})
    expired = create(store, key="expired", expiry=time.time() - 1)
    assert not await authorized(expired.reference())
    store.update(SCOPE, operation("update"), original.id, 1, "new", sources=[], guard=allow)
    assert not await authorized(original.reference())


@pytest.mark.parametrize("memory_scope", ["none", "agent", "conversation"])
async def test_artifact_namespace_is_independent_of_memory_scope(stores, memory_scope):
    store, memories = stores
    record = create(store)
    provider = ContextProvider(memories, memories, artifacts=store)
    messages = [
        Message(role="user", content="Input"),
        Message(role="assistant", content="Result", sources=[record.reference()]),
    ]
    assert provider.commit_final(context(), messages, guard=allow)
    config = definition(context={"memory_scope": memory_scope})
    bundle = await provider.load(config, context())
    assert len(bundle.history) == 2 and await provider.revalidate(bundle, config, context())
    store.revoke(SCOPE, operation("revoke"), record.id, 1, guard=allow)
    assert not await provider.revalidate(bundle, config, context())
    assert not (await provider.load(config, context())).history


async def test_entire_source_closure_is_rechecked_after_external_awaits(stores):
    store, memories = stores
    original = create(store)

    async def checker(identity, agent, source):
        store.update(
            SCOPE, operation("during-await"), original.id, 1, "changed", sources=[], guard=allow
        )
        return True

    provider = ContextProvider(memories, memories, checker, artifacts=store)
    assert not await provider.authorized_sources(
        Identity(owner="user"),
        "alpha",
        [original.reference(), {"kind": "external"}],
        artifact_namespace="same",
    )


async def test_cross_kind_cycle_is_rejected_but_equal_ids_are_not_a_cycle(stores):
    store, memories = stores
    artifact = create(store)
    memory = memories.create_memory(
        "user", "alpha", "same", "value", sources=[artifact.reference()]
    )
    # A deliberately constructed equal identifier across two independent storage kinds.
    same_id = memory.model_copy(update={"id": artifact.id})
    with memories.connection() as db:
        db.execute(
            "UPDATE agent_memories SET id=?,payload=? WHERE id=?",
            (artifact.id, same_id.model_dump_json(), memory.id),
        )
    provider = ContextProvider(memories, memories, artifacts=store)
    args = {"memory_conversation": "same", "artifact_namespace": "same"}
    assert await provider.authorized_sources(
        Identity(owner="user"), "alpha", [provider.memory_reference(same_id)], **args
    )
    cyclic_artifact = store.update(
        SCOPE,
        operation("cycle"),
        artifact.id,
        1,
        "value",
        sources=[provider.memory_reference(same_id)],
        guard=allow,
    )
    cyclic_memory = same_id.model_copy(update={"sources": [cyclic_artifact.reference()]})
    with memories.connection() as db:
        db.execute(
            "UPDATE agent_memories SET payload=? WHERE id=?",
            (cyclic_memory.model_dump_json(), artifact.id),
        )
    assert not await provider.authorized_sources(
        Identity(owner="user"), "alpha", [cyclic_artifact.reference()], **args
    )


async def test_source_depth_bound_preserves_valid_shared_leaves(stores):
    store, memories = stores
    provider = ContextProvider(memories, memories, artifacts=store)
    leaf = create(store)
    shared = create(store, key="shared", sources=[leaf.reference(), leaf.reference()])
    assert await provider.authorized_sources(
        Identity(owner="user"), "alpha", [shared.reference()], artifact_namespace="same"
    )
    current = leaf
    for index in range(17):
        current = create(store, key=f"level{index}", sources=[current.reference()])
        valid = await provider.authorized_sources(
            Identity(owner="user"), "alpha", [current.reference()], artifact_namespace="same"
        )
        assert valid == (index < 16)


@pytest.mark.parametrize(
    "payload",
    ["null", "1", '"text"', "[]", "[" * 2000 + "0" + "]" * 2000, " " * 1114113],
    ids=["null", "integer", "string", "array", "deep", "oversized"],
)
async def test_corrupt_persisted_json_fails_integrity_and_source_authorization(stores, payload):
    store, memories = stores
    original = create(store)
    with store.connection() as db:
        db.execute(
            "UPDATE agent_artifact_revisions SET payload=?,size_bytes=?",
            (payload, len(payload.encode("utf-8"))),
        )
    with pytest.raises(AgentError) as error:
        store.read(SCOPE, original.id)
    assert error.value.code == "artifact_corrupt"
    assert not await ContextProvider(memories, memories, artifacts=store).authorized_sources(
        Identity(owner="user"), "alpha", [original.reference()], artifact_namespace="same"
    )


@pytest.mark.parametrize("method", ["read", "list", "expected", "source"])
def test_sqlite_integer_boundaries_return_contract_errors(stores, method):
    store, _ = stores
    original = create(store)
    with pytest.raises(AgentError) as error:
        if method == "read":
            store.read(SCOPE, original.id, version=2**64)
        elif method == "list":
            store.list(SCOPE, offset=2**64)
        elif method == "expected":
            store.revoke(SCOPE, operation("huge"), original.id, 2**64, guard=allow)
        else:
            store.restore(SCOPE, operation("huge"), original.id, 1, 2**64, guard=allow)
    assert error.value.code == ("artifact_page" if method == "list" else "artifact_version")


def test_byte_quota_uses_actual_persisted_bytes_even_if_accounting_is_corrupt(stores):
    store, _ = stores
    original = create(store)
    with store.connection() as db:
        size = db.execute("SELECT size_bytes FROM agent_artifact_revisions").fetchone()[0]
        db.execute("UPDATE agent_artifact_revisions SET size_bytes=0")
    with pytest.raises(AgentError) as error:
        store.read(SCOPE, original.id)
    assert error.value.code == "artifact_corrupt"
    limited = SqliteArtifactStore(store.connection, ArtifactLimits(max_namespace_bytes=size))
    with pytest.raises(AgentError) as error:
        create(limited, key="another")
    assert error.value.code == "artifact_quota"
    with store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM agent_artifact_revisions").fetchone()[0] == 1


def test_reduced_quotas_still_allow_each_existing_live_head_to_be_revoked(stores):
    store, _ = stores
    originals = [create(store, key=f"create{index}") for index in range(3)]
    reduced = SqliteArtifactStore(
        store.connection,
        ArtifactLimits(
            max_namespace_mutations=1,
            max_owner_mutations=1,
            max_namespace_artifacts=1,
            max_owner_artifacts=1,
        ),
    )
    for index, record in enumerate(originals):
        assert not reduced.revoke(
            SCOPE, operation(f"revoke{index}"), record.id, 1, guard=allow
        ).enabled
    with pytest.raises(AgentError) as error:
        reduced.revoke(SCOPE, operation("extra"), originals[0].id, 1, guard=allow)
    assert error.value.code == "artifact_quota"
    with store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM agent_artifact_mutations").fetchone()[0] == 6


def test_receipt_replay_requires_guard_and_unchanged_configuration_binding(stores):
    store, _ = stores
    record = create(store)
    with pytest.raises(AgentError) as error:
        store.create(SCOPE, operation("create"), record.value, sources=[], guard=lambda db: False)
    assert error.value.code == "artifact_guard"
    with pytest.raises(AgentError) as error:
        store.create(SCOPE, operation("create", binding="1"), record.value, sources=[], guard=allow)
    assert error.value.code == "artifact_operation_conflict"
