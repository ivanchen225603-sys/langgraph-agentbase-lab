import errno
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from raglab.persistence import ProcessLock
from scripts import backup


def make_source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("business.sqlite", "checkpoints.sqlite"):
        with closing(sqlite3.connect(source / name)) as database:
            database.execute("CREATE TABLE marker(value TEXT)")
            database.execute("INSERT INTO marker VALUES (?)", (name,))
            database.commit()
    return source


def manifest(mode="backup"):
    # This is the schema=1 format emitted before this hardening change.
    return {
        "schema": 1,
        "mode": mode,
        "files": ["business.sqlite", "checkpoints.sqlite"],
        "requirement": "same application release; no automatic interrupted-run replay",
    }


def test_round_trip_preserves_data_and_closes_connections(tmp_path):
    source = make_source(tmp_path)
    saved, restored = tmp_path / "saved", tmp_path / "restored"
    backup.main("backup", source, saved)
    backup.main("restore", saved, restored)
    assert json.loads((restored / "backup.json").read_text())["mode"] == "restore"
    assert (restored / "backup-source.json").read_bytes() == (saved / "backup.json").read_bytes()
    for folder in (source, saved, restored):
        for name in ("business.sqlite", "checkpoints.sqlite"):
            with closing(sqlite3.connect(folder / name)) as database:
                assert database.execute("SELECT value FROM marker").fetchone() == (name,)
            # Windows rejects this if the backup left a SQLite connection open.
            path = folder / name
            path.rename(folder / (name + ".closed"))


@pytest.mark.parametrize("mode", ["backup", "restore"])
def test_unregistered_database_is_rejected_before_creating_destination(tmp_path, mode):
    source = make_source(tmp_path)
    (source / "backup.json").write_text(json.dumps(manifest()), encoding="utf-8")
    with closing(sqlite3.connect(source / "documents.sqlite")) as database:
        database.execute("CREATE TABLE document(text TEXT)")
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="unregistered SQLite resources: documents.sqlite"):
        backup.main(mode, source, destination)
    assert not destination.exists()
    lock = ProcessLock(source / "owner.lock")
    lock.acquire()
    lock.close()


@pytest.mark.parametrize("mode", ["backup", "restore"])
@pytest.mark.parametrize("name", ["business.sqlite", "checkpoints.sqlite"])
def test_missing_required_database_is_rejected(tmp_path, mode, name):
    source = make_source(tmp_path)
    (source / name).unlink()
    (source / "backup.json").write_text(json.dumps(manifest()), encoding="utf-8")
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="required SQLite resource"):
        backup.main(mode, source, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "contents, error",
    [
        (None, "restore requires a regular backup.json"),
        (b"{", "invalid backup.json"),
        (b"\xff", "invalid backup.json"),
        (b"[]", "manifest must be an object"),
        (b"null", "manifest must be an object"),
    ],
)
def test_missing_or_unreadable_manifest_is_rejected(tmp_path, contents, error):
    source = make_source(tmp_path)
    if contents is not None:
        (source / "backup.json").write_bytes(contents)
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match=error):
        backup.main("restore", source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("version", [None, 0, 2, True, "1", 1.0])
def test_unsupported_manifest_format_is_rejected(tmp_path, version):
    source = make_source(tmp_path)
    metadata = manifest()
    metadata["schema"] = version
    (source / "backup.json").write_text(json.dumps(metadata), encoding="utf-8")
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="unsupported backup format version"):
        backup.main("restore", source, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "files",
    [
        None,
        "business.sqlite",
        [],
        ["business.sqlite"],
        ["business.sqlite", "checkpoints.sqlite", "documents.sqlite"],
        ["business.sqlite", "checkpoints.sqlite", "business.sqlite"],
        ["business.sqlite", 123],
        ["business.sqlite", {"name": "checkpoints.sqlite"}],
        ["business.sqlite", "../checkpoints.sqlite"],
    ],
)
def test_manifest_files_must_match_registered_resources(tmp_path, files):
    source = make_source(tmp_path)
    metadata = manifest()
    metadata["files"] = files
    (source / "backup.json").write_text(json.dumps(metadata), encoding="utf-8")
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="files must match"):
        backup.main("restore", source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("mode", [None, "other", []])
def test_manifest_mode_must_describe_backup_or_restore(tmp_path, mode):
    source = make_source(tmp_path)
    (source / "backup.json").write_text(json.dumps(manifest(mode)), encoding="utf-8")
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="invalid backup.json mode"):
        backup.main("restore", source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("mode", ["backup", "restore"])
def test_existing_manifest_formats_remain_restorable(tmp_path, mode):
    source = make_source(tmp_path)
    metadata = manifest(mode)
    metadata["files"].reverse()
    original = json.dumps(metadata)
    (source / "backup.json").write_text(original, encoding="utf-8")
    destination = tmp_path / "destination"
    backup.main("restore", source, destination)
    assert (destination / "backup-source.json").read_text(encoding="utf-8") == original


@pytest.mark.parametrize("failure", ["source_connect", "target_connect", "copy", "integrity"])
def test_failed_backup_has_no_success_manifest_and_releases_handles(tmp_path, monkeypatch, failure):
    source = make_source(tmp_path)
    destination = tmp_path / "destination"
    real_connect = sqlite3.connect
    opened = []

    class Connection(sqlite3.Connection):
        def backup(self, destination, *args, **kwargs):
            if failure == "copy" and len(opened) == 4:
                raise sqlite3.OperationalError("synthetic backup failure")
            return super().backup(destination, *args, **kwargs)

        def execute(self, sql, *args, **kwargs):
            if (
                failure == "integrity"
                and self is opened[-1]
                and len(opened) == 4
                and sql == "PRAGMA integrity_check"
            ):
                return super().execute("SELECT 'synthetic integrity failure'")
            return super().execute(sql, *args, **kwargs)

    def connect(*args, **kwargs):
        if (failure == "source_connect" and len(opened) == 2) or (
            failure == "target_connect" and len(opened) == 3
        ):
            raise sqlite3.OperationalError("synthetic connection failure")
        connection = real_connect(*args, factory=Connection, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(backup.sqlite3, "connect", connect)
    with pytest.raises((sqlite3.OperationalError, ValueError), match="failure|integrity"):
        backup.main("backup", source, destination)
    assert not (destination / "backup.json").exists()
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    lock = ProcessLock(source / "owner.lock")
    lock.acquire()
    lock.close()
    monkeypatch.setattr(backup.sqlite3, "connect", real_connect)
    with pytest.raises(ValueError, match="required SQLite resource|restore requires"):
        backup.main("restore", destination, tmp_path / "restored")
    assert not (tmp_path / "restored").exists()


def test_backup_reads_committed_wal_without_copying_sidecars(tmp_path):
    source = make_source(tmp_path)
    with closing(sqlite3.connect(source / "business.sqlite")) as database:
        database.execute("PRAGMA journal_mode=WAL")
        database.execute("INSERT INTO marker VALUES ('committed WAL content')")
        database.commit()
        assert (source / "business.sqlite-wal").is_file()
        saved = tmp_path / "saved"
        backup.main("backup", source, saved)
        with closing(sqlite3.connect(saved / "business.sqlite")) as copied:
            assert copied.execute("SELECT value FROM marker ORDER BY rowid").fetchall() == [
                ("business.sqlite",),
                ("committed WAL content",),
            ]
        assert {path.name for path in saved.iterdir()} == {
            "business.sqlite",
            "checkpoints.sqlite",
            "backup.json",
        }


@pytest.mark.parametrize("name", ["business.sqlite", "backup.json", "owner.lock"])
def test_symlinked_members_are_rejected(tmp_path, name):
    source = make_source(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("outside resource", encoding="utf-8")
    member = source / name
    if member.exists():
        member.unlink()
    try:
        member.symlink_to(outside)
    except OSError as exc:
        if getattr(exc, "winerror", None) != 1314 and exc.errno not in {errno.EPERM, errno.EACCES}:
            raise
        pytest.skip(f"filesystem does not permit creating symbolic links: {exc}")
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="regular|symbolic link"):
        backup.main("restore", source, destination)
    assert not destination.exists()
    assert outside.read_text(encoding="utf-8") == "outside resource"


def test_destination_and_source_path_guards(tmp_path):
    source = make_source(tmp_path)
    with pytest.raises(ValueError, match="separate non-nested"):
        backup.main("backup", source, source / "nested")
    destination = tmp_path / "destination"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError, match="destination must be empty"):
        backup.main("backup", source, destination)
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="source must be an existing"):
        backup.main("backup", missing, tmp_path / "new-destination")
    assert not missing.exists()


def test_invalid_operation_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="mode must be backup or restore"):
        backup.main("invalid", Path("unused"), tmp_path / "unused")
