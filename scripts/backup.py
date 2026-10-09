"""Cold paired backup/restore: require stopped service via the same OS process lock."""

import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from raglab.persistence import ProcessLock

FILES = ("business.sqlite", "checkpoints.sqlite")
BACKUP_SCHEMA = 1


def source_files(source):
    """Discover unregistered SQLite resources without automatically trusting them."""
    discovered = {path.name for path in source.glob("*.sqlite")}
    unregistered = sorted(discovered - set(FILES))
    if unregistered:
        raise ValueError("unregistered SQLite resources: " + ", ".join(unregistered))
    for name in FILES:
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("required SQLite resource must be a regular file: " + name)
    return FILES


def restore_manifest(source, files):
    """Validate archive format/resources; this is not an application-release check."""
    path = source / "backup.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("restore requires a regular backup.json manifest")
    try:
        raw = path.read_text(encoding="utf-8")
        metadata = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid backup.json manifest") from exc
    if not isinstance(metadata, dict):
        raise ValueError("backup.json manifest must be an object")
    if type(metadata.get("schema")) is not int or metadata["schema"] != BACKUP_SCHEMA:
        raise ValueError("unsupported backup format version")
    if metadata.get("mode") not in ("backup", "restore"):
        raise ValueError("invalid backup.json mode")
    listed = metadata.get("files")
    if (
        not isinstance(listed, list)
        or not all(isinstance(name, str) for name in listed)
        or len(listed) != len(set(listed))
        or set(listed) != set(files)
    ):
        raise ValueError("backup.json files must match the registered SQLite resources")
    return raw


def main(mode, source, destination):
    if mode not in ("backup", "restore"):
        raise ValueError("mode must be backup or restore")
    source, destination = source.resolve(), destination.resolve()
    if (
        source == destination
        or source.is_relative_to(destination)
        or destination.is_relative_to(source)
    ):
        raise ValueError("backup locations must be separate non-nested directories")
    if not source.is_dir():
        raise ValueError("source must be an existing data directory")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError("destination must be empty; existing data is never overwritten")
    if (source / "owner.lock").is_symlink():
        raise ValueError("owner.lock must not be a symbolic link")
    lock = ProcessLock(source / "owner.lock")
    lock.acquire()
    try:
        files = source_files(source)
        provenance = restore_manifest(source, files) if mode == "restore" else None
        destination.mkdir(parents=True, exist_ok=True)
        for name in files:
            with (
                closing(
                    sqlite3.connect((source / name).as_uri() + "?mode=ro", uri=True)
                ) as original,
                closing(sqlite3.connect(destination / name)) as copied,
            ):
                original.backup(copied)
                if copied.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise ValueError("SQLite integrity check failed: " + name)
        metadata = {
            "schema": BACKUP_SCHEMA,
            "mode": mode,
            "files": list(files),
            "requirement": "same application release; no automatic interrupted-run replay",
        }
        if provenance is not None:
            (destination / "backup-source.json").write_text(provenance, encoding="utf-8")
        (destination / "backup.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(json.dumps(metadata))
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("backup", "restore"))
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    main(args.mode, args.source, args.destination)
