import asyncio
import json
import sqlite3
import sys

import pytest

from tests.conftest import ROOT


async def test_real_process_crash_and_recovery(tmp_path):
    crash = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "scripts.restart_worker",
        str(tmp_path),
        "crash",
        cwd=ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    output, error = await asyncio.wait_for(crash.communicate(), 20)
    assert crash.returncode == 23, (output, error)
    restart = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "scripts.restart_worker",
        str(tmp_path),
        "resume",
        cwd=ROOT,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    output, error = await asyncio.wait_for(restart.communicate(), 20)
    assert restart.returncode == 0, error.decode(errors="replace")
    assert output.count(b"HANDLER ") == 1
    result = json.loads(output.splitlines()[-1])
    assert result["before"]["status"] == "interrupted"
    assert result["after"]["status"] == "succeeded"
    assert result["before"]["deadline"] == result["after"]["deadline"]
    assert result["after"]["results"]["a"]["value"]["resumed"] is True


async def test_read_crash_three_processes_preserve_attempt_budget_and_window(tmp_path):
    async def process(mode):
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "scripts.restart_worker",
            str(tmp_path),
            mode,
            cwd=ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        output, error = await asyncio.wait_for(child.communicate(), 25)
        return child.returncode, output, error

    def records():
        with sqlite3.connect(tmp_path / "business.sqlite") as db:
            db.row_factory = sqlite3.Row
            operation = dict(db.execute("SELECT * FROM agent_tool_operations").fetchone())
            links = [
                dict(row)
                for row in db.execute("SELECT * FROM legacy_action_links ORDER BY attempt")
            ]
            row = dict(db.execute("SELECT * FROM requests").fetchone())
        return operation, links, row

    code, output, error = await process("crash")
    assert code == 23 and output.count(b"HANDLER ") == 1, (output, error)
    first, first_links, first_run = records()
    code, output, error = await process("crash_resume")
    assert code == 23 and output.count(b"HANDLER ") == 1, (output, error)
    second, second_links, second_run = records()
    code, output, error = await process("resume")
    assert code == 0 and b"HANDLER " not in output, (output, error)
    third, third_links, third_run = records()
    report = json.loads(output.splitlines()[-1])
    assert report["after"]["status"] == "failed"
    assert report["after"]["results"]["a"]["error"] == "tool_attempt_budget"
    assert [first["attempts"], second["attempts"], third["attempts"]] == [1, 2, 2]
    assert [len(first_links), len(second_links), len(third_links)] == [1, 2, 2]
    assert first["attempt_token"] != second["attempt_token"] and third["attempt_token"] is None
    assert first_links[0] == second_links[0] == third_links[0]
    assert second_links[1] == third_links[1]
    assert third["phase"] == "failed" and third["error"] == "process_lost"
    for field in ("key", "fingerprint", "owner", "agent_id", "run_id", "binding", "call_deadline"):
        assert first[field] == second[field] == third[field]
    for field in ("run_id", "deadline", "binding", "payload", "epoch", "thread"):
        assert first_run[field] == second_run[field] == third_run[field]


@pytest.mark.parametrize("window", ["done", "unknown"])
async def test_actual_write_receipt_crash_windows(tmp_path, window):
    async def process(mode):
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "scripts.effect_worker",
            str(tmp_path),
            mode,
            cwd=ROOT,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        output, error = await asyncio.wait_for(child.communicate(), 25)
        return child.returncode, output.decode(), error.decode()

    code, output, error = await process("crash_" + window)
    assert code == 24, (output, error)
    assert output.count("EFFECT") == 1
    code, output, error = await process("resume")
    assert code == 0, (output, error)
    assert "EFFECT" not in output
    report = json.loads(output)
    assert report["effects"] == 1
    assert report["before"]["deadline"] == report["after"]["deadline"]
    if window == "done":
        assert report["after"]["status"] == "succeeded"
    else:
        assert report["after"]["status"] == "needs_reconciliation"
        assert report["rejected"] == ["not_recoverable", "not_reconciled"]
