"""Verify the offline entry's boundaries without invoking its subprocess suite."""

import json

import pytest

from scripts import verify_offline


def manifest(root, names):
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts/offline_inputs.json").write_text(
        json.dumps({"schema_version": 1, "files": names}), encoding="utf-8"
    )


def test_environment_disables_remote_modes_plugins_and_package_downloads(tmp_path, monkeypatch):
    for name in (
        "OPENAI_API_KEY",
        "AZURE_OPENAI_API_KEY",
        "LANGSMITH_API_KEY",
        "PYTHONPATH",
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "MYPYPATH",
    ):
        monkeypatch.setenv(name, "synthetic-sensitive-value")
    monkeypatch.setenv("LAB_MODE", "openai")
    env = verify_offline.environment(tmp_path)
    assert env["LAB_MODE"] == "local" and env["PIP_NO_INDEX"] == "1"
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1"
    assert env["LANGSMITH_TRACING"] == env["LANGCHAIN_TRACING_V2"] == "false"
    assert "synthetic-sensitive-value" not in str(env)


@pytest.mark.parametrize(
    "name", ["../outside", "/absolute", "scripts\\file.py", "scripts/./file.py"]
)
def test_input_manifest_rejects_external_or_noncanonical_paths(tmp_path, name):
    manifest(tmp_path, [name])
    with pytest.raises(ValueError):
        verify_offline.inputs(tmp_path)


def test_manifest_rejects_unreviewed_files_and_private_environment(tmp_path):
    names = ["scripts/offline_inputs.json"]
    manifest(tmp_path, names)
    (tmp_path / "scripts/unreviewed.py").write_text("pass", encoding="utf-8")
    with pytest.raises(ValueError, match="unreviewed"):
        verify_offline.inputs(tmp_path)
    (tmp_path / "scripts/unreviewed.py").unlink()
    assert verify_offline.inputs(tmp_path) == names
    (tmp_path / ".env").write_text("SYNTHETIC_PRIVATE_ENV", encoding="utf-8")
    with pytest.raises(ValueError, match="private"):
        verify_offline.inputs(tmp_path)


def test_offline_commands_have_no_online_audit_install_or_weakened_coverage(tmp_path):
    plan = verify_offline.commands(
        ["tests/test_example.py", "agentbase/contracts.py"], tmp_path, tmp_path
    )
    flattened = " ".join(word for _, argv in plan for word in argv)
    assert "pip_audit" not in flattened
    assert not any(argv[:3] == ["-m", "pip", "install"] for _, argv in plan)
    assert dict(plan)["distribution"][1] == "scripts.check_distribution"
    assert "--cov-fail-under=80" in flattened
    tests = dict(plan)["tests"]
    assert "tests/test_example.py" in tests and "--ignore" not in tests
    assert {r["check"]: r["status"] for r in verify_offline.NOT_RUN}["audit"] == "not_executed"
    assert {r["check"]: r["status"] for r in verify_offline.NOT_RUN}["browser_bfcache"] == "blocked"


def test_missing_or_corrupt_results_never_pass(tmp_path):
    assert verify_offline.report_outcomes([], tmp_path)[2] is False
    (tmp_path / "junit.xml").write_text("not XML", encoding="utf-8")
    assert verify_offline.report_outcomes([], tmp_path)[2] is False


def test_dependency_preflight_reports_mismatch_without_installing(tmp_path, monkeypatch):
    (tmp_path / "requirements-dev.lock").write_text("synthetic-package==1.0\n", encoding="utf-8")
    monkeypatch.setattr(verify_offline.metadata, "version", lambda name: "2.0")
    result = verify_offline.installed_lock(tmp_path)
    assert (
        result["mismatched"] == ["synthetic-package"] and result["installation_attempted"] is False
    )


def test_fingerprint_changes_when_a_reviewed_input_changes(tmp_path):
    target = tmp_path / "input.txt"
    target.write_text("before", encoding="utf-8")
    original = verify_offline.fingerprint(tmp_path, ["input.txt"])
    target.write_text("after", encoding="utf-8")
    assert verify_offline.fingerprint(tmp_path, ["input.txt"]) != original


def test_manifest_rejects_unreviewed_root_file(tmp_path):
    manifest(tmp_path, ["scripts/offline_inputs.json"])
    (tmp_path / "private.txt").write_text("synthetic private value", encoding="utf-8")
    with pytest.raises(ValueError, match="unreviewed root"):
        verify_offline.inputs(tmp_path)


@pytest.mark.parametrize(
    "partial,hit,expected", [(False, 9, True), (True, 9, False), (False, 7, False)]
)
def test_report_requires_all_checks_and_original_coverage_thresholds(
    tmp_path, partial, hit, expected
):
    (tmp_path / "junit.xml").write_text(
        '<testsuites><testsuite><testcase classname="technical" name="one"/></testsuite></testsuites>',
        encoding="utf-8",
    )
    files = {
        name + "/module.py": {"summary": {"num_statements": 10, "covered_lines": hit}}
        for name in ("agentbase", "raglab")
    }
    (tmp_path / "coverage.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    (tmp_path / "coverage.xml").write_text(
        f'<coverage lines-valid="20" lines-covered="{hit * 2}"/>', encoding="utf-8"
    )
    records = [
        {"check": name, "exit_code": 0}
        for name, _ in verify_offline.commands(["tests/test_contract.py"], tmp_path, tmp_path)
    ]
    if partial:
        records.pop()
    assert verify_offline.report_outcomes(records, tmp_path)[2] is expected


def test_duplicate_junit_or_mismatched_coverage_is_rejected(tmp_path):
    path = tmp_path / "junit.xml"
    path.write_text(
        '<testsuites><testsuite><testcase classname="a" name="same"/><testcase classname="a" name="same"/></testsuite></testsuites>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicated"):
        verify_offline.junit_summary(path)
    data = {"files": {"raglab/m.py": {"summary": {"num_statements": 10, "covered_lines": 9}}}}
    (tmp_path / "coverage.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "coverage.xml").write_text(
        '<coverage lines-valid="10" lines-covered="8"/>', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="differ"):
        verify_offline.coverage_rates(tmp_path / "coverage.json", tmp_path / "coverage.xml")
