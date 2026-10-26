"""Verify the offline entry's boundaries without invoking its subprocess suite."""

import ast
import json
import socket
import subprocess
from pathlib import Path

import pytest
import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

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


@pytest.fixture
def audit_step():
    workflow = yaml.safe_load(
        (verify_offline.ROOT / ".github/workflows/generic-validation.yml").read_text(
            encoding="utf-8"
        )
    )
    return next(
        step["run"]
        for step in workflow["jobs"]["validate"]["steps"]
        if step["name"] == "Audit locked dependencies without fixing them"
    )


@pytest.fixture
def locked_pins(audit_step):
    declarations = ast.Module(
        body=[
            node
            for node in ast.parse(audit_step).body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(declarations, "audit-lock-declarations", "exec"), namespace)
    return namespace["pins"]


def real_lock_requirements(name):
    requirements = [
        Requirement(line)
        for raw in (verify_offline.ROOT / name).read_text(encoding="utf-8").splitlines()
        if (line := raw.strip()) and not line.startswith("#")
    ]
    assert all(
        not item.url
        and not item.extras
        and not item.marker
        and len(item.specifier) == 1
        and next(iter(item.specifier)).operator == "=="
        for item in requirements
    )
    expected = {
        canonicalize_name(item.name): next(iter(item.specifier)).version for item in requirements
    }
    assert len(expected) == len(requirements)
    return expected


@pytest.mark.parametrize("name,count", [("requirements.lock", 64), ("requirements-dev.lock", 99)])
def test_ci_pins_preserves_every_real_lock_requirement(locked_pins, name, count):
    expected = real_lock_requirements(name)
    assert len(expected) == count
    assert locked_pins(verify_offline.ROOT / name) == expected


@pytest.mark.parametrize("name", ["requirements.lock", "requirements-dev.lock"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_ci_pins_accepts_comments_and_blanks_around_real_entries(
    locked_pins, tmp_path, name, newline
):
    lines = (verify_offline.ROOT / name).read_text(encoding="utf-8").splitlines()
    decorated = newline.join(
        ["", "  # 离线格式控制", *("  " + line + "  " for line in lines), " ", "# end", ""]
    )
    target = tmp_path / name
    target.write_bytes(decorated.encode("utf-8"))
    assert locked_pins(target) == real_lock_requirements(name)


@pytest.mark.parametrize(
    "entry",
    [
        "demo>=1.0",
        "demo==*",
        "demo==1.*",
        "demo===arbitrary",
        "demo[extra]==1.0",
        "demo==1.0; python_version < '3.12'",
        "demo @ https://example.invalid/demo.whl",
        "-r requirements.lock",
        "--index-url https://example.invalid/simple",
        "-e .",
        "./local.whl",
        "demo==1.0 --hash=sha256:0000",
        "demo==1.0 # inline comment",
        "demo==not-a-version",
        "demo==",
        "==1.0",
        "demo==1.0==2.0",
        "démo==1.0",
        "-demo==1.0",
        "demo-==1.0",
        "demo==1.0\\\n --hash=sha256:0000",
    ],
)
def test_ci_pins_rejects_unsupported_or_invalid_entries(locked_pins, tmp_path, entry):
    target = tmp_path / "unsupported.lock"
    target.write_text("# header\nvalid==1.0\n" + entry + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"unsupported\.lock:3:"):
        locked_pins(target)


@pytest.mark.parametrize("second", ["demo_one==1.0", "Demo.One==1.0", "demo-one==2.0"])
def test_ci_pins_rejects_duplicate_normalized_names(locked_pins, tmp_path, second):
    target = tmp_path / "duplicate.lock"
    target.write_text("demo-one==1.0\n" + second + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"duplicate\.lock:2:.*duplicate"):
        locked_pins(target)


@pytest.mark.parametrize("text", ["", "\n \n", "# no dependencies\n"])
def test_ci_pins_rejects_empty_locks(locked_pins, tmp_path, text):
    target = tmp_path / "empty.lock"
    target.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        locked_pins(target)


@pytest.mark.parametrize(
    "case",
    [
        "complete",
        "vulnerability",
        "vulnerability_zero_exit",
        "skipped",
        "missing",
        "duplicate",
        "version_mismatch",
        "process_error",
    ],
)
def test_ci_audit_checks_complete_real_locks_offline(audit_step, tmp_path, monkeypatch, case):
    expected = real_lock_requirements("requirements-dev.lock")
    runtime = real_lock_requirements("requirements.lock")
    assert all(expected.get(name) == version for name, version in runtime.items())
    for name in ("requirements.lock", "requirements-dev.lock"):
        (tmp_path / name).write_bytes((verify_offline.ROOT / name).read_bytes())
    monkeypatch.chdir(tmp_path)
    (tmp_path / "evidence/ci").mkdir(parents=True)
    calls = []

    def no_network(*args, **kwargs):
        raise AssertionError("CI audit contract must remain offline")

    monkeypatch.setattr(socket.socket, "connect", no_network)

    def audit_result(argv, **kwargs):
        assert "--strict" in argv and "--disable-pip" in argv and "--no-deps" in argv
        assert "--fix" not in argv and "--ignore-vuln" not in argv
        assert kwargs["timeout"] == 1200
        calls.append(argv)
        dependencies = [
            {"name": name, "version": version, "vulns": []} for name, version in expected.items()
        ]
        code = 0
        if case in {"vulnerability", "vulnerability_zero_exit"}:
            dependencies[0]["vulns"] = [{"id": "SYNTHETIC-AUDIT-CONTROL"}]
            code = 1 if case == "vulnerability" else 0
        if case == "skipped":
            dependencies[0]["skip_reason"] = "synthetic unavailable package"
        if case == "missing":
            dependencies.pop()
        if case == "duplicate":
            dependencies[1] = dependencies[0]
        if case == "version_mismatch":
            dependencies[0]["version"] = "0.0.0"
        if case == "process_error":
            code = 2
        Path("evidence/ci/audit.json").write_text(
            json.dumps({"dependencies": dependencies}), encoding="utf-8"
        )
        return subprocess.CompletedProcess(argv, code)

    monkeypatch.setattr(subprocess, "run", audit_result)
    if case == "complete":
        exec(audit_step, {})
    elif case in {"vulnerability", "process_error"}:
        with pytest.raises(SystemExit) as outcome:
            exec(audit_step, {})
        assert outcome.value.code == (1 if case == "vulnerability" else 2)
    else:
        with pytest.raises(AssertionError):
            exec(audit_step, {})
    assert len(calls) == 1
    process = json.loads(Path("evidence/ci/audit-process.json").read_text(encoding="utf-8"))
    assert process["actual_exit_code"] == {"vulnerability": 1, "process_error": 2}.get(case, 0)
