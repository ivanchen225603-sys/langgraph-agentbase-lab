"""Reviewed offline checks only; never reports the full scripts.verify as passed."""

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = {"agentbase", "raglab", "examples", "scripts", "tests", "data", "docs"}
NOT_RUN = [
    {
        "check": "audit",
        "status": "not_executed",
        "reason": "pip_audit queries online advisory services; deferred to the separately authorized delivery stage; not run by this offline entry",
    },
    {
        "check": "original_hygiene",
        "status": "not_executed",
        "reason": "scripts.hygiene also requires private ARCHITECTURE/CI/lifecycle inputs; tracked hygiene below is a narrower check",
    },
    {
        "check": "browser_bfcache",
        "status": "blocked",
        "reason": "prior explicit denied origin approval for http://127.0.0.1:18085; not retried",
    },
    {
        "check": "real_model_quality",
        "status": "not_executed",
        "reason": "only lexical/scripted fixtures are reviewed for this entry",
    },
    {
        "check": "real_environment",
        "status": "not_executed",
        "reason": "Docker/Compose, remote CI, production IdP, suppliers and multi-instance acceptance remain outside this offline entry",
    },
]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_environment():
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in {"PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE", "PYTEST_ADDOPTS"}
        and not key.startswith("LAB_")
    }
    environment.update(
        PYTHONIOENCODING="utf-8",
        LAB_MODE="local",
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
    )
    return environment


def coverage_rates(path, xml=None):
    files = json.loads(path.read_text(encoding="utf-8"))["files"]
    counts = {"agentbase": [0, 0], "raglab": [0, 0]}
    seen = set()
    for filename, data in files.items():
        normalized = filename.replace("\\", "/")
        parts = normalized.split("/")
        if (
            normalized.startswith("/")
            or any(part in {"", ".", ".."} for part in parts)
            or parts[0] not in counts
            or normalized in seen
        ):
            raise ValueError("invalid or duplicate coverage path")
        seen.add(normalized)
        summary = data["summary"]
        total, hit = summary["num_statements"], summary["covered_lines"]
        if type(total) is not int or type(hit) is not int or not 0 <= hit <= total:
            raise ValueError("invalid package coverage counters")
        counts[parts[0]][0] += total
        counts[parts[0]][1] += hit
    if xml is not None:
        document = ET.parse(xml).getroot()
        if int(document.attrib["lines-valid"]) != sum(value[0] for value in counts.values()) or int(
            document.attrib["lines-covered"]
        ) != sum(value[1] for value in counts.values()):
            raise ValueError("coverage JSON/XML totals differ")
    rates = {name: hit / total if total else 0.0 for name, (total, hit) in counts.items()}
    total = sum(total for total, hit in counts.values())
    rates["combined"] = sum(hit for total, hit in counts.values()) / total if total else 0.0
    return rates


def junit_summary(path):
    cases = ET.parse(path).getroot().findall(".//testcase")
    result = {"collected": len(cases), "passed": 0, "failed": 0, "errors": 0, "skipped": []}
    names = set()
    for case in cases:
        name = case.attrib.get("classname", "") + "::" + case.attrib["name"]
        if name in names:
            raise ValueError("JUnit names are duplicated")
        names.add(name)
        statuses = [
            (kind, case.find(kind))
            for kind in ("failure", "error", "skipped")
            if case.find(kind) is not None
        ]
        if len(statuses) > 1:
            raise ValueError("JUnit case has conflicting outcomes")
        if not statuses:
            result["passed"] += 1
        elif statuses[0][0] == "failure":
            result["failed"] += 1
        elif statuses[0][0] == "error":
            result["errors"] += 1
        else:
            node = statuses[0][1]
            result["skipped"].append(
                {
                    "name": name,
                    "type": node.attrib.get("type", ""),
                    "reason": node.attrib.get("message", ""),
                }
            )
    return result


def allowed_platform_skips(summary, platform_name):
    expected = {
        f"tests.test_backup::test_symlinked_members_are_rejected[{name}]"
        for name in ("business.sqlite", "backup.json", "owner.lock")
    }
    return all(
        platform_name.startswith("Windows-")
        and item["name"] in expected
        and item["type"] == "pytest.skip"
        and item["reason"].startswith("filesystem does not permit creating symbolic links:")
        and "[WinError 1314]" in item["reason"]
        for item in summary["skipped"]
    )


def execute(name, args, directory, timeout, environment):
    argv = [sys.executable, *args]
    timer = time.perf_counter()
    started = datetime.now(UTC).isoformat()
    try:
        result = subprocess.run(
            argv,
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        code, output = result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired as error:
        code = 124
        output = "bounded check timed out\n" + str(error.stdout or "") + str(error.stderr or "")
    except OSError as error:
        code, output = 125, type(error).__name__ + ": " + str(error)
    log = directory / (name + ".log")
    log.write_text(output, encoding="utf-8")
    return {
        "check": name,
        "argv": argv,
        "cwd": str(ROOT),
        "started_utc": started,
        "seconds": round(time.perf_counter() - timer, 3),
        "exit_code": code,
        "log": log.relative_to(ROOT).as_posix(),
        "log_sha256": sha256(log),
    }


def inputs(root: Path) -> list[str]:
    manifest = json.loads((root / "scripts/offline_inputs.json").read_text(encoding="utf-8"))
    files = manifest["files"]
    if (
        manifest.get("schema_version") != 1
        or not isinstance(files, list)
        or not files
        or any(not isinstance(name, str) for name in files)
    ):
        raise ValueError("unsupported or empty offline input manifest")
    if len(files) != len(set(files)):
        raise ValueError("duplicate offline input")
    for name in files:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name or str(path) != name:
            raise ValueError("offline inputs must be normalized relative paths")
        target = root / name
        if (
            not target.resolve().is_relative_to(root.resolve())
            or any(
                root.joinpath(*path.parts[:i]).is_symlink() for i in range(1, len(path.parts) + 1)
            )
            or not target.is_file()
        ):
            raise ValueError("missing, symlinked or external offline input: " + name)
    if (root / ".env").exists():
        raise ValueError("use a clean tracked export without a private .env")
    if any(p.is_file() and p.name not in files for p in root.iterdir()):
        raise ValueError("unreviewed root input; use a clean tracked export")
    unexpected = [
        p.relative_to(root).as_posix()
        for folder in FOLDERS
        for p in (root / folder).rglob("*")
        if p.is_file()
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
        and p.relative_to(root).as_posix() not in files
    ]
    if unexpected:
        raise ValueError("use a tracked export; unreviewed development inputs are present")
    return sorted(files)


def fingerprint(root: Path, files: list[str]) -> dict:
    records = {name: sha256(root / name) for name in files}
    return {
        "files": records,
        "digest": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
    }


def environment(evidence: Path) -> dict[str, str]:
    result = clean_environment()
    for key in list(result):
        if key.startswith(
            ("OPENAI_", "AZURE_OPENAI_", "LANGSMITH_", "LANGCHAIN_", "PYTEST_", "COVERAGE_")
        ) or key in {"MYPYPATH", "PYTHONWARNINGS"}:
            del result[key]
    result.update(
        LAB_MODE="local",
        LANGSMITH_TRACING="false",
        LANGCHAIN_TRACING_V2="false",
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PIP_NO_INDEX="1",
        PIP_DISABLE_PIP_VERSION_CHECK="1",
        HF_HUB_OFFLINE="1",
        COVERAGE_FILE=str(evidence / ".coverage"),
    )
    return result


def installed_lock(root: Path) -> dict:
    records, missing, mismatched = {}, [], []
    for line in (root / "requirements-dev.lock").read_text(encoding="utf-8").splitlines():
        if not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+", line):
            raise ValueError("offline development lock must contain portable exact pins only")
        name, expected = line.split("==")
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = None
            missing.append(name)
        if actual is not None and actual != expected:
            mismatched.append(name)
        records[name] = {"expected": expected, "installed": actual}
    return {
        "packages": records,
        "missing": missing,
        "mismatched": mismatched,
        "installation_attempted": False,
    }


def tracked_hygiene(root: Path, files: list[str]) -> dict:
    findings = []
    for name in files:
        text = (root / name).read_text(encoding="utf-8")
        for pattern, rule in (
            (r"sk-[A-Za-z0-9_-]{24,}", "possible OpenAI token"),
            (r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", "private key material"),
        ):
            if re.search(pattern, text):
                findings.append({"file": name, "rule": rule})
        if name.endswith(".lock") and re.search(r"(?m)^.*(?:file://|@\s*(?:[A-Za-z]:|/))", text):
            findings.append({"file": name, "rule": "nonportable lock path"})
    return {
        "files_checked": len(files),
        "findings": findings,
        "scope": "reviewed tracked inputs only; not the original full hygiene check or a complete secret detector",
    }


def commands(files: list[str], evidence: Path, build: Path) -> list[tuple[str, list[str]]]:
    python = [name for name in files if name.endswith(".py")]
    tests = [name for name in python if name.startswith("tests/test_")]
    if not tests:
        raise ValueError("no reviewed tests")
    return [
        ("dependencies", ["-m", "pip", "check"]),
        ("lint", ["-m", "ruff", "check", "--no-respect-gitignore", *python]),
        ("format", ["-m", "ruff", "format", "--check", "--no-respect-gitignore", *python]),
        ("typing", ["-m", "mypy", "agentbase", "raglab", "examples"]),
        (
            "tests",
            [
                "-m",
                "pytest",
                "-p",
                "pytest_asyncio.plugin",
                "-p",
                "pytest_cov.plugin",
                "-p",
                "no:cacheprovider",
                *tests,
                "--junitxml=" + str(evidence / "junit.xml"),
                "--cov=agentbase",
                "--cov=raglab",
                "--cov-report=xml:" + str(evidence / "coverage.xml"),
                "--cov-report=json:" + str(evidence / "coverage.json"),
                "--cov-report=term-missing",
                "--cov-fail-under=80",
            ],
        ),
        ("client", ["-m", "scripts.check_client"]),
        ("examples", ["-m", "scripts.check_examples", "--output", str(evidence / "examples.json")]),
        ("evaluation", ["-m", "scripts.evaluate", "--output", str(evidence / "evaluation.json")]),
        ("demos", ["-m", "scripts.demo", "--output", str(evidence / "demos.json")]),
        (
            "performance",
            ["-m", "scripts.load_test", "--output", str(evidence / "performance.json")],
        ),
        ("build", ["-m", "build", "--no-isolation", "--outdir", str(build)]),
        (
            "distribution",
            [
                "-m",
                "scripts.check_distribution",
                "--build-dir",
                str(build),
                "--output",
                str(evidence / "distribution.json"),
            ],
        ),
    ]


def report_outcomes(records: list[dict], evidence: Path) -> tuple[dict, dict, bool]:
    try:
        tests = junit_summary(evidence / "junit.xml")
        coverage = coverage_rates(evidence / "coverage.json", evidence / "coverage.xml")
    except (OSError, ValueError, KeyError, ET.ParseError) as exc:
        return {"error": type(exc).__name__}, {}, False
    passed = (
        [record["check"] for record in records]
        == [name for name, _ in commands(["tests/test_contract.py"], evidence, evidence)]
        and tests["collected"] > 0
        and tests["failed"] == tests["errors"] == 0
        and allowed_platform_skips(tests, platform.platform())
        and coverage.get("combined", 0) >= 0.80
        and coverage.get("raglab", 0) >= 0.80
        and coverage.get("agentbase", 0) > 0
        and all(record["exit_code"] == 0 for record in records)
    )
    return tests, coverage, passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit) or not 30 <= args.timeout <= 3600:
        parser.error("use a 40-character source commit and timeout 30..3600")
    for target, parent in ((args.evidence_dir, ROOT / "evidence"), (args.build_dir, ROOT / "dist")):
        if (
            not target.resolve().is_relative_to(parent.resolve())
            or not target.resolve().is_relative_to(ROOT.resolve())
            or target.exists()
        ):
            parser.error("use new evidence/build directories within the clean export")
    try:
        files = inputs(ROOT)
        initial = fingerprint(ROOT, files)
        lock = installed_lock(ROOT)
        hygiene = tracked_hygiene(ROOT, files)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    args.evidence_dir.mkdir(parents=True)
    args.build_dir.mkdir(parents=True)
    args.evidence_dir.joinpath("dependency-inventory.json").write_text(
        json.dumps(lock, indent=2), encoding="utf-8"
    )
    args.evidence_dir.joinpath("tracked-hygiene.json").write_text(
        json.dumps(hygiene, indent=2), encoding="utf-8"
    )
    records = []
    if not lock["missing"] and not lock["mismatched"] and not hygiene["findings"]:
        for name, argv in commands(files, args.evidence_dir.resolve(), args.build_dir.resolve()):
            record = execute(
                name,
                argv,
                args.evidence_dir.resolve(),
                args.timeout,
                environment(args.evidence_dir.resolve()),
            )
            records.append(record)
            print(name + ": exit=" + str(record["exit_code"]), flush=True)
    final = fingerprint(ROOT, files)
    tests, coverage, passed = report_outcomes(records, args.evidence_dir)
    passed = (
        passed
        and initial == final
        and bool(records)
        and len(list(args.build_dir.glob("*.whl"))) == 1
        and len(list(args.build_dir.glob("*.tar.gz"))) == 1
    )
    report = {
        "source_commit_supplied_by_exporter": args.source_commit,
        "source_baseline": initial,
        "baseline_unchanged": initial == final,
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "offline_gates_passed": passed,
        "full_verify_passed": False,
        "dependency_preflight_passed": not lock["missing"] and not lock["mismatched"],
        "tracked_hygiene_passed": not hygiene["findings"],
        "checks": records,
        "matrix": [
            {
                "check": r["check"],
                "status": "passed" if r["exit_code"] == 0 else "failed",
                "log": r["log"],
            }
            for r in records
        ]
        + NOT_RUN,
        "test_summary": tests,
        "package_coverage": coverage,
        "network_boundary": "Reviewed lexical/scripted providers, temporary databases and local ASGI/loopback transport tests; no OS network sandbox claim. Only the built local wheel is installed into a temporary target with --no-index/--no-deps; no package download. Online pip_audit excluded.",
        "unexecuted_if_preflight_failed": [
            name for name, _ in commands(files, args.evidence_dir, args.build_dir)
        ]
        if not records
        else [],
        "products_sha256": {p.name: sha256(p) for p in args.evidence_dir.iterdir() if p.is_file()},
        "builds": [
            {"name": p.name, "sha256": sha256(p)} for p in args.build_dir.iterdir() if p.is_file()
        ],
    }
    args.evidence_dir.joinpath("verification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("offline_gates_passed=" + str(passed) + "; full_verify_passed=False", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
