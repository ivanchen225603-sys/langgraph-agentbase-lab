"""Offline artifact checks and temporary installation of the wheel just built."""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


async def installed_probe(target):
    import httpx

    import agentbase
    import raglab
    from agentbase.application import create_agent_app, open_runtime
    from agentbase.configuration import AgentPackage
    from agentbase.contracts import AgentError, Identity
    from agentbase.registry import ComponentRegistry
    from agentbase.runtime_contracts import TERMINAL
    from agentbase.workflows import register_workflows

    for package in (agentbase, raglab):
        assert (await asyncio.to_thread(Path(package.__file__).resolve)).is_relative_to(target)
    registry, calls = ComponentRegistry(), []
    register_workflows(registry)

    async def echo(arguments, context):
        calls.append(context.run_id)
        return arguments

    registry.register_handler("package.echo", echo)
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }
    registry.bind(
        AgentPackage.model_validate(
            {
                "tools": [
                    {
                        "name": "package.echo",
                        "handler": "package.echo",
                        "description": "Synthetic package smoke",
                        "input_schema": schema,
                        "output_schema": schema,
                    }
                ],
                "agents": [
                    {
                        "id": "package-check",
                        "description": "Synthetic package smoke",
                        "workflow": "fixed",
                        "prompt": {"system_template": "Run the declared echo."},
                        "input_schema": schema,
                        "tools": ["package.echo"],
                        "tasks": [
                            {
                                "id": "echo",
                                "tool": "package.echo",
                                "input_bindings": {"text": "/text"},
                            }
                        ],
                    }
                ],
            }
        )
    )
    actor = Identity(owner="package-fixture", scopes=frozenset({"read"}))

    async def resolve(owner):
        assert owner == actor.owner
        return actor

    async def authenticate(request):
        if request.headers.get("authorization") != "Bearer local-package-fixture":
            raise AgentError("unauthorized", "synthetic package credential required", 401)
        return actor

    with tempfile.TemporaryDirectory(prefix="installed-runtime-") as directory:

        @asynccontextmanager
        async def factory():
            async with open_runtime(
                registry, data_dir=Path(directory), identity_resolver=resolve
            ) as engine:
                yield engine

        app = create_agent_app(factory, authenticate=authenticate, refresh_identity=authenticate)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://test",
                headers={"authorization": "Bearer local-package-fixture"},
            ) as client:
                engine, base = app.state.engine, "/api/v1/agents/package-check/runs"
                original = engine.update_checkpoint

                async def fault(*args, **kwargs):
                    raise AgentError(
                        "checkpoint_unavailable", "synthetic initialization fault", 503
                    )

                engine.update_checkpoint = fault
                try:
                    response = await client.post(
                        base,
                        json={"input": {"text": "installed synthetic"}},
                        headers={"Idempotency-Key": "package-recovery"},
                    )
                finally:
                    engine.update_checkpoint = original
                assert response.status_code == 503
                response = await client.get(base + "?limit=20")
                assert response.status_code == 200 and len(response.json()["runs"]) == 1
                run_id = response.json()["runs"][0]["run_id"]
                path = base + "/" + run_id
                before = (await client.get(path)).json()
                assert before["status"] == "initialization_incomplete" and before["recoverable"]
                assert before["interrupts"] == [] and calls == []
                assert (await client.post(path + "/resume", json={})).status_code == 200
                async with asyncio.timeout(10):
                    while True:
                        view = (await client.get(path)).json()
                        if view["status"] in TERMINAL:
                            break
                        await asyncio.sleep(0.01)
                assert view["status"] == "succeeded"
                assert view["result"]["echo"]["value"] == {"text": "installed synthetic"}
                assert calls == [run_id]
                assert (await client.post(path + "/resume", json={})).status_code == 409
                with engine.store.connection() as db:
                    counts = {
                        name: db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                        for name in (
                            "requests",
                            "agent_tool_operations",
                            "agent_reviews",
                            "agent_memories",
                            "agent_model_reservations",
                        )
                    }
                assert counts == {
                    "requests": 1,
                    "agent_tool_operations": 1,
                    "agent_reviews": 0,
                    "agent_memories": 0,
                    "agent_model_reservations": 0,
                }

    def package_origins():
        return {
            name: str(Path(module.__file__).resolve())
            for name, module in sys.modules.items()
            if name.split(".")[0] in {"agentbase", "raglab"} and getattr(module, "__file__", None)
        }

    origins = await asyncio.to_thread(package_origins)
    assert all(Path(origin).is_relative_to(target) for origin in origins.values())
    return {
        "runtime_imports_target_only": True,
        "origins": origins,
        "explicit_resume": True,
        "counts": counts,
        "browser_verified": False,
    }


def run_child(args, cwd, env, log):
    result = subprocess.run(
        [sys.executable, *args],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError("distribution subprocess failed; see " + log.name)
    return result.stdout


def check(build, output):
    wheels, sources = list(build.glob("*.whl")), list(build.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("exactly one local wheel and sdist required")
    runtime = [
        path
        for folder in ("agentbase", "raglab")
        for path in (ROOT / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    ]
    packages = []
    for artifact in [*wheels, *sources]:
        if artifact.suffix == ".whl":
            with zipfile.ZipFile(artifact) as archive:
                members = {
                    name: archive.read(name)
                    for name in archive.namelist()
                    if not name.endswith("/")
                }
        else:
            with tarfile.open(artifact) as archive:
                members = {
                    member.name.split("/", 1)[1]: archive.extractfile(member).read()
                    for member in archive.getmembers()
                    if member.isfile()
                }
        assert all(
            members[path.relative_to(ROOT).as_posix()] == path.read_bytes() for path in runtime
        )
        forbidden = [
            name
            for name in members
            if any(
                part in {".git", ".venv", "evidence", "__pycache__"} or part.startswith(".runtime")
                for part in Path(name).parts
            )
            or name.endswith((".db", ".sqlite", ".pyc", ".key", ".pem"))
            or name == ".env"
        ]
        assert not forbidden, "unexpected private/runtime artifact members"
        packages.append(
            {
                "name": artifact.name,
                "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "runtime_files_matched": len(runtime),
            }
        )
    with tempfile.TemporaryDirectory(prefix="package-check-") as directory:
        work = Path(directory)
        target = work / "installed"
        env = dict(
            os.environ, PYTHONPATH=str(target), PIP_NO_INDEX="1", PYTHONDONTWRITEBYTECODE="1"
        )
        run_child(
            [
                "-m",
                "pip",
                "install",
                "--no-index",
                "--no-deps",
                "--no-compile",
                "--target",
                str(target),
                str(wheels[0].resolve()),
            ],
            work,
            env,
            output.with_name("wheel-install.log"),
        )
        smoke = json.loads(
            run_child(
                [str(Path(__file__).resolve()), "--worker", str(target)],
                work,
                env,
                output.with_name("installed-host.log"),
            )
        )
        for path in runtime:
            assert (target / path.relative_to(ROOT)).read_bytes() == path.read_bytes()
        starters = []
        for name in ("extraction", "research", "confirmed-action"):
            destination = work / name
            run_child(
                ["-m", "agentbase.scaffold", name, str(destination)],
                work,
                env,
                output.with_name("scaffold-" + name + ".log"),
            )
            data = json.loads(
                run_child(
                    [str(destination / "run.py"), "--cases", str(destination / "cases.jsonl")],
                    destination,
                    env,
                    output.with_name("starter-" + name + ".log"),
                )
            )
            assert data["passed"] == data["total"] == 10 and data["failed"] == 0
            starters.append(
                {
                    "template": name,
                    "passed": data["passed"],
                    "failed": data["failed"],
                    "real_model_quality_verified": data["real_model_quality_verified"],
                }
            )
    output.write_text(
        json.dumps(
            {
                "packages": packages,
                "installed_host": smoke,
                "starters": starters,
                "scope": "local artifacts and synthetic in-process ASGI only; no browser or model quality proof",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Local wheel/sdist, installed host and three offline starters passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(asyncio.run(installed_probe(args.worker.resolve())), ensure_ascii=False))
    elif args.build_dir and args.output:
        check(args.build_dir.resolve(), args.output.resolve())
    else:
        parser.error("provide build-dir/output or worker")


if __name__ == "__main__":
    main()
