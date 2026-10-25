"""Bounded in-process ASGI load measurement. Not a network or production capacity benchmark."""

import argparse
import asyncio
import ctypes
import json
import math
import os
import platform
import shutil
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from raglab.api import create_app
from raglab.config import Settings
from raglab.models import BuildInput

ROOT = Path(__file__).resolve().parents[1]


def peak_rss():
    if os.name != "nt":
        import importlib

        resource = importlib.import_module("resource")
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
            1 if platform.system() == "Darwin" else 1024
        )

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in (
                "peak_working",
                "working",
                "peak_paged",
                "paged",
                "peak_nonpaged",
                "nonpaged",
                "pagefile",
                "peak_pagefile",
            )
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    values = Counters()
    values.cb = ctypes.sizeof(values)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(values), values.cb):
        return None
    return values.peak_working


def stats(records, elapsed, cpu):
    times = sorted(r["ms"] for r in records)

    def quantile(p):
        return round(times[max(0, math.ceil(len(times) * p) - 1)], 3)

    return {
        "requests": len(records),
        "elapsed_seconds": round(elapsed, 3),
        "throughput_per_second": round(len(records) / elapsed, 3),
        "p50_ms": quantile(0.50),
        "p95_ms": quantile(0.95),
        "p99_ms": quantile(0.99),
        "error_rate": sum(r["error"] for r in records) / len(records),
        "degradation_rate": sum(r.get("degraded", False) for r in records) / len(records),
        "process_cpu_seconds": round(cpu, 3),
        "peak_rss_bytes": peak_rss(),
        "records": records,
    }


async def batch(count, concurrency, fn):
    semaphore = asyncio.Semaphore(concurrency)

    async def one(i):
        async with semaphore:
            started = time.perf_counter()
            item = await fn(i)
            return {"ms": (time.perf_counter() - started) * 1000, **item}

    started, cpu = time.perf_counter(), time.process_time()
    records = await asyncio.gather(*(one(i) for i in range(count)))
    return stats(records, time.perf_counter() - started, time.process_time() - cpu)


async def main(output, count, concurrency):
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        knowledge = root / "knowledge"
        shutil.copytree(ROOT / "data/knowledge", knowledge)
        for i in range(32):
            (knowledge / f"synthetic-{i}.md").write_text(
                f"# SyntheticFixture{i}\n\n这是合成干扰资料编号 {i}。隔离演示服务每周核验审计与有效性。"
                * 6,
                encoding="utf-8",
            )
        app = create_app(
            Settings(data_dir=root / "state", knowledge_root=knowledge, rate_per_minute=100000)
        )
        async with app.router.lifespan_context(app):
            lab = app.state.service
            begin = time.perf_counter()
            manifest = (await lab.build("local", BuildInput()))["manifest"]
            build_ms = (time.perf_counter() - begin) * 1000
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8001"
            ) as client:

                async def query(i, warm=False):
                    response = await client.post(
                        "/api/v1/search",
                        json={
                            "question": "RAG 为什么要把文档分块？",
                            "conversation_id": f"{'warm' if warm else 'cold'}{i % 4 if warm else i}",
                        },
                    )
                    body = response.json()
                    return {
                        "error": response.status_code != 200,
                        "status_code": response.status_code,
                        "degraded": bool(body.get("degradation")),
                        "stages_ms": body.get("stages_ms", {}),
                    }

                cold = await batch(count, concurrency, query)
                for i in range(4):
                    await query(i, True)
                before = lab.cache.hits
                warm = await batch(count, concurrency, lambda i: query(i, True))
                warm["cache_hits"] = lab.cache.hits - before
                submits = []

                async def run(i):
                    begin = time.perf_counter()
                    response = await client.post(
                        "/api/v1/agent/runs",
                        headers={"Idempotency-Key": f"load-{i}"},
                        json={
                            "message": "RAG 为什么分块？；长期记忆如何过滤？",
                            "conversation_id": f"run{i}",
                            "timeout_seconds": 60,
                        },
                    )
                    submits.append(
                        {
                            "ms": (time.perf_counter() - begin) * 1000,
                            "error": response.status_code != 202,
                        }
                    )
                    if response.status_code != 202:
                        return {"error": True, "status": response.status_code}
                    run_id = response.json()["run_id"]
                    async with asyncio.timeout(65):
                        while True:
                            body = (await client.get("/api/v1/agent/runs/" + run_id)).json()
                            if body["status"] not in ("queued", "running"):
                                return {
                                    "error": body["status"] != "succeeded",
                                    "status": body["status"],
                                    "degraded": any(
                                        r["value"].get("degradation")
                                        for r in body["results"].values()
                                    ),
                                }
                            await asyncio.sleep(0.015)

                completed = await batch(16, 2, run)
                submission = stats(submits, completed["elapsed_seconds"], 0)
                submission["throughput_note"] = (
                    "admissions divided by full completion-batch wall time; not saturation admission capacity"
                )
            report = {
                "created_utc": datetime.now(UTC).isoformat(),
                "python": platform.python_version(),
                "platform": platform.platform(),
                "logical_cpus": os.cpu_count(),
                "method": "in-process HTTP ASGI transport; real graph and SQLite; no sockets/TLS/proxy; search endpoint currently executes the full evidence-generation pipeline",
                "concurrency": concurrency,
                "agent_concurrency": 2,
                "index": manifest,
                "build_ms": build_ms,
                "cold_search": cold,
                "warm_search": warm,
                "agent_submission": submission,
                "agent_completion": completed,
                "model_cost": {
                    "provider_calls": 0,
                    "usd": 0,
                    "note": "offline lexical mode; excludes hardware/electricity",
                },
                "limitations": "small synthetic corpus, one local run, no capacity/SLA inference; stage_ms on cached responses is the original computation trace",
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            for key, expected in (
                ("cold_search", count),
                ("warm_search", count),
                ("agent_submission", 16),
                ("agent_completion", 16),
            ):
                row = report[key]
                if (
                    row["requests"] != expected
                    or row["error_rate"] != 0
                    or row["degradation_rate"] != 0
                ):
                    raise RuntimeError(
                        f"{key}: requests failed or degraded; see the written report"
                    )
                for metric in (
                    "elapsed_seconds",
                    "throughput_per_second",
                    "p50_ms",
                    "p95_ms",
                    "p99_ms",
                    "process_cpu_seconds",
                ):
                    if not math.isfinite(row[metric]) or row[metric] < 0:
                        raise RuntimeError(f"{key}: invalid metric {metric}")
            print(
                json.dumps(
                    {
                        k: {a: b for a, b in v.items() if a != "records"}
                        for k, v in report.items()
                        if k
                        in ("cold_search", "warm_search", "agent_submission", "agent_completion")
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evidence/performance.json"))
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.requests <= 200 or not 1 <= args.concurrency <= 8:
        parser.error("use requests 1..200 and concurrency 1..8")
    asyncio.run(main(args.output, args.requests, args.concurrency))
