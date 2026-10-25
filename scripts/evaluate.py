import argparse
import asyncio
import json
import platform
import tempfile
from pathlib import Path

from raglab.config import Settings
from raglab.models import BuildInput, Principal, QueryInput, RunInput, TaskSpec
from raglab.service import LabService

ROOT = Path(__file__).resolve().parents[1]


async def suite(cases, knowledge):
    with tempfile.TemporaryDirectory() as folder:
        lab = await LabService(Settings(data_dir=Path(folder), knowledge_root=knowledge)).start()
        records = []
        try:
            await lab.build("local", BuildInput())
            for case in cases:
                result = await lab.answer(
                    Principal(owner="local"),
                    QueryInput(question=case["question"], conversation_id=case["id"]),
                )
                sources = [h["chunk"]["source"] for h in result["evidence"]]
                rank = sources.index(case["source"]) + 1 if case["source"] in sources else None
                valid = all(
                    c["quote"]
                    in next(
                        h["chunk"]["text"]
                        for h in result["evidence"]
                        if h["chunk"]["chunk_id"] == c["chunk_id"]
                    )
                    and c["index_version"] == result["index_version"]
                    for c in result["citations"]
                )
                support = (
                    all(
                        line in [c["quote"] for c in result["citations"]]
                        for line in result["answer"].splitlines()
                    )
                    if result["status"] == "answered"
                    else None
                )
                records.append(
                    {
                        "id": case["id"],
                        "rank": rank,
                        "expected_source": case["source"],
                        "expected_status": case["status"],
                        "status": result["status"],
                        "status_ok": result["status"] == case["status"],
                        "citation_valid": valid,
                        "extractive_claim_supported": support,
                        "complete": all(t in result["answer"] for t in case["terms"]),
                        "answer": result["answer"],
                        "reason": result["reason"],
                    }
                )
            answerable = [r for r in records if r["expected_source"]]
            refused = [r for r in records if r["status"] == "refused"]
            expected_refused = [r for r in records if r["expected_status"] == "refused"]
            true_refusals = sum(r["status"] == "refused" for r in expected_refused)
            answered = [r for r in records if r["status"] == "answered"]
            return {
                "cases": records,
                "metrics": {
                    "hit_at_4": sum(r["rank"] is not None for r in answerable)
                    / max(1, len(answerable)),
                    "mrr": sum(1 / r["rank"] if r["rank"] else 0 for r in answerable)
                    / max(1, len(answerable)),
                    "status_match": sum(r["status_ok"] for r in records) / len(records),
                    "refusal_precision": true_refusals / max(1, len(refused)),
                    "refusal_recall": true_refusals / max(1, len(expected_refused)),
                    "citation_validity": sum(r["citation_valid"] for r in answered)
                    / max(1, len(answered)),
                    "extractive_support": sum(
                        r["extractive_claim_supported"] is True for r in answerable
                    )
                    / max(1, len(answerable)),
                    "answer_completeness": sum(r["complete"] for r in answerable)
                    / max(1, len(answerable)),
                },
                "denominators": {
                    "cases": len(records),
                    "expected_answerable": len(answerable),
                    "answered_with_citations": len(answered),
                    "expected_refusals": len(expected_refused),
                },
            }
        finally:
            await lab.close()


async def main(output):
    fixed = json.loads((ROOT / "data/evaluation/cases.json").read_text(encoding="utf-8"))
    heldout = json.loads((ROOT / "data/evaluation/heldout.json").read_text(encoding="utf-8"))
    report = {
        "python": platform.python_version(),
        "method": "synthetic lexical evaluation; support means exact quote provenance, not general factual correctness",
        "fixed": await suite(fixed, ROOT / "data/knowledge"),
    }
    with tempfile.TemporaryDirectory() as folder:
        knowledge = Path(folder)
        fixtures = {
            "zephyr.md": "# Zephyr\n\nZephyr 的备份时间是 03:15，Zephyr 禁止外网传输。",
            "expired.md": "<!-- valid_until: 1 -->\n# OldBeacon\n\nOldBeacon 的超时时间是 99 秒。",
            "conflict1.md": "# DeltaService\n\nDeltaService 的超时时间是 30 秒。[fact:DeltaService_timeout=30]",
            "conflict2.md": "# DeltaService\n\nDeltaService 的超时时间是 60 秒。[fact:DeltaService_timeout=60]",
        }
        for name, content in fixtures.items():
            (knowledge / name).write_text(content, encoding="utf-8")
        report["heldout"] = await suite(heldout, knowledge)
    report["scenarios"] = await scenarios()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                name: value["metrics"]
                for name, value in report.items()
                if name in ("fixed", "heldout")
            },
            ensure_ascii=False,
        )
    )
    return (
        0
        if all(
            r["status_ok"] and r["citation_valid"] and r["complete"]
            for name in ("fixed", "heldout")
            for r in report[name]["cases"]
        )
        else 1
    )


async def scenarios():
    with tempfile.TemporaryDirectory() as folder:
        lab = await LabService(
            Settings(data_dir=Path(folder), knowledge_root=ROOT / "data/knowledge")
        ).start()
        try:
            await lab.build("local", BuildInput())
            principal = Principal(owner="local")
            first = await lab.answer(
                principal, QueryInput(question="Orion 服务如何备份？", conversation_id="scenario")
            )
            follow = await lab.answer(
                principal, QueryInput(question="它的超时时间是多少？", conversation_id="scenario")
            )
            plan = [
                TaskSpec(id="a", tool="rag.answer", arguments={"question": "Orion 服务如何备份？"}),
                TaskSpec(
                    id="b",
                    tool="rag.answer",
                    arguments={"question": "它的超时时间是多少？"},
                    depends_on=["a"],
                    use_dependencies=True,
                ),
            ]
            run = await lab.submit(
                principal, "evaluation-dag", RunInput(tasks=plan, conversation_id="scenario-dag")
            )
            async with asyncio.timeout(10):
                while True:
                    result = await lab.view("local", run["run_id"])
                    if result["status"] not in ("queued", "running"):
                        break
                    await asyncio.sleep(0.01)
            values = [first, follow] + [r["value"] for r in result["results"].values()]
            support = all(
                v["status"] == "answered"
                and all(
                    c["quote"]
                    in next(
                        h["chunk"]["text"]
                        for h in v["evidence"]
                        if h["chunk"]["chunk_id"] == c["chunk_id"]
                    )
                    for c in v["citations"]
                )
                for v in values
            )
            resolution = (
                follow["resolved_question"] == "Orion的超时时间是多少？"
                and "8 秒" in follow["answer"]
            )
            dependency = (
                result["status"] == "succeeded"
                and "8 秒" in result["results"]["b"]["value"]["answer"]
            )
            assert resolution and dependency and support
            return {
                "reference_resolution_pass": resolution,
                "dependency_consumption_pass": dependency,
                "citation_provenance_pass": support,
                "followup": follow,
                "dag": result,
                "note": "two small deterministic scenarios; not a general conversation benchmark",
            }
        finally:
            await lab.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evidence/evaluation.json"))
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(args.output)))
