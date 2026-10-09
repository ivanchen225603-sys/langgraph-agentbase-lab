"""Three reproducible offline demos, always in a fresh synthetic temporary directory."""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from raglab.config import Settings
from raglab.models import (
    BuildInput,
    MemoryInput,
    MemoryPatch,
    Principal,
    QueryInput,
    RunInput,
    TaskSpec,
)
from raglab.service import LabService

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Principal(owner="local")


async def wait(lab, run_id):
    async with asyncio.timeout(10):
        while True:
            result = await lab.view("local", run_id)
            if result["status"] not in ("queued", "running"):
                return result
            await asyncio.sleep(0.01)


async def main(output):
    with tempfile.TemporaryDirectory() as folder:
        lab = await LabService(
            Settings(data_dir=Path(folder), knowledge_root=ROOT / "data/knowledge")
        ).start()
        try:
            await lab.build("local", BuildInput())
            rag = await lab.answer(LOCAL, QueryInput(question="RAG 为什么要把文档分块？"))
            assert rag["status"] == "answered" and rag["citations"]
            first = await lab.answer(
                LOCAL, QueryInput(question="Orion 服务如何备份？", conversation_id="multi")
            )
            second = await lab.answer(
                LOCAL, QueryInput(question="它的超时时间是多少？", conversation_id="multi")
            )
            assert (
                second["resolved_question"] == "Orion的超时时间是多少？"
                and "8 秒" in second["answer"]
            )
            memory = lab.memories.create(
                "local", MemoryInput(content="我偏好 Python 示例", kind="preference")
            )
            corrected = lab.memories.patch(
                "local",
                memory["memory_id"],
                MemoryPatch(content="我偏好异步 Python 示例", expected_version=1),
            )
            recalled = lab.memories.search("local", "Python 示例")
            assert recalled[0]["version"] == 2
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
                LOCAL,
                "demo-resume",
                RunInput(tasks=plan, pause_for_review=True, timeout_seconds=60),
            )
            paused = await wait(lab, run["run_id"])
            assert paused["status"] == "awaiting_review" and paused["recoverable"]
            await lab.resume(LOCAL, run["run_id"], approve=True)
            resumed = await wait(lab, run["run_id"])
            assert (
                resumed["status"] == "succeeded"
                and "8 秒" in resumed["results"]["b"]["value"]["answer"]
            )
            other = await lab.submit(
                LOCAL,
                "demo-cancel",
                RunInput(
                    message="RAG 为什么分块？；长期记忆如何过滤？",
                    pause_for_review=True,
                    timeout_seconds=60,
                ),
            )
            await wait(lab, other["run_id"])
            cancelled = await lab.cancel("local", other["run_id"])
            assert cancelled["status"] == "cancelled" and not cancelled["recoverable"]
            report = {
                "rag": rag,
                "multi_turn": {
                    "first": first,
                    "second": second,
                    "corrected_memory": corrected,
                    "recalled": recalled,
                },
                "agent": {"paused": paused, "resumed": resumed, "cancelled": cancelled},
                "scope": "temporary synthetic offline fixtures; no provider calls",
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "rag": rag["status"],
                        "followup": second["resolved_question"],
                        "memory_version": corrected["version"],
                        "resumed": resumed["status"],
                        "cancelled": cancelled["status"],
                    },
                    ensure_ascii=False,
                )
            )
        finally:
            await lab.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("evidence/demos.json"))
    asyncio.run(main(parser.parse_args().output))
