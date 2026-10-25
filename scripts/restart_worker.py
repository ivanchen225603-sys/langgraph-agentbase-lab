"""Crash/resume harness. Only accepts a caller-owned disposable directory."""

import asyncio
import json
import os
import sys
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import StructuredTool

from raglab.config import Settings
from raglab.models import Principal, RunInput, TaskSpec
from raglab.service import LabService
from raglab.tools import ReadArgs, ToolSpec

ROOT = Path(__file__).resolve().parents[1]


async def main():
    folder, mode = Path(sys.argv[1]), sys.argv[2]
    service = await LabService(
        Settings(data_dir=folder, knowledge_root=ROOT / "data/knowledge")
    ).start()

    async def tool(question, config: RunnableConfig):
        ctx = config["configurable"]["tool_context"]
        with service.store.connection() as db:
            operation = dict(
                db.execute(
                    "SELECT * FROM agent_tool_operations WHERE key=?",
                    (ctx["public_operation_key"],),
                ).fetchone()
            )
        print(
            "HANDLER "
            + json.dumps(
                {
                    "key": operation["key"],
                    "token": ctx["attempt_token"],
                    "attempts": operation["attempts"],
                    "call_deadline": operation["call_deadline"],
                }
            ),
            flush=True,
        )
        if mode in {"crash", "crash_resume"}:
            os._exit(23)
        return {"resumed": True}

    service.registry.register(
        ToolSpec(
            "test.restart",
            StructuredTool.from_function(
                coroutine=tool,
                name="test.restart",
                description="subprocess fault harness",
                args_schema=ReadArgs,
            ),
            timeout=60,
        )
    )
    try:
        if mode == "crash":
            await service.submit(
                Principal(owner="local"),
                "subprocess",
                RunInput(
                    tasks=[
                        TaskSpec(id="a", tool="test.restart", arguments={"question": "synthetic"})
                    ],
                    timeout_seconds=60,
                ),
            )
        else:
            run_id = service.store.requests("local")[0]["run_id"]
            before = await service.view("local", run_id)
            await service.resume(Principal(owner="local"), run_id)
            for _ in range(200):
                after = await service.view("local", run_id)
                if after["status"] not in ("queued", "running"):
                    break
                await asyncio.sleep(0.01)
            print(json.dumps({"before": before, "after": after}), flush=True)
        if mode in {"crash", "crash_resume"}:
            await asyncio.sleep(10)
    finally:
        await service.close()


if __name__ == "__main__":
    asyncio.run(main())
