"""Actual process crashes at synthetic write receipt windows. No real business integration."""

import asyncio
import json
import os
import sys
from pathlib import Path

from raglab.config import Settings
from raglab.errors import LabError
from raglab.models import Principal, RunInput, TaskSpec
from raglab.service import LabService


async def wait(lab, run_id):
    async with asyncio.timeout(10):
        while True:
            view = await lab.view("local", run_id)
            if view["status"] not in ("queued", "running"):
                return view
            await asyncio.sleep(0.01)


async def main():
    folder, mode = Path(sys.argv[1]), sys.argv[2]
    lab = await LabService(Settings(data_dir=folder, enable_simulation=True)).start()
    try:
        if mode.startswith("crash"):
            receipt = lab.store.receipt
            effect = lab.store.simulation_effect

            def write(key, value):
                print("EFFECT", flush=True)
                effect(key, value)

            def persist(key, value, phase="done"):
                if mode == "crash_done":
                    receipt(key, value, phase)
                os._exit(24)

            lab.store.simulation_effect = write
            lab.store.receipt = persist
            run = await lab.submit(
                Principal(owner="local"),
                "write-window",
                RunInput(
                    tasks=[TaskSpec(id="w", tool="demo.write", arguments={"value": "synthetic"})],
                    timeout_seconds=60,
                ),
            )
            await wait(lab, run["run_id"])
            await lab.resume(Principal(owner="local"), run["run_id"], approve=True)
            await asyncio.sleep(20)
        else:
            row = lab.store.requests("local")[0]
            before = await lab.view("local", row["run_id"])
            rejected = []
            try:
                await lab.resume(Principal(owner="local"), row["run_id"])
            except LabError as exc:
                rejected.append(exc.code)
            if before["status"] == "needs_reconciliation":
                try:
                    lab.compensate("local", row["run_id"])
                except LabError as exc:
                    rejected.append(exc.code)
                after = await lab.view("local", row["run_id"])
            else:
                after = await wait(lab, row["run_id"])
            with lab.store.connection() as db:
                effects = db.execute("SELECT count(*) FROM simulation").fetchone()[0]
            print(
                json.dumps(
                    {"before": before, "after": after, "effects": effects, "rejected": rejected}
                ),
                flush=True,
            )
    finally:
        await lab.close()


if __name__ == "__main__":
    asyncio.run(main())
