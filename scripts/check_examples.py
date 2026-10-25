"""Run the real source examples against disposable stores."""

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

from examples.agentbase_catalog import run_demo as catalog
from examples.agentbase_enhancements import run_demo as enhancements
from examples.agentbase_minimal import run_demo as minimal
from examples.agentbase_offline import run_demo as offline
from examples.intent_synthetic_execution import run_demo as intent_synthetic
from examples.offline_ticket_receipts import OfflineTicketDemo


async def receipts():
    with tempfile.TemporaryDirectory(prefix="receipt-check-") as directory:
        demo = OfflineTicketDemo(Path(directory))
        first = await demo.submit()
        assert first["phase"] == "unknown"
        assert await demo.adapter.query(demo.principal, demo.key) == "applied"
        result = await demo.executor.execute(demo.context, demo.call)
        assert result.succeeded
        assert demo.remote.submissions == demo.remote.committed_tickets == 1
        return {
            "phase": demo.ledger.operation(demo.key)["phase"],
            "submissions": 1,
            "committed_tickets": 1,
        }


async def main(output):
    report = {
        "minimal": await minimal(),
        "offline": await offline(),
        "catalog": await catalog(),
        "intent_synthetic": await intent_synthetic(),
        "offline_receipts": await receipts(),
        "generic_enhancements": await enhancements(),
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Six technical source examples completed; installed extension is a separate check")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(main(parser.parse_args().output))
