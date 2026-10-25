import asyncio
from pathlib import Path

import pytest

from raglab.config import Settings
from raglab.models import BuildInput, Principal
from raglab.service import LabService

ROOT = Path(__file__).resolve().parents[1]
LOCAL = Principal(owner="local")


@pytest.fixture
async def service(tmp_path):
    lab = await LabService(
        Settings(
            data_dir=tmp_path / "state",
            knowledge_root=ROOT / "data/knowledge",
            enable_simulation=True,
        )
    ).start()
    await lab.build("local", BuildInput())
    try:
        yield lab
    finally:
        await lab.close()


async def wait_run(service, run_id, statuses=None):
    desired = statuses or {
        "succeeded",
        "partial",
        "failed",
        "cancelled",
        "timed_out",
        "needs_reconciliation",
        "awaiting_review",
        "interrupted",
    }
    for _ in range(200):
        run = await service.view("local", run_id)
        if run["status"] in desired:
            return run
        await asyncio.sleep(0.01)
    raise AssertionError(f"run did not reach {desired}: {run}")
