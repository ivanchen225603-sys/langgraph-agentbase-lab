"""Offline fixture bootstrap for a bearer-demo container; has no remote model mode."""

import asyncio

from raglab.config import Settings
from raglab.models import BuildInput
from raglab.service import LabService


async def main():
    settings = Settings()
    if settings.mode != "local":
        raise RuntimeError("container bootstrap only supports offline synthetic teaching mode")
    lab = await LabService(settings).start()
    try:
        if not lab.store.active_index():
            await lab.build("local", BuildInput())
    finally:
        await lab.close()


if __name__ == "__main__":
    asyncio.run(main())
