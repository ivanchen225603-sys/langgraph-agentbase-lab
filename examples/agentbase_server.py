"""Trusted HTTP assembly. Classification is an offline protocol fixture.

Run: python -m examples.agentbase_server --data-dir <new-directory>
All Agents and original RAG routes share one engine. No model quality claim.
"""

import argparse
import asyncio
from pathlib import Path

import uvicorn

from agentbase import load_package
from agentbase.contracts import ModelChunk
from agentbase.schema import canonical_json
from agentbase.workflows import register_workflows
from raglab.api import create_app
from raglab.config import Settings


class OfflineFixture:
    native_stream = True

    def __init__(self):
        self.calls = 0
        self.release = asyncio.Event()
        self.finished = False

    async def complete(self, request):
        raise AssertionError("fixture uses native async streaming")

    async def stream(self, request):
        self.calls += 1
        yield ModelChunk(text_delta='{"category":')
        await self.release.wait()
        text = request.messages[-1].content.rstrip()
        yield ModelChunk(text_delta='"question"}' if text.endswith(("?", "？")) else '"statement"}')
        self.finished = True
        yield ModelChunk(finish_reason="stop")

    def count_input_tokens(self, request):
        return len(canonical_json(request.model_dump(mode="json")))

    def count_output_tokens(self, text):
        return len(text.encode("utf-8"))


async def number(arguments, context):
    if context.dependency_results:
        return {"value": sum(value["value"] for value in context.dependency_results.values())}
    return {"value": arguments["x"] * 2}


def configure_registry(registry, package_path=None):
    register_workflows(registry)

    def provider(config):
        fixture = OfflineFixture()
        fixture.release.set()
        return fixture

    registry.register_provider("demo.fixture", provider)
    registry.register_handler("demo.number", number)
    registry.bind(load_package(package_path or Path(__file__).with_name("agentbase.json")))


def app_for(settings, package_path=None):
    return create_app(
        settings, configure_registry=lambda registry: configure_registry(registry, package_path)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(".runtime-agentbase-demo"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    settings = Settings(mode="local", data_dir=args.data_dir, port=args.port)
    uvicorn.run(
        app_for(settings, args.config), host=settings.host, port=settings.port, access_log=False
    )
