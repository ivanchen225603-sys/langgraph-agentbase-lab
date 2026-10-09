import math
import re
import time
import uuid
from pathlib import Path
from typing import TypedDict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

from .errors import LabError
from .models import BuildInput, Chunk, DocumentRecord, IndexManifest
from .persistence import digest
from .resilience import Circuit, guarded, remaining


class IngestState(TypedDict, total=False):
    owner: str
    request: dict
    deadline: float
    documents: list[dict]
    chunks: list[dict]
    vectors: list[list[float]]
    cache: dict
    manifest: dict


def load_documents(root: Path, request: BuildInput, owner, settings, deadline=None):
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise LabError(
            "source_unavailable", "configured knowledge root is unavailable", 404
        ) from exc
    deadline = deadline or time.time() + 10
    if request.paths is not None:
        paths = [root / name for name in request.paths]
    else:
        paths = []
        for scanned, path in enumerate(root.rglob("*"), 1):
            remaining(deadline)
            if scanned > 10000 or len(path.relative_to(root).parts) > 12:
                raise LabError("source_scan_limit", "controlled directory exceeds traversal budget")
            if path.is_file() and path.suffix.lower() in (".md", ".txt"):
                paths.append(path)
                if len(paths) > settings.max_documents:
                    raise LabError("source_limit", "too many source documents")
        paths.sort()
    if not paths or len(paths) > settings.max_documents:
        raise LabError("source_limit", "select 1..max_documents controlled UTF-8 md/txt files")
    result = []
    seen = set()
    for path in paths:
        remaining(deadline)
        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            raise LabError("source_unavailable", "selected source is unavailable", 404) from exc
        if (
            not resolved.is_relative_to(root)
            or path.is_symlink()
            or any(p.is_symlink() for p in path.parents if p != root)
        ):
            raise LabError(
                "source_forbidden", "source path escapes root or contains a symlink", 403
            )
        if resolved in seen:
            continue
        seen.add(resolved)
        if (
            path.suffix.lower() not in (".md", ".txt")
            or path.stat().st_size > settings.max_file_bytes
        ):
            raise LabError("source_limit", "unsupported type or file too large")
        try:
            with path.open("rb") as stream:
                raw = stream.read(settings.max_file_bytes + 1)
            if len(raw) > settings.max_file_bytes:
                raise LabError("source_limit", "file grew beyond byte limit")
            content = raw.decode("utf-8-sig").strip()
        except UnicodeError as exc:
            raise LabError("invalid_utf8", "only UTF-8 Markdown/Text is supported") from exc
        if not content:
            raise LabError("empty_document", "empty documents are rejected")
        source = path.relative_to(root).as_posix()
        title = next((s.lstrip("# ") for s in content.splitlines() if s.startswith("#")), path.stem)
        expiry = re.search(r"^<!-- valid_until: ([0-9.]+) -->\r?$", content, re.M)
        record = DocumentRecord(
            document_id=digest([owner if request.visibility == "private" else "public", source])[
                :24
            ],
            source=source,
            title=title[:200],
            content=content,
            digest=digest(content),
            owner=owner if request.visibility == "private" else "public",
            valid_until=float(expiry[1]) if expiry else None,
        )
        result.append(record.model_dump())
    return result


def split_documents(documents: list[dict], request: BuildInput, max_chunks: int) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=request.chunk_size,
        chunk_overlap=request.overlap,
        add_start_index=True,
        separators=["\n\n", "\n", "。", " ", ""],
    )
    result = []
    for d in documents:
        lc_doc = Document(page_content=d["content"], metadata={"document_id": d["document_id"]})
        for piece in splitter.split_documents([lc_doc]):
            start = piece.metadata["start_index"]
            if (
                start < 0
                or d["content"][start : start + len(piece.page_content)] != piece.page_content
            ):
                raise LabError("split_offset", "splitter did not preserve source offsets")
            cid = digest(
                [
                    d["document_id"],
                    d["digest"],
                    start,
                    piece.page_content,
                    request.chunk_size,
                    request.overlap,
                ]
            )[:32]
            result.append(
                Chunk(
                    chunk_id=cid,
                    document_id=d["document_id"],
                    source=d["source"],
                    title=d["title"],
                    text=piece.page_content,
                    start=start,
                    end=start + len(piece.page_content),
                    digest=digest(piece.page_content),
                ).model_dump()
            )
            if len(result) > max_chunks:
                raise LabError("chunk_limit", "candidate index exceeds configured chunk limit")
    return result


class Ingestion:
    def __init__(self, store, embeddings, settings, embedding_id):
        self.store, self.embeddings, self.settings, self.embedding_id = (
            store,
            embeddings,
            settings,
            embedding_id,
        )
        self.circuit = Circuit()
        graph = StateGraph(IngestState)
        for name in ("parse", "split", "embed", "validate", "publish"):
            graph.add_node(name, getattr(self, name))
        graph.add_edge(START, "parse")
        for left, right in zip(
            ("parse", "split", "embed", "validate"),
            ("split", "embed", "validate", "publish"),
            strict=True,
        ):
            graph.add_edge(left, right)
        graph.add_edge("publish", END)
        self.graph = graph.compile()

    async def parse(self, state):
        remaining(state["deadline"])
        req = BuildInput.model_validate(state["request"])
        if req.visibility == "public" and state["owner"] != "local":
            raise LabError(
                "forbidden", "public synthetic index administration requires local principal", 403
            )
        return {
            "documents": load_documents(
                self.settings.knowledge_root, req, state["owner"], self.settings, state["deadline"]
            )
        }

    async def split(self, state):
        remaining(state["deadline"])
        return {
            "chunks": split_documents(
                state["documents"],
                BuildInput.model_validate(state["request"]),
                self.settings.max_chunks,
            )
        }

    async def embed(self, state):
        chunks, cache, vectors, missing = state["chunks"], {}, [], []
        for i, chunk in enumerate(chunks):
            key = digest([self.embedding_id, chunk["digest"]])
            cached = self.store.cached_vector(key)
            vectors.append(cached)
            if cached is None:
                missing.append((i, key, chunk["text"]))
        for offset in range(0, len(missing), self.settings.embedding_batch):
            batch = missing[offset : offset + self.settings.embedding_batch]
            result = await guarded(
                lambda batch=batch: self.embeddings.aembed_documents([x[2] for x in batch]),
                state["deadline"],
                self.circuit,
            )
            if len(result) != len(batch):
                raise LabError("embedding_invalid", "embedding batch length mismatch")
            for (i, key, _), vector in zip(batch, result, strict=True):
                vectors[i], cache[key] = vector, vector
        version = uuid.uuid4().hex
        dimensions = len(vectors[0]) if vectors else 0
        manifest = IndexManifest(
            owner=state["owner"] if state["request"]["visibility"] == "private" else "public",
            version=version,
            created_at=time.time(),
            embedding=self.embedding_id,
            dimensions=dimensions,
            chunk_size=state["request"]["chunk_size"],
            overlap=state["request"]["overlap"],
            documents=len(state["documents"]),
            chunks=len(chunks),
            reused_vectors=len(chunks) - len(missing),
            content_digest=digest([x["digest"] for x in state["documents"]]),
        )
        return {
            "vectors": vectors,
            "cache": cache,
            "manifest": manifest.model_dump(),
            "chunks": [{**x, "index_version": version} for x in chunks],
        }

    async def validate(self, state):
        remaining(state["deadline"])
        dims = state["manifest"]["dimensions"]
        if (
            not state["chunks"]
            or dims != self.settings.embedding_dimensions
            or len(state["vectors"]) != len(state["chunks"])
        ):
            raise LabError("index_invalid", "empty or inconsistent candidate index")
        if any(len(v) != dims or not all(math.isfinite(x) for x in v) for v in state["vectors"]):
            raise LabError("index_invalid", "invalid vector dimensions or nonfinite values")
        return {}

    async def publish(self, state):
        remaining(state["deadline"])
        self.store.publish(
            state["manifest"], state["documents"], state["chunks"], state["vectors"], state["cache"]
        )
        return {}
