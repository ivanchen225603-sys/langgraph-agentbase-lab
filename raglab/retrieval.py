import asyncio
import copy
import math
from typing import Any

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict
from rank_bm25 import BM25Okapi

from .embedding import cosine, relevance, tokens
from .models import RetrievalParams


class SnapshotRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rows: list[Any]
    embeddings: Any
    channel: str
    candidate_pool: int = 24

    def _scores(self, query, vector=None):
        if not self.rows:
            return []
        if self.channel == "vector":
            vector = vector or self.embeddings.embed_query(query)
            scores = [
                cosine(vector, v)
                / (math.sqrt(sum(x * x for x in v)) or 1)
                / (math.sqrt(sum(x * x for x in vector)) or 1)
                for _, v in self.rows
            ]
        else:
            corpus = [tokens(c["text"]) or ["__empty__"] for c, _ in self.rows]
            scores = BM25Okapi(corpus).get_scores(tokens(query)).tolist()
        ranked = sorted(enumerate(scores), key=lambda x: (-x[1], self.rows[x[0]][0]["chunk_id"]))
        return [
            Document(
                page_content=self.rows[i][0]["text"],
                metadata={"chunk": self.rows[i][0], "score": float(score)},
            )
            for i, score in ranked[: self.candidate_pool]
            if relevance(query, self.rows[i][0]["text"]) > 0
        ]

    def _get_relevant_documents(self, query: str, *, run_manager=None):
        return self._scores(query)

    async def _aget_relevant_documents(self, query: str, *, run_manager=None):
        await asyncio.sleep(0)
        vector = await self.embeddings.aembed_query(query) if self.channel == "vector" else None
        return self._scores(query, vector)


def fuse(branches: dict, query: str, params: RetrievalParams) -> list[dict]:
    merged: dict[str, dict] = {}
    weights = {"vector": params.vector_weight, "keyword": 1 - params.vector_weight}
    for channel, result in branches.items():
        for rank, doc in enumerate(result.get("documents", []), 1):
            chunk = doc["chunk"]
            entry = merged.setdefault(
                chunk["chunk_id"],
                {
                    "schema_version": 1,
                    "chunk": chunk,
                    "channels": [],
                    "vector_score": None,
                    "keyword_score": None,
                    "fusion_score": 0.0,
                    "rerank_score": 0.0,
                    "relevance": relevance(query, chunk["text"]),
                },
            )
            entry["channels"].append(channel)
            entry[f"{channel}_score"] = doc["score"]
            entry["fusion_score"] += weights[channel] / (params.rrf_k + rank)
    return sorted(merged.values(), key=lambda x: (-x["fusion_score"], x["chunk"]["chunk_id"]))


def lexical_rerank(hits: list[dict], question: str) -> list[dict]:
    result = copy.deepcopy(hits)
    # 含实体的英文 token 必须完全匹配（Orion != Orion-Test）。
    english = {t for t in tokens(question) if t.isascii() and not t.isdigit()}
    for hit in result:
        exact = (
            len(english & set(tokens(hit["chunk"]["text"]))) / max(1, len(english))
            if english
            else 1
        )
        hit["rerank_score"] = 0.85 * hit["relevance"] + 0.15 * exact
    return sorted(
        result, key=lambda x: (-x["rerank_score"], -x["fusion_score"], x["chunk"]["chunk_id"])
    )


def validate_rerank(canonical, result):
    original = {x["chunk"]["chunk_id"]: x for x in canonical}
    if len(result) != len(canonical) or len({x["chunk"]["chunk_id"] for x in result}) != len(
        result
    ):
        raise ValueError("reranker must return exactly the same candidate IDs")
    for item in result:
        if item["chunk"] != original[item["chunk"]["chunk_id"]]["chunk"] or not math.isfinite(
            item["rerank_score"]
        ):
            raise ValueError("reranker changed evidence or returned nonfinite score")


def mmr(hits, weight):
    if weight is None:
        return hits
    pending = list(hits)
    chosen: list[dict] = []
    while pending:

        def score(hit):
            words = set(tokens(hit["chunk"]["text"]))
            redundancy = max(
                (
                    len(words & set(tokens(x["chunk"]["text"])))
                    / max(1, len(words | set(tokens(x["chunk"]["text"]))))
                    for x in chosen
                ),
                default=0,
            )
            return weight * hit["rerank_score"] - (1 - weight) * redundancy

        best = max(pending, key=score)
        pending.remove(best)
        chosen.append(best)
    return chosen
