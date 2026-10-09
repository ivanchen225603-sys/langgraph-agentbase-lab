import asyncio
import hashlib
import math
import re
from collections import Counter

from langchain_core.embeddings import Embeddings

STOPWORDS = set(
    "的 了 是 为 为什么 如何 什么 哪些 有 和 与 在 吗 呢 它 这个 那个 请 我 你 a an the is are what why how to of for in it its".split()
)


def tokens(text: str) -> list[str]:
    english = re.findall(r"[a-z0-9_]+(?:-[a-z0-9_]+)*", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]+", text)
    return [t for t in english if t not in STOPWORDS] + [
        word[i : i + 2]
        for word in chinese
        for i in range(len(word) - 1)
        if word[i : i + 2] not in STOPWORDS
    ]


def cosine(a, b) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def relevance(question: str, text: str) -> float:
    q = set(tokens(question))
    return len(q & set(tokens(text))) / max(1, len(q))


class LexicalHashEmbeddings(Embeddings):
    """确定性的词／中文二元组哈希，不具备任何神经语义能力。"""

    def __init__(self, dimensions=384):
        self.dimensions = dimensions

    def embed_query(self, text: str) -> list[float]:
        result = [0.0] * self.dimensions
        for word, count in Counter(tokens(text)).items():
            raw = hashlib.blake2b(word.encode(), digest_size=8).digest()
            result[int.from_bytes(raw, "little") % self.dimensions] += 1 + math.log(count)
        norm = math.sqrt(sum(v * v for v in result)) or 1
        return [v / norm for v in result]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(x) for x in texts]

    async def aembed_query(self, text: str) -> list[float]:
        await asyncio.sleep(0)
        return self.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        result = []
        for text in texts:
            await asyncio.sleep(0)
            result.append(self.embed_query(text))
        return result
