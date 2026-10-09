from contextlib import contextmanager
from contextvars import ContextVar

from langchain_core.embeddings import Embeddings

from .errors import LabError
from .resilience import remaining

_scope: ContextVar[tuple[str, int, float] | None] = ContextVar("provider_scope", default=None)


@contextmanager
def provider_scope(scope: str, limit: int, deadline: float):
    token = _scope.set((scope, limit, deadline))
    try:
        yield
    finally:
        _scope.reset(token)


class ProviderBudget:
    def __init__(self, store):
        self.store = store

    def reserve(self):
        scope = _scope.get()
        if scope is None:
            raise LabError(
                "model_budget_context", "provider calls require a bounded request scope", 409
            )
        key, limit, deadline = scope
        remaining(deadline)
        with self.store.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM model_budget WHERE deadline<?", (__import__("time").time(),))
            row = db.execute("SELECT used FROM model_budget WHERE scope=?", (key,)).fetchone()
            used = row[0] if row else 0
            if used >= limit:
                raise LabError(
                    "model_call_budget",
                    "remote call budget exhausted; use current local evidence",
                    409,
                )
            db.execute(
                "INSERT OR REPLACE INTO model_budget VALUES(?,?,?)", (key, used + 1, deadline)
            )


class BudgetedEmbeddings(Embeddings):
    def __init__(self, provider, budget):
        self.provider, self.budget = provider, budget

    def embed_query(self, text):
        self.budget.reserve()
        return self.provider.embed_query(text)

    def embed_documents(self, texts):
        self.budget.reserve()
        return self.provider.embed_documents(texts)

    async def aembed_query(self, text):
        self.budget.reserve()
        return await self.provider.aembed_query(text)

    async def aembed_documents(self, texts):
        self.budget.reserve()
        return await self.provider.aembed_documents(texts)


class BudgetedGenerator:
    def __init__(self, provider, budget):
        self.provider, self.budget = provider, budget

    async def generate(self, question, hits):
        self.budget.reserve()
        return await self.provider.generate(question, hits)
