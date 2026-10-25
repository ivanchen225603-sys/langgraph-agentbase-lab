# Repository Guidelines

## Project Structure & Module Organization

- `agentbase/`: reusable Agent runtime, tool authorization, persistence, HTTP/SSE, evaluation, and starter templates.
- `raglab/`: LangChain/LangGraph RAG application; browser assets live in `raglab/web/`.
- `examples/`: standalone Agent integrations and configuration examples.
- `scripts/`: indexing, backups, demos, and verification.
- `data/knowledge/`: synthetic Markdown knowledge fixtures.
- Selected integration tests in `tests/` and guides in `docs/` are versioned explicitly; private development materials remain local. Generated state and reports belong in `.runtime*/` and `evidence/`.

## Build, Test, and Development Commands

Use Python 3.11–3.13; prefer 3.11. Activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.lock
python -m pip install --no-deps --no-build-isolation -e .
```

For Git checkouts without the local development lock, use `python -m pip install -e '.[dev]'`. Minimal source distributions use `requirements.lock`.

- `python -m raglab`: serve at `http://127.0.0.1:8001`; build the example index through the UI before querying.
- `python -m examples.agentbase_minimal`: exercise standalone tools and idempotent execution offline.
- `python -m scripts.demo`: run RAG, memory, and task-recovery demonstrations.
- `python -m pip check`: check installed dependency compatibility.
- `python -m build`: produce wheel and source distributions.
- `python -m scripts.check_client`: verify SSE and both UI test suites with Node.js 22+.
- `python -m scripts.verify`: optional full local pipeline; requires separately available development materials.

## Coding Style & Naming Conventions

Use four-space Python indentation, `snake_case` functions/modules, and `PascalCase` classes. Follow existing typing and async patterns. Ruff targets Python 3.11 with a 100-character width; use `python -m ruff check .` and `python -m ruff format --check .`. Run `python -m mypy`. Keep execution centralized in `RunEngine`.

## Testing Guidelines

Use pytest and pytest-asyncio; name files `tests/test_*.py` and functions `test_*`. Run `python -m pytest --cov=agentbase --cov=raglab --cov-fail-under=80`. Full verification additionally requires at least 80% coverage for `raglab` and nonzero coverage for `agentbase`. Use temporary directories, synthetic data, and deterministic providers. Cover authorization, replay, cancellation, and recovery when changing runtime behavior.

## Commit & Pull Request Guidelines

Commits use `feat:`, `fix:`, and `chore:` prefixes with imperative summaries. PRs should describe behavior, link relevant issues, report verification and limitations, and include screenshots for UI changes. Preserve unrelated edits.

## Configuration & Agent Instructions

Start from `.env.example`; keep secrets, databases, and generated evidence out of commits. Default to offline local mode. Use one service process per data directory. Follow the explicit integration allowlist in `.gitignore`; review publication scope before staging exceptions.

Run `$simplify` only when explicitly requested, limited to task changes; never apply it automatically.
