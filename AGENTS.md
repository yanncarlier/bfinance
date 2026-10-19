# AGENTS.md

Python project. Read this before making changes. If anything below disagrees
with the code or `pyproject.toml`, trust the code and tell me.

## Setup

- Python version: see `requires-python` in `pyproject.toml`.
- Package manager: `uv`. Always run tools through it, never bare `python` or `pip`.

```bash
uv sync                 # install project + dev dependencies
```

## Commands

```bash
uv run pytest                              # all tests
uv run pytest tests/test_x.py::test_name   # one test
uv run pytest -x -q -k "keyword"           # stop at first failure, filter by name
uv run ruff check . --fix                  # lint
uv run ruff format .                       # format
uv run mypy src/                           # type check (or pyright, if configured)
```

Before saying a task is done, run tests, lint, format check and type check,
and report any that fail. Do not claim success without running them.

## Layout

- `src/<package>/` application code
- `tests/` mirrors `src/`; one test file per module
- `pyproject.toml` is the only place for tool config and dependencies

## Conventions

- `ruff` and the type checker are the source of truth for style. Do not
  hand-format or argue with them.
- Type-hint all public functions. Scope any `# type: ignore` narrowly and
  explain why.
- Raise specific exceptions. No bare `except:`. Never swallow errors silently.
- Use `pathlib`, not `os.path`. Use `subprocess.run([...])` with a list,
  never `shell=True`.
- Use dataclasses or pydantic models for structured data crossing module
  boundaries, not loose dicts.
- Keep pure logic separate from I/O so it can be unit tested.

## Testing

- Add or update tests with every behavior change. Cover failure and edge cases.
- Never delete, skip or weaken a test to make it pass. If a test looks wrong,
  say so and ask.
- Mock external services in unit tests. No real network calls.

## Boundaries

- Ask before adding a dependency, and say why it is needed.
- Ask before changing public APIs, config schemas, or CI files.
- Never commit secrets or `.env` files. Read config from environment variables.
- Keep changes small and focused. Do not refactor unrelated code in the same change.
- Never run destructive commands (`rm -rf`, `git reset --hard`, force push)
  without asking.

## Project-specific notes

<!-- Fill in what an agent could not guess: how to run the app locally,
required env vars, how to start a database, known flaky tests, gotchas. -->