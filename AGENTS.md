# AGENTS.md

## Project

Project name: Hybrid Recommendation and Ranking Service.

This is a production-like ML system for content/movie recommendations with a two-stage architecture:

1. Candidate generation / retrieval.
2. Ranking.

The first dataset target is MovieLens. The advanced migration target is MIND.

This is not a notebook-only project. The system must be implemented as modular Python code with tests, configs, Docker, PostgreSQL, FastAPI, and reproducible pipelines.

## Operating Mode

Work phase by phase.

Do not ask the user for confirmation for every small step inside the active phase. If the task is inside the current phase scope, proceed autonomously.

You may autonomously:
- create new project files required by the active phase;
- edit existing source files required by the active phase;
- add tests for modified code;
- run tests;
- fix failing tests;
- run local validation commands;
- update README for the active phase;
- create configs required by the active phase.

You must ask for confirmation before:
- deleting files or directories;
- modifying files under data/raw/;
- modifying .env with real secrets;
- running destructive shell commands;
- running commands outside the repository root;
- changing the database destructively using DROP, TRUNCATE, DELETE, or destructive migrations;
- pushing to remote Git;
- changing the project architecture beyond the active phase;
- adding new major dependencies not required by the phase;
- downloading large external datasets unless the active phase explicitly asks for data acquisition.

## Phase Discipline

Never implement multiple phases at once unless explicitly requested.

Before starting a phase:
1. Read README.md.
2. Read configs/.
3. Read src/ structure.
4. Read tests/.
5. Identify current phase scope.
6. Produce a short implementation plan.
7. Then implement without repeatedly asking for confirmation.

At the end of each phase, provide:
- files changed;
- commands run;
- test results;
- known limitations;
- whether acceptance criteria are satisfied.

## Architecture Rules

Use this repository structure:

- src/api/ for FastAPI routes and schemas.
- src/core/ for settings, logging, random seed helpers.
- src/data/ for raw loading, validation, preparation, temporal split.
- src/features/ for user, item, and pair features.
- src/retrieval/ for candidate generation.
- src/ranking/ for ranking dataset and ranker models.
- src/evaluation/ for top-K metrics and offline evaluation.
- src/db/ for SQLAlchemy models and sessions.
- src/jobs/ for batch jobs.
- configs/ for YAML configs.
- tests/ for pytest tests.
- artifacts/ for models, metrics, reports, and candidate caches.
- data/raw/ for external raw datasets, never committed.
- data/processed/ for generated parquet outputs, never committed.

Do not put business logic into notebooks.

Notebooks are allowed only for EDA and exploratory analysis.

## ML Rules

Use temporal split only. Do not use random split for recommendation evaluation.

For MovieLens:
- Convert ratings to implicit feedback.
- Use rating >= 4.0 as positive feedback unless the config says otherwise.
- Use only train history to build features for validation/test.
- Do not leak future interactions into user/item features.
- Exclude already-seen train items from recommendations.
- Implement cold-start fallback.

The system must separate:
- retrieval candidate generation;
- ranking;
- offline evaluation;
- API serving.

## Testing Rules

Every phase must add or update tests.

Use pytest.

Unit tests must use small synthetic data created with tmp_path.

Do not require the full MovieLens dataset for normal unit tests.

Run:

```bash
pytest -q
```

before considering a phase complete.

For commands that depend on real data, add a separate smoke command and document it in README.

Code Style

Use Python 3.11.

Use type hints.

Prefer small functions with explicit inputs and outputs.

Avoid hidden global state.

Avoid one giant script.

Use YAML configs instead of hardcoded paths and thresholds.

Use pathlib for file paths.

Use pandas for initial data pipelines.

Use SQLAlchemy for database access.

Use Pydantic for API schemas.

Data Rules

Do not commit raw datasets.

Do not commit generated parquet files unless explicitly requested.

Do not modify files under data/raw/ except through an explicit data acquisition/preparation command.

All raw data assumptions must be encoded in configs and validation code.

Database Rules

PostgreSQL is used for:

users;
items;
interactions;
candidate cache;
ranked recommendations;
recommendation requests;
feedback logs;
model versions.

Do not run destructive DB operations without explicit user confirmation.

README Rules

After every phase, update README with:

current phase status;
commands added;
expected outputs;
test instructions;
known limitations.
Completion Definition

A phase is complete only when:

all required files exist;
acceptance criteria are satisfied;
pytest passes;
relevant CLI commands run successfully;
README is updated;
no unrelated scope was implemented.
