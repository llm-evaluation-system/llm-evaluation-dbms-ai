# LLM Evaluator — Standalone Setup Guide

## Prerequisites
- Docker + Docker Compose
- Python 3.12+ (only needed to run marathon_runner.py outside Docker)
- API keys: OpenAI (GPT-4o judge + challenger), Groq (Llama 3.1 70B)

## Step 1 — Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys:
#   OPENAI_API_KEY=sk-...
#   GROQ_API_KEY=gsk_...
# (Anthropic and Google keys only needed if adding those models)
```

## Step 2 — Start the stack

```bash
docker compose up --build -d
# Starts: FastAPI (port 8000), PostgreSQL (5432), test-DB (5433), Redis (6379), Celery worker
```

Wait ~30 seconds for all services to be healthy:
```bash
docker compose ps   # all should show "healthy" or "running"
```

## Step 3 — Initialize the database

```bash
# Load schema + question bank into PostgreSQL
docker compose exec app python question_bank/loader.py

# Seed the test database fixtures (SQL harness)
docker compose exec app python scripts/init_test_db.py
```

## Step 4 — Run the full benchmark

```bash
# From your local machine (not inside Docker):
python marathon_runner.py
# Runs ~80 questions × 2 models × 8 hyperparam configs = 1,120 evaluations
# Followed by pairwise contest sweep + report export
# Outputs: final_results/{leaderboard,results_summary,full_export,hyperparams_*}.json
```

Or run inside Docker:
```bash
docker compose exec app python marathon_runner.py
```

## Verify the API is up

```bash
curl http://localhost:8000/health
curl http://localhost:8000/eval/leaderboard
```

## What's in final_results/

These are the pre-computed results from a completed run:
- `full_export.json`     — all 1,128 scored runs (0 null scores after fix)
- `leaderboard.json`     — model rankings with MCS, DB correctness, LLM quality
- `results_summary.json` — per-model × per-subtopic breakdown
- `hyperparams_llama.json` / `hyperparams_gpt4o.json` — temperature & top_p sensitivity

## Fixes applied in this version

1. **`data/question_bank.json`** — Deduplicated from 84→80 entries (4 questions were
   duplicated under both SCHEMA REFINEMENT and SECURITY AND AUTHORIZATION subtopics).

2. **`evaluators/eval_service.py`** — Judge fallback scorer: if the Judge API call
   fails (timeout on very long expected answers), a conservative automated score is
   computed instead of leaving null values.

3. **`marathon_runner.py`** — (a) hyperparams/compare now passes required `param_name`
   query parameter; (b) contest phase runs as a dedicated Phase 2 after all generation
   is complete; (c) leaderboard refresh called after contests.

4. **`tasks/__init__.py`** — contest_task now has a readiness guard that waits until
   both models have scored answers before firing the Judge contest (max 12 retries ×
   30s = 6 min window). Previously it fired immediately and always failed.
