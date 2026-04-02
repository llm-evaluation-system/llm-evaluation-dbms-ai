# Master LLM Evaluator System

**Comprehensive Evaluation Benchmarking Framework for Database Management Systems Question Answering**

---

## Overview

This system is a fully automated, API-driven evaluation platform that assesses the capability of four Large Language Models in answering questions rooted in Database Management Systems theory and practice. All questions are sourced from a curated bank derived from the standard *Database Management Systems* textbook (Ramakrishnan & Gehrke), covering **21 subtopics** across **5 major DBMS topic areas**, benchmarked against PostgreSQL 16 as the reference SQL dialect.

The evaluation pipeline follows a two-phase architecture as specified in the design document:

**Phase 1 — Generation:** Each of the four challenger LLMs independently answers every question in the question bank under various prompting strategies and hyperparameter configurations.

**Phase 2 — Judging:** The fifth, most powerful Judge LLM performs absolute scoring of each model's answer against the ground-truth, and pairwise/tournament-style contest ranking across all four models simultaneously.

---

## Architecture

```
llm_evaluator/
├── main.py                        # FastAPI application entry point
├── config.py                      # All constants, weights, model registry
├── database.py                    # Async SQLAlchemy engine + session factory
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── models/
│   ├── db_models.py               # SQLAlchemy ORM (14 tables)
│   └── schemas.py                 # Pydantic v2 request/response models
│
├── question_bank/
│   ├── parser.py                  # One-time Excel → JSON converter
│   └── loader.py                  # JSON → PostgreSQL loader (idempotent)
│
├── llm_clients/
│   ├── base_client.py             # Abstract base with retry + latency tracking
│   └── providers.py               # OpenAI / Anthropic / Google / Groq clients
│
├── prompting/
│   ├── templates.py               # All 9 prompting strategy builders
│   └── few_shot_store.py          # Pre-computed example bank + leakage guard
│
├── judge/
│   ├── judge_llm.py               # Absolute scoring + pairwise contest protocols
│   └── elo.py                     # Elo rating system (K=32, starts at 1200)
│
├── evaluators/
│   ├── eval_service.py            # Core orchestration: generation + scoring
│   ├── sql_harness.py             # Automated SQL execution harness (PostgreSQL 16)
│   ├── hallucination.py           # Multi-tier hallucination detection pipeline
│   ├── format_compliance.py       # Automated format compliance checker
│   └── robustness.py              # Perturbation generator + consistency scorer
│
├── scoring/
│   └── composite.py               # MCS formula + all pillar/sub-score engines
│
├── routers/
│   ├── eval.py                    # /eval/generate, /eval/questions, /eval/models
│   ├── judge.py                   # /eval/judge/score, /eval/judge/contest
│   └── results.py                 # /eval/results, /eval/leaderboard, /eval/export
│
├── tasks/
│   └── __init__.py                # Celery async task definitions
│
├── alembic/
│   └── env.py                     # Alembic migration environment
│
└── data/
    └── question_bank.json         # Pre-parsed question bank (auto-generated)
```

---

## Prerequisites

- Python 3.12+
- Docker + Docker Compose (for the full stack)
- PostgreSQL 16 (two instances: primary + sandboxed test)
- Redis 7 (Celery broker)
- API keys for: OpenAI, Anthropic, Google (Gemini), Groq

---

## Quick Start

**Step 1 — Clone and configure environment variables:**

```bash
cp .env.example .env
# Edit .env and fill in your four API keys
```

**Step 2 — Start the full stack with Docker Compose:**

```bash
docker compose up --build -d
```

This brings up the FastAPI app on port 8000, the two PostgreSQL instances, Redis, and the Celery worker.

**Step 3 — On first run, the application automatically:**
- Creates all 14 database tables.
- Parses the Excel question bank into structured JSON (103 questions).
- Loads all topics, subtopics, questions, and model registry into PostgreSQL.
- Seeds the few-shot example store.

**Step 4 — Verify the system is running:**

```bash
curl http://localhost:8000/health
```

**Step 5 — Access the interactive API documentation:**

Open `http://localhost:8000/docs` in your browser.

---

## Running Without Docker

```bash
# Install dependencies
pip install -r requirements.txt

# Start PostgreSQL and Redis (adjust connection strings in .env as needed)

# Parse and load the question bank
python question_bank/parser.py
python question_bank/loader.py

# Start the application
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Reference

All endpoints are fully documented in the interactive OpenAPI spec at `/docs`.

### Phase 1 — Generation

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/eval/generate` | Submit a question to a challenger model |
| `GET` | `/eval/questions` | List question bank (filterable by topic, type, difficulty) |
| `GET` | `/eval/models` | List registered models |
| `POST` | `/eval/seed-examples` | Seed the few-shot example store |

**Example — Generate an answer:**

```bash
curl -X POST http://localhost:8000/eval/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "gpt-4o",
    "question_id": "<question-uuid>",
    "prompt_strategy": "few-shot-cot",
    "hyperparams": {
      "temperature": 0.3,
      "top_p": 0.9,
      "max_tokens": 1024,
      "system_prompt_style": "expert-persona"
    }
  }'
```

### Phase 2 — Judging

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/eval/judge/score` | Judge scores one model's answer (0–10 + full breakdown) |
| `POST` | `/eval/judge/contest` | Judge ranks all four models for one question + updates Elo |

**Example — Run a pairwise contest:**

```bash
curl -X POST http://localhost:8000/eval/judge/contest \
  -H "Content-Type: application/json" \
  -d '{
    "question_id": "<question-uuid>",
    "run_ids": ["<run-1>", "<run-2>", "<run-3>", "<run-4>"]
  }'
```

### Analytics & Reporting

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/eval/results/summary` | Aggregate scores by model/topic/strategy |
| `GET` | `/eval/hyperparams/compare` | Cross-model hyperparameter sensitivity analysis |
| `GET` | `/eval/prompts/compare` | Prompting strategy scorecard per model |
| `GET` | `/eval/leaderboard` | Global leaderboard (MCS + Elo + win rate) |
| `GET` | `/eval/export/json` | Full results export as JSON |
| `GET` | `/eval/export/csv` | Full results export as CSV |

---

## Evaluation Dimensions

### Section 1 — Database Correctness (40% of MCS)

The DB Correctness pillar aggregates five cluster scores, each routing to the appropriate evaluator based on question type:

**SQL Syntactic & Semantic Correctness** runs the generated SQL through a sandboxed PostgreSQL 16 instance using `EXPLAIN` for syntactic validation and direct execution against schema fixtures for result set F1 scoring.

**Conceptual Accuracy** uses the Judge LLM to evaluate theory questions (relational algebra, normalization, indexing, transactions, crash recovery) against textbook ground truth on four axes: factual correctness, completeness, absence of contradiction, and topic specificity.

**Schema Design Quality** checks entity coverage, foreign key correctness, normalization compliance, and index appropriateness via automated `information_schema` queries plus judge assessment.

**Query Plan & Optimization Awareness** evaluates join algorithm selection, index selectivity reasoning, cost estimation accuracy, and plan tree correctness.

**Transaction & Concurrency** applies a 60/40 auto/judge split — deterministic serializability and ARIES trace questions are auto-scored; explanation questions go to the judge.

### Section 2 — LLM Quality (30% of MCS)

The LLM Quality pillar measures cross-cutting model behaviour: hallucination rate (with severity weighting), chain-of-thought coherence, text-level precision/recall/F1, format compliance, and cross-perturbation consistency.

The hallucination detection pipeline operates in three tiers: regex + PostgreSQL catalog lookup for fabricated SQL functions, pattern matching against a registry of known wrong DBMS facts, and LLM judge review for semantic contradictions.

### Section 3 — Prompting Effectiveness (15% of MCS)

Nine strategies are benchmarked: zero-shot, one-shot, few-shot (3-shot), chain-of-thought, few-shot+CoT, self-consistency (k=5), role prompting, least-to-most, and ReAct. Accuracy lift over the zero-shot baseline is the primary metric, supplemented by token efficiency and consistency variance.

### Section 4 — Efficiency (15% of MCS)

Latency (time-to-first-token and total generation time), throughput (tokens/second), API retry rate, and cost-per-correct-answer are tracked for every run via FastAPI middleware and stored per evaluation run.

---

## Composite Scoring Formula

```
MCS = 0.50 × DB_Correctness + 0.30 × LLM_Quality
    + 0.15 × Prompting_Effectiveness + 0.05 × Efficiency
```

All pillars are normalised to a 0–100 scale. The MCS is decomposable — any downstream analytics query can drill down from the MCS into any sub-dimension.

---

## Elo Rating System

Contest results feed a dynamic Elo system. Each question contest is treated as a four-player match. Expected scores use the standard Elo formula with K=32. All models start at Elo 1200. Ratings and history are persisted in PostgreSQL and surfaced on the `/eval/leaderboard` endpoint.

---

## Hyperparameter Sweep

The system supports systematic single-axis hyperparameter sweeps via the Celery task queue. To trigger a sweep:

```bash
# Via the Celery task directly
celery -A tasks call tasks.hyperparam_sweep_task \
  --args='["gpt-4o", "temperature"]'
```

Results are queryable via `GET /eval/hyperparams/compare?model_id=gpt-4o&param_name=temperature`.

---

## Question Bank Summary

The question bank contains **103 questions** parsed from the provided Excel file, covering:

| Topic Area | Subtopics | Questions |
|---|---|---|
| Foundations | Introduction to DB Design, Relational Model, Relational Algebra, SQL | 27 |
| Application Development | DB App Development, Internet Applications | 4 |
| Storage & Indexing | Overview, Disks & Files, Tree Indexing, Hash Indexing | 12 |
| Query Evaluation | Overview, External Sorting, Relational Operators, Query Optimizer | 10 |
| Transaction Management | Overview, Concurrency Control, Crash Recovery | 31 |
| Database Design & Tuning | Schema Refinement, Physical Design, Security | 16 |
| Additional Topics | Data Warehousing & Decision Support | 3 |

---

## Idempotency Guarantee

Every generation run is identified by the composite key `(question_id, model_id, prompt_strategy, hyperparam_hash)`. Re-running the same configuration overwrites the existing record and does not create duplicates. This makes the system fully replay-safe for batch operations.

---

## Challenger Models

| Model ID | Provider | Role |
|---|---|---|
| `gpt-4o` | OpenAI | Challenger + Judge |
| `claude-3-5-sonnet` | Anthropic | Challenger |
| `gemini-1.5-pro` | Google | Challenger |
| `llama-3.1-70b` | Groq | Challenger |
