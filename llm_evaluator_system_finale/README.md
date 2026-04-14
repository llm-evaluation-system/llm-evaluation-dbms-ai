# Master LLM Evaluator System

## Overview
This system is a fully automated, API-driven evaluation platform that benchmarks Large Language Models on questions rooted in Database Management Systems (DBMS) theory and practice. Questions are sourced from a curated bank derived from Database Management Systems (Ramakrishnan & Gehrke), covering over 20 subtopics across 5 major DBMS topic areas, benchmarked against PostgreSQL 16.

**The pipeline runs in two phases:**

Phase 1 — Generation: Four challenger LLMs independently answer every question under various prompting strategies and hyperparameter configurations.
Phase 2 — Judging: A Judge LLM performs absolute scoring against ground-truth and pairwise/tournament-style ranking across all four models.

 [Detailed explanation can be found here.](./Group4_EvaluatingLLM_Capabilities_CS5421.pdf)
---

## Architecture

```
llm_evaluator_system_finale/
│
├── main.py                              # FastAPI application entry point
├── config.py                            # All constants, weights, model registry
├── database.py                          # Async SQLAlchemy engine + session factory
├── marathon_runner.py                   # Full benchmark orchestrator (Phase 1 + 2)
├── marathon_runner_with_gemini.py       # Benchmark orchestrator including Gemini
├── full_infra_test.py                   # Infrastructure integration test suite
├── smoke_test.py                        # Quick smoke test for the API
├── requirements.txt                     # Python dependencies
├── Dockerfile                           # Docker image definition
├── docker-compose.yml                   # Full stack: API + DBs + Redis + Celery
├── alembic.ini                          # Alembic migration config
├── .env.example                         # Environment variable template
├── .env.safe                            # Safe/redacted env reference
├── SETUP.md                             # Standalone setup guide
├── README.md                            # This file
├── LLM_Benchmarking_DBMS_Report.docx   # Full benchmarking report
│
├── models/                              # Data layer
│   ├── __init__.py
│   ├── db_models.py                     # SQLAlchemy ORM (14 tables)
│   └── schemas.py                       # Pydantic v2 request/response models
│
├── question_bank/                       # Question ingestion pipeline
│   ├── __init__.py
│   ├── parser.py                        # Excel → JSON converter (one-time)
│   ├── loader.py                        # JSON → PostgreSQL loader (idempotent)
│   └── sql_fixtures.py                  # SQL schema fixtures for the test harness
│
├── llm_clients/                         # LLM provider abstraction
│   ├── __init__.py
│   ├── base_client.py                   # Abstract base with retry + latency tracking
│   └── providers.py                     # OpenAI / Anthropic / Google / Groq clients
│
├── prompting/                           # Prompt strategy engine
│   ├── __init__.py
│   ├── templates.py                     # All 9 prompting strategy builders
│   └── few_shot_store.py                # Pre-computed example bank + leakage guard
│
├── judge/                               # Judge LLM subsystem
│   ├── __init__.py
│   ├── judge_llm.py                     # Absolute scoring + pairwise contest protocols
│   └── elo.py                           # Elo rating system (K=32, starting Elo 1200)
│
├── evaluators/                          # Evaluation logic
│   ├── __init__.py
│   ├── eval_service.py                  # Core orchestration: generation + scoring
│   ├── sql_harness.py                   # Automated SQL execution (PostgreSQL 16 sandbox)
│   ├── hallucination.py                 # Multi-tier hallucination detection pipeline
│   ├── format_compliance.py             # Automated format compliance checker
│   └── robustness.py                    # Perturbation generator + consistency scorer
│
├── scoring/                             # Composite scoring engine
│   ├── __init__.py
│   └── composite.py                     # MCS formula + pillar/sub-score engines
│
├── routers/                             # FastAPI route handlers
│   ├── __init__.py
│   ├── eval.py                          # /eval/generate, /eval/questions, /eval/models
│   ├── judge.py                         # /eval/judge/score, /eval/judge/contest
│   └── results.py                       # /eval/results, /eval/leaderboard, /eval/export
│
├── tasks/                               # Async task queue
│   └── __init__.py                      # Celery task definitions
│
├── alembic/                             # Database migrations
│   ├── __init__.py
│   └── env.py                           # Alembic migration environment
│
├── scripts/                             # Utility scripts
│   └── init_test_db.py                  # Seeds the sandboxed test PostgreSQL instance
│
├── data/                                # Static data assets
│   ├── question_bank.json               # Pre-parsed question bank (103 questions)
│   ├── er_diagram_questions.json        # ER diagram specific questions
│   └── artifacts/                       # Additional data artifacts
│
├── final_results/                       # Pre-computed results (Phase 1 + 2, 3 models)
│   ├── SUMMARY.txt                      # Human-readable run summary
│   ├── leaderboard.json                 # Model rankings: MCS, Elo, win rate
│   ├── results_summary.json             # Per-model × per-subtopic breakdown
│   ├── full_export.json                 # All scored runs (complete export)
│   ├── audit.json                       # Audit log for the run
│   ├── run_log.txt                      # Detailed run log
│   ├── phase1_generate.json             # Phase 1 raw generation outputs
│   ├── phase1_contests.json             # Phase 1 contest results
│   ├── phase1_scores.json               # Phase 1 scores
│   ├── phase2_generate.json             # Phase 2 raw generation outputs
│   ├── phase2_scores.json               # Phase 2 scores
│   ├── phase3_hyperparam_generate.json  # Phase 3 hyperparameter sweep outputs
│   ├── phase3_scores.json               # Phase 3 scores
│   ├── hyperparams_gpt4o.json           # GPT-4o hyperparameter sensitivity
│   ├── hyperparams_llama.json           # Llama hyperparameter sensitivity
│   ├── prompts_compare_gpt4o.json       # GPT-4o prompting strategy scorecard
│   └── prompts_compare_llama.json       # Llama prompting strategy scorecard
│
├── final_results_with_all_models/       # Pre-computed results (all 4 models incl. Gemini)
│   ├── SUMMARY.txt
│   ├── leaderboard.json
│   ├── results_summary.json
│   ├── full_export.json
│   ├── audit.json
│   ├── run_log.txt
│   ├── phase1_generate.json
│   ├── phase1_contests.json
│   ├── phase1_scores.json
│   ├── phase2_generate.json
│   ├── phase2_scores.json
│   ├── phase3_hyperparam_generate.json
│   ├── phase3_scores.json
│   ├── hyperparams_gpt4o.json
│   ├── hyperparams_llama.json
│   ├── hyperparams_gemini.json          # Gemini hyperparameter sensitivity
│   ├── prompts_compare_gpt4o.json
│   ├── prompts_compare_llama.json
│   └── prompts_compare_gemini.json      # Gemini prompting strategy scorecard
│
├── mcs_updated_results/                 # MCS-recalculated result snapshots
│
└── fixes/                               # Patched output files
    └── mnt/user-data/outputs/
        ├── leaderboard.json
        ├── results_summary.json
        ├── full_export.json
        ├── prompts_compare_gpt4o.json
        └── prompts_compare_llama.json
```

---

## Prerequisites

- Python 3.12+
- Docker + Docker Compose (for the full stack)
- PostgreSQL 16 (two instances: primary + sandboxed test)
- Redis 7 (Celery broker)
- API keys for: OpenAI and Groq (LLaMa)

---

## Quick Start

**Step 1 — Clone and configure environment variables:**

```bash
cp .env.example .env
# Edit .env and fill in your LLM API keys
```

**Step 2 — Start the full stack with Docker Compose inside ./llm_evaluator_system_finale:**

```bash
docker compose down -v
docker system prune -a --volumes  
docker compose build --no-cache   
docker compose -f docker-compose.yml up -d
docker compose exec app python scripts/init_test_db.py
```

**Step 3 - Open a new terminal to login to the DB and check for tables. Use this when you want to watch the data population upon calling APIs.**

```bash
psql -h localhost -p 5432 -U postgres -d llm_evaluator
\x auto
```

The Steps 2 and 3 bring up the FastAPI app on port 8000, the two PostgreSQL instances, Redis, and the Celery worker.

**Step 3 — On first run, the application automatically:**
- Creates all 14 database tables.
- Parses the Excel question bank into structured JSON (103 questions).
- Loads all topics, subtopics, questions, and model registry into PostgreSQL.
- Seeds the few-shot example store (ONLY WHEN `marathon_runner.py` is executed, else you need to trigger the API yourself).

**Step 4 — Verify the system is running:**

```bash
curl http://localhost:8000/health
```

**Step 5 — Access the interactive API documentation:**

Open `http://localhost:8000/docs` in your browser. This opens a playground for you to test various APIs.

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

### Phase 2 — Judging

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/eval/judge/score` | Judge scores one model's answer (0–10 + full breakdown) |
| `POST` | `/eval/judge/contest` | Judge ranks all four models for one question + updates Elo |

**Example — Run a pairwise contest:**

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

# `marathon_runner.py` is an automation script that executes the complete non-visual ER flow against all questions, all hyperparameters, etc. BUT, YOU NEED TO ENSURE THE API BILLING IS SET AS IT IS AN EXPENSIVE OPERATION.

## Challenger Models

| Model ID | Provider | Role |
|---|---|---|
| `gpt-4o` | OpenAI | (Challenger + Judge in Non-ER) + (Judge in ER)|
| `Claude Sonnet 4.6` | Anthropic | Challenger for ER Generation |
| `Gemini 3 Flash` | Google | Challenger for ER Generation |
| `llama-3.3-70b` | Groq | Challenger for Non-ER and ER Interpretation |
