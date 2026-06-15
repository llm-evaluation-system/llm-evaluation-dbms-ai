# LLM Evaluation for DBMS — Automated Benchmarking Platform

An API-driven evaluation platform that benchmarks Large Language Models on **Database Management Systems (DBMS)** theory and practice. LLM answers are scored by a Judge LLM, validated against a live **PostgreSQL 16** sandbox, and ranked via **Elo-rated pairwise contests**.

> 📂 Full project, setup guide, and detailed docs: [`llm_evaluator_system_finale/`](./llm_evaluator_system_finale/)

---

## 🏆 Benchmark Results

**2,880 judged LLM calls** across 3 phases (zero-shot baseline → 9 prompting strategies → temperature/top-p hyperparameter sweep), over a curated bank of 80+ DBMS questions spanning 20+ subtopics.

| Rank | Model | MCS Score | DB Correctness | Elo | Win Rate | Hallucination Rate |
|------|-------|-----------|----------------|-----|----------|--------------------|
| 🥇 1 | GPT-4o | 70.91 | 77.3 | 1274 | 63.8% | 28.5% |
| 🥈 2 | Llama 3.1 70B | 67.39 | 70.4 | 1126 | 36.3% | 39.3% |

Interesting findings:
- **GPT-4o** performed best with plain **zero-shot** prompting; **Llama 3.1 70B** needed **few-shot chain-of-thought** to reach its best scores.
- Both models struggled most with **concurrency control** and **hash-based indexing**; both excelled at **query evaluation** and **storage/indexing overviews**.
- Llama was ~3.3× faster per call (2.8s vs 9.3s avg latency) but hallucinated more.

Full data: [`final_results/`](./llm_evaluator_system_finale/final_results/) · Run summary: [`SUMMARY.txt`](./llm_evaluator_system_finale/final_results/SUMMARY.txt)

---

## ⚙️ How It Works

```
Question Bank (PostgreSQL)          Phase 1: Generation
  103 curated DBMS questions   ───►   Challenger LLMs answer every question
  from Ramakrishnan & Gehrke          under 9 prompting strategies + HP sweeps
                                              │
                                              ▼
SQL Sandbox (PostgreSQL 16)         Phase 2: Judging
  Executes generated SQL       ◄───   Judge LLM scores answers vs ground truth
  against schema fixtures             + pairwise contests with Elo ratings
                                              │
                                              ▼
                                    Composite Scoring (MCS)
                                      correctness · quality · format
                                      compliance · hallucination ·
                                      robustness → leaderboard
```

## ✨ Key Features

- **FastAPI + async SQLAlchemy** backend with 14-table ORM, Alembic migrations, Celery task queue
- **Multi-provider LLM client layer** (OpenAI, Anthropic, Google, Groq) with retry and latency tracking
- **Automated SQL execution harness** — generated SQL is actually run against a sandboxed PostgreSQL 16 instance
- **Multi-tier hallucination detection**, format compliance checking, and perturbation-based robustness scoring
- **Judge LLM subsystem** with absolute scoring and tournament-style pairwise contests (Elo K=32)
- **9 prompting strategies** compared head-to-head, plus temperature/top-p sensitivity analysis
- **Fully Dockerized**: API + databases + Redis + Celery via one `docker-compose up`

## 🚀 Quick Start

```bash
cd llm_evaluator_system_finale
cp .env.example .env        # add your API keys
docker-compose up -d        # API + Postgres + Redis + Celery
python marathon_runner.py   # run the full benchmark
```

See [`SETUP.md`](./llm_evaluator_system_finale/SETUP.md) for the full guide.

## 📄 Report

Academic report (CS5421): [`Group4_EvaluatingLLM_Capabilities_CS5421.pdf`](./llm_evaluator_system_finale/Group4_EvaluatingLLM_Capabilities_CS5421.pdf)

## 🛠 Tech Stack

`Python` · `FastAPI` · `SQLAlchemy (async)` · `PostgreSQL 16` · `Alembic` · `Celery` · `Redis` · `Docker` · `Pydantic v2` · OpenAI / Anthropic / Google / Groq APIs
