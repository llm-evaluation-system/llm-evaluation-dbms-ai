"""
main.py — FastAPI application entry point for the Master LLM Evaluator System.

This module:
  1. Creates the FastAPI application instance with metadata and CORS.
  2. Registers all routers (/eval, /eval/judge, /eval/results, /eval/leaderboard,
     /eval/hyperparams, /eval/prompts, /eval/export).
  3. Adds a latency-tracking middleware per the spec's throughput benchmarks.
  4. Runs database migrations and question bank loading on startup.
  5. Exposes a /health endpoint for orchestration readiness checks.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import APP_DESCRIPTION, APP_TITLE, APP_VERSION, CORS_ORIGINS
from database import create_all_tables
from routers.eval import router as eval_router
from routers.judge import router as judge_router
from routers.results import router as results_router


# ── Startup / Shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    On startup: create all DB tables, load question bank and model registry.
    On shutdown: close the DB engine cleanly.
    """
    # Ensure all ORM tables exist
    await create_all_tables()

    # Load question bank and model registry (idempotent)
    try:
        from question_bank.loader import run_loader
        await run_loader()
    except Exception as exc:
        # Non-fatal on startup — the app still runs; warn in logs
        print(f"[WARN] Question bank loader encountered an issue: {exc}")

    yield

    # Cleanup
    from database import engine
    await engine.dispose()


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=APP_TITLE,
    version=APP_VERSION,
    description=APP_DESCRIPTION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Latency Tracking Middleware ───────────────────────────────────────────────
@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    """
    Middleware that measures total request latency and injects it into the
    response headers as X-Response-Time-Ms.  This satisfies the spec's
    requirement for FastAPI middleware-based latency tracking (Section 2.7).
    """
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
    return response


# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(eval_router)
app.include_router(judge_router)
app.include_router(results_router)


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"], summary="Health and readiness check")
async def health_check():
    """
    Returns the application status, version, and a summary of registered
    components.  Used by container orchestrators for readiness probes.
    """
    return {
        "status": "healthy",
        "version": APP_VERSION,
        "title": APP_TITLE,
        "endpoints": [
            "POST /eval/generate",
            "POST /eval/batch",
            "POST /eval/self-consistency",
            "POST /eval/judge/score",
            "POST /eval/judge/contest",
            "GET  /eval/results/summary",
            "GET  /eval/hyperparams/compare",
            "GET  /eval/prompts/compare",
            "GET  /eval/leaderboard",
            "POST /eval/leaderboard/refresh",
            "GET  /eval/questions",
            "GET  /eval/models",
            "GET  /eval/export/json",
            "GET  /eval/export/csv",
            "POST /eval/seed-examples",
        ],
    }


# ── Global Exception Handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url),
        },
    )
