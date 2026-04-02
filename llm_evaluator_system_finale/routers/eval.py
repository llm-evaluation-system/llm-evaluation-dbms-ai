"""
routers/eval.py — FastAPI router for evaluation generation endpoints.

Implements:
  POST /eval/generate          — submit a question to a challenger model
  POST /eval/batch             — batch evaluation across models/questions (Celery)
  POST /eval/self-consistency  — sample k=5 responses and aggregate
  GET  /eval/questions         — list questions from the question bank
  GET  /eval/models            — list registered models
  POST /eval/seed-examples     — seed the few-shot example store
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from evaluators.eval_service import run_generation, run_self_consistency
from models.db_models import LLMModel, Question, Subtopic, Topic
from models.schemas import (
    BatchEvalRequest,
    BatchEvalResponse,
    GenerateRequest,
    GenerateResponse,
    ModelSchema,
    QuestionListResponse,
    QuestionSchema,
)
from prompting.few_shot_store import seed_few_shot_examples

router = APIRouter(prefix="/eval", tags=["Evaluation"])


@router.post("/generate", response_model=GenerateResponse, summary="Generate answer for one question")
async def generate_answer(
    body: GenerateRequest,
    db: AsyncSession = Depends(get_db),
) -> GenerateResponse:
    """
    Submit a question to a challenger LLM under a specified prompting strategy
    and hyperparameter configuration.  If the same (question, model, strategy,
    hyperparams) combination was already evaluated, returns the cached run
    unless force_rerun is set.

    When async_run=True the job is dispatched to the Celery task queue and
    the response contains a task_id instead of a completed answer.

    When prompt_strategy='self-consistency', k=5 samples are drawn at
    temperature=0.7 and aggregated via plurality vote (Section 4.2.4).
    """
    # ── Async dispatch (Section 3.2 / FastAPI Implementation Checklist) ──
    if body.async_run:
        try:
            from tasks import generate_answer_task
            task = generate_answer_task.delay(
                model_id=body.model_id,
                question_id=body.question_id,
                prompt_strategy=body.prompt_strategy,
                hyperparams=body.hyperparams.model_dump(),
            )
            return GenerateResponse(
                run_id="pending",
                model_id=body.model_id,
                question_id=body.question_id,
                prompt_strategy=body.prompt_strategy,
                status="queued",
                task_id=task.id,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Celery dispatch failed: {exc}")

    # ── Self-consistency (Section 4.2.4) ─────────────────────────────────────
    if body.prompt_strategy == "self-consistency":
        try:
            sc_result = await run_self_consistency(
                session=db,
                model_id=body.model_id,
                question_id=body.question_id,
                base_hyperparams=body.hyperparams.model_dump(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc))

        # Return the consensus run as the primary response
        first_run_id = sc_result["run_ids"][0] if sc_result["run_ids"] else "unknown"
        return GenerateResponse(
            run_id=first_run_id,
            model_id=body.model_id,
            question_id=body.question_id,
            prompt_strategy="self-consistency",
            status="completed",
            model_answer=sc_result["best_answer"],
        )

    # ── Synchronous single generation ─────────────────────────────────────────
    try:
        run = await run_generation(
            session=db,
            model_id=body.model_id,
            question_id=body.question_id,
            prompt_strategy=body.prompt_strategy,
            hyperparams=body.hyperparams.model_dump(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # Look up display model_id from DB model UUID
    model_result = await db.execute(
        select(LLMModel).where(LLMModel.id == run.model_id)
    )
    db_model = model_result.scalar_one_or_none()
    logical_model_id = db_model.model_id if db_model else body.model_id

    return GenerateResponse(
        run_id=run.id,
        model_id=logical_model_id,
        question_id=run.question_id,
        prompt_strategy=run.prompt_strategy,
        status=run.status,
        model_answer=run.model_answer,
        input_tokens=run.input_tokens,
        output_tokens=run.output_tokens,
        cost_usd=run.cost_usd,
        total_latency_ms=run.total_latency_ms,
    )


@router.post(
    "/batch",
    response_model=BatchEvalResponse,
    summary="Dispatch batch evaluation jobs via Celery",
)
async def batch_evaluate(
    body: BatchEvalRequest,
    db: AsyncSession = Depends(get_db),
) -> BatchEvalResponse:
    """
    Dispatch a batch evaluation across all combinations of
    models × questions × strategies using the Celery task queue.

    If question_ids is omitted the full question bank is used.
    Returns a batch_id and the list of dispatched Celery task IDs so the
    caller can poll /eval/results/summary for progress.

    Implements Section 3.2 of the spec: the /eval/hyperparams/compare endpoint
    accepts a param_grid and dispatches sweep jobs to a Celery task queue.
    """
    try:
        from tasks import batch_eval_task
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Celery task queue unavailable: {exc}",
        )

    # Resolve question IDs if not provided
    question_ids = body.question_ids
    if not question_ids:
        result = await db.execute(select(Question.id))
        question_ids = [str(r[0]) for r in result.all()]

    strategies = body.prompt_strategies or ["zero-shot", "few-shot-cot"]

    try:
        task = batch_eval_task.delay(
            model_ids=body.model_ids,
            question_ids=question_ids,
            prompt_strategies=strategies,
            hyperparams=body.hyperparams.model_dump(),
            run_judge=body.run_judge,
            run_contest_after=body.run_contest,
        )
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Celery dispatch failed: {exc}")

    total_jobs = len(body.model_ids) * len(question_ids) * len(strategies)
    batch_id = str(uuid.uuid4())

    return BatchEvalResponse(
        batch_id=batch_id,
        total_jobs=total_jobs,
        task_ids=[task.id],
        estimated_completion_seconds=total_jobs * 5.0,
    )


@router.post(
    "/self-consistency",
    summary="Run self-consistency sampling (k=5) for one model-question pair",
)
async def self_consistency_endpoint(
    model_id: str,
    question_id: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Sample k=5 responses at temperature=0.7 and aggregate via plurality vote
    (SQL questions) or Judge LLM synthesis (conceptual questions).

    Section 4.2.4: Self-consistency costs 5× the API calls.  Limit to Hard
    difficulty questions or those showing historically high variance.

    Returns the best aggregated answer, the individual run IDs, and k statistics.
    """
    try:
        result = await run_self_consistency(
            session=db,
            model_id=model_id,
            question_id=question_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return result
@router.get("/questions", response_model=QuestionListResponse, summary="List questions from question bank")
async def list_questions(
    topic: Optional[str] = Query(None, description="Filter by topic name"),
    subtopic: Optional[str] = Query(None, description="Filter by subtopic name"),
    question_type: Optional[str] = Query(None, description="sql|conceptual|schema|…"),
    difficulty: Optional[str] = Query(None, description="easy|medium|hard"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> QuestionListResponse:
    """Return a paginated list of questions from the question bank."""
    stmt = (
        select(Question, Subtopic, Topic)
        .join(Subtopic, Question.subtopic_id == Subtopic.id)
        .join(Topic, Subtopic.topic_id == Topic.id)
    )
    if topic:
        stmt = stmt.where(Topic.name.ilike(f"%{topic}%"))
    if subtopic:
        stmt = stmt.where(Subtopic.name.ilike(f"%{subtopic}%"))
    if question_type:
        stmt = stmt.where(Question.question_type == question_type)
    if difficulty:
        stmt = stmt.where(Question.difficulty == difficulty)

    count_stmt = select(func.count()).select_from(stmt.subquery())
    total_result = await db.execute(count_stmt)
    total = total_result.scalar() or 0

    stmt = stmt.offset(offset).limit(limit)
    result = await db.execute(stmt)
    rows = result.all()

    questions = [
        QuestionSchema(
            id=q.id,
            subtopic=sub.name,
            topic=top.name,
            exercise_number=q.exercise_number,
            question_text=q.question_text,
            expected_answer=q.expected_answer,
            question_type=q.question_type,
            difficulty=q.difficulty,
            tags=q.tags,
        )
        for q, sub, top in rows
    ]
    return QuestionListResponse(total=total, questions=questions)


@router.get("/models", response_model=list[ModelSchema], summary="List registered models")
async def list_models(
    db: AsyncSession = Depends(get_db),
) -> list[ModelSchema]:
    """Return all LLM models registered in the system (challengers + judge)."""
    result = await db.execute(select(LLMModel))
    models = result.scalars().all()
    return [
        ModelSchema(
            model_id=m.model_id,
            display_name=m.display_name,
            provider=m.provider,
            api_model=m.api_model,
            is_judge=m.is_judge,
            max_context_tokens=m.max_context_tokens,
            supports_seed=m.supports_seed,
        )
        for m in models
    ]


@router.post("/seed-examples", summary="Seed the few-shot example store")
async def seed_examples(
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Build and persist the pre-computed few-shot example store from the
    existing question bank.  Idempotent — safe to call multiple times.
    """
    inserted = await seed_few_shot_examples(db)
    return {"status": "ok", "examples_inserted": inserted}
