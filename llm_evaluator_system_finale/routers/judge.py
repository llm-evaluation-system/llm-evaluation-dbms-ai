"""
routers/judge.py — FastAPI router for Judge LLM scoring endpoints.

Implements:
  POST /eval/judge/score    — Judge scores one model's answer (absolute)
  POST /eval/judge/contest  — Judge ranks all four models (pairwise contest)
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from evaluators.eval_service import run_contest, run_judge_score
from models.db_models import JudgeScore
from models.schemas import (
    ContestRequest,
    ContestResponse,
    JudgeScoreRequest,
    JudgeScoreResponse,
    SQLExecutionDetails,
)

router = APIRouter(prefix="/eval/judge", tags=["Judge"])


@router.post("/score", response_model=JudgeScoreResponse, summary="Judge scores one model answer")
async def judge_score_endpoint(
    body: JudgeScoreRequest,
    db: AsyncSession = Depends(get_db),
) -> JudgeScoreResponse:
    """
    Ask the Judge LLM to evaluate a single model answer against the ground
    truth.  Returns a structured score (0–10), justification, list of
    hallucinations detected, missing points, and the full composite score.

    If run_id is provided, scores the existing run in the database.
    If model_answer is provided without run_id, scores ad-hoc text.
    """
    if not body.run_id and not body.model_answer:
        raise HTTPException(
            status_code=422,
            detail="Either run_id or model_answer must be provided.",
        )

    run_id = body.run_id

    # If raw answer provided without run_id, generate a stub run for scoring
    if not run_id and body.model_answer:
        from evaluators.eval_service import run_generation
        from models.db_models import EvaluationRun, LLMModel
        from config import DEFAULT_HYPERPARAMS
        import hashlib, json

        model_result = await db.execute(
            select(LLMModel).where(LLMModel.model_id == body.model_id)
        )
        db_model = model_result.scalar_one_or_none()
        if db_model is None:
            raise HTTPException(status_code=404, detail=f"Model not found: {body.model_id}")

        hp_hash = hashlib.sha256(b"ad-hoc").hexdigest()[:16]
        stub_run = EvaluationRun(
            question_id=body.question_id,
            model_id=db_model.id,
            prompt_strategy="zero-shot",
            hyperparam_hash=hp_hash,
            hyperparams=DEFAULT_HYPERPARAMS,
            model_answer=body.model_answer,
            status="completed",
        )
        db.add(stub_run)
        await db.flush()
        run_id = stub_run.id

    try:
        score = await run_judge_score(
            session=db,
            run_id=run_id,
            custom_rubric=body.scoring_rubric,
            force_rescore=body.force_rescore,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Judge API error: {exc}")

    # Build sql_execution_details only when the harness was activated
    sql_details = None
    if score.sql_harness_ran:
        sql_details = SQLExecutionDetails(
            harness_ran=True,
            syntactic_parse_success=score.syntactic_parse_success,
            result_set_f1=score.result_set_f1,
            idiomatic_postgresql=score.idiomatic_postgresql,
            db_execution_context=score.db_execution_context,
        )

    return JudgeScoreResponse(
        score_id=score.id,
        run_id=run_id,
        judge_score_0_10=score.judge_score_0_10 or 0.0,
        justification=score.justification or "",
        hallucinations_detected=score.hallucinations_detected or [],
        missing_points=score.missing_points or [],
        db_correctness_score=score.db_correctness_score or 0.0,
        llm_quality_score=score.llm_quality_score or 0.0,
        prompting_effectiveness_score=score.prompting_effectiveness_score or 0.0,
        efficiency_score=score.efficiency_score or 0.0,
        master_composite_score=score.master_composite_score or 0.0,
        scored_at=score.scored_at or datetime.utcnow(),
        sql_execution_details=sql_details,
    )


@router.post("/contest", response_model=ContestResponse, summary="Judge ranks all model answers")
async def contest_endpoint(
    body: ContestRequest,
    db: AsyncSession = Depends(get_db),
) -> ContestResponse:
    """
    Submit all model answers for one question to the Judge LLM simultaneously.
    The Judge anonymises answers as A/B/C/D, ranks them, and returns placements
    with justifications.  Elo ratings are updated automatically.
    """
    try:
        result = await run_contest(
            session=db,
            question_id=body.question_id,
            run_ids=body.run_ids,
            answers_map=body.answers_map,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Contest error: {exc}")

    return ContestResponse(
        contest_id=result["contest_id"],
        question_id=result["question_id"],
        ranked_model_ids=result["ranked_model_ids"],
        ranking_with_scores=result["rankings"],
        tie_exists=result["tie_exists"],
        tie_model_ids=None,
        judge_reasoning=result["reasoning"],
        elo_updates=result["elo_updates"],
    )
