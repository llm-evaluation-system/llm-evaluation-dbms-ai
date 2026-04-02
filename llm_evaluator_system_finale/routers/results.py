"""
routers/results.py — Results summary, hyperparameter comparison,
                     prompt strategy comparison, and leaderboard endpoints.

Implements:
  GET  /eval/results/summary       — aggregate scores per model/topic/strategy
  GET  /eval/hyperparams/compare   — compare one model across hyperparam grid
  GET  /eval/prompts/compare       — compare prompting strategies per model
  GET  /eval/leaderboard           — global leaderboard across all axes
  GET  /eval/export/json           — full results export (JSON)
  GET  /eval/export/csv            — full results export (CSV)
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select, and_, text
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models.db_models import (
    EvaluationRun,
    JudgeScore,
    Leaderboard,
    LLMModel,
    Question,
    Subtopic,
    Topic,
    HyperparamSweep,
    PromptStrategyComparison,
)
from models.schemas import (
    LeaderboardEntry,
    LeaderboardResponse,
    ModelTopicScore,
    ResultsSummaryResponse,
)

router = APIRouter(tags=["Results & Leaderboard"])


# ── /eval/results/summary ─────────────────────────────────────────────────────
@router.get(
    "/eval/results/summary",
    response_model=ResultsSummaryResponse,
    summary="Aggregate scores per model/topic/strategy",
)
async def results_summary(
    model_id: Optional[str] = Query(None, description="Filter by model ID"),
    topic: Optional[str] = Query(None, description="Filter by topic name"),
    subtopic: Optional[str] = Query(None, description="Filter by subtopic name"),
    prompt_type: Optional[str] = Query(None, description="Filter by prompt strategy"),
    difficulty: Optional[str] = Query(None, description="Filter by difficulty tier"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> ResultsSummaryResponse:
    """
    Return aggregate performance statistics grouped by model, subtopic,
    and prompting strategy.  Supports fine-grained filtering by any dimension.
    """
    stmt = (
        select(
            LLMModel.model_id,
            LLMModel.display_name,
            Subtopic.name.label("subtopic"),
            EvaluationRun.prompt_strategy,
            func.count(EvaluationRun.id).label("question_count"),
            func.avg(JudgeScore.master_composite_score).label("avg_mcs"),
            func.avg(JudgeScore.db_correctness_score).label("avg_db_correctness"),
            func.avg(JudgeScore.llm_quality_score).label("avg_llm_quality"),
            func.avg(JudgeScore.hallucination_rate).label("avg_hallucination_rate"),
            func.avg(EvaluationRun.total_latency_ms).label("avg_latency_ms"),
        )
        .join(LLMModel, EvaluationRun.model_id == LLMModel.id)
        .join(JudgeScore, JudgeScore.run_id == EvaluationRun.id)
        .join(Question, EvaluationRun.question_id == Question.id)
        .join(Subtopic, Question.subtopic_id == Subtopic.id)
        .join(Topic, Subtopic.topic_id == Topic.id)
        .where(EvaluationRun.status == "completed")
    )

    if model_id:
        stmt = stmt.where(LLMModel.model_id == model_id)
    if topic:
        stmt = stmt.where(Topic.name.ilike(f"%{topic}%"))
    if subtopic:
        stmt = stmt.where(Subtopic.name.ilike(f"%{subtopic}%"))
    if prompt_type:
        stmt = stmt.where(EvaluationRun.prompt_strategy == prompt_type)
    if difficulty:
        stmt = stmt.where(Question.difficulty == difficulty)

    stmt = stmt.group_by(
        LLMModel.model_id,
        LLMModel.display_name,
        Subtopic.name,
        EvaluationRun.prompt_strategy,
    ).offset(offset).limit(limit)

    result = await db.execute(stmt)
    rows = result.all()

    entries = [
        ModelTopicScore(
            model_id=row.model_id,
            model_display_name=row.display_name,
            subtopic=row.subtopic,
            prompt_strategy=row.prompt_strategy,
            question_count=row.question_count,
            avg_mcs=round(row.avg_mcs or 0.0, 2),
            avg_db_correctness=round(row.avg_db_correctness or 0.0, 2),
            avg_llm_quality=round(row.avg_llm_quality or 0.0, 2),
            avg_hallucination_rate=round(row.avg_hallucination_rate or 0.0, 4),
            avg_latency_ms=round(row.avg_latency_ms or 0.0, 1),
        )
        for row in rows
    ]

    return ResultsSummaryResponse(
        total=len(entries),
        results=entries,
        generated_at=datetime.utcnow(),
    )


@router.get(
    "/eval/hyperparams/compare",
    summary="Compare model performance across hyperparameter values",
)
async def hyperparams_compare(
    model_id: str = Query(..., description="Logical model ID"),
    param_name: str = Query(..., description="Hyperparameter to compare (e.g. temperature)"),
    subtopic: Optional[str] = Query(None),
    trigger_sweep: bool = Query(False, description="Dispatch new Celery sweep jobs for any missing param values"),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Retrieve aggregate MCS scores for each distinct value of a given
    hyperparameter for the specified model.  Returns the cross-model
    sensitivity summary and the recommended optimal value.

    When trigger_sweep=True, dispatches a Celery sweep task for the full
    HYPERPARAM_GRID values and persists a HyperparamSweep metadata record
    (Section 3.2 / FastAPI Implementation Checklist Phase 3).
    """
    from config import HYPERPARAM_GRID, FAST_SWEEP_SAMPLE_FRACTION
    from models.db_models import HyperparamSweep

    model_result = await db.execute(
        select(LLMModel).where(LLMModel.model_id == model_id)
    )
    db_model = model_result.scalar_one_or_none()
    if db_model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    # ── Persist HyperparamSweep metadata record ───────────────────────────────
    sweep_id = None
    if trigger_sweep:
        param_values = HYPERPARAM_GRID.get(param_name)
        if param_values is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown hyperparameter '{param_name}'. "
                       f"Valid params: {list(HYPERPARAM_GRID.keys())}",
            )
        sweep_record = HyperparamSweep(
            model_id=db_model.id,
            sweep_param=param_name,
            param_grid={param_name: param_values},
            status="dispatched",
        )
        db.add(sweep_record)
        await db.flush()
        sweep_id = sweep_record.id

        try:
            from tasks import hyperparam_sweep_task
            hyperparam_sweep_task.delay(
                model_id=model_id,
                sweep_param=param_name,
                param_values=param_values,
            )
            sweep_record.status = "running"
        except Exception:
            sweep_record.status = "dispatch_failed"
        await db.flush()

    # ── Aggregate existing runs ───────────────────────────────────────────────
    param_value = EvaluationRun.hyperparams.op("->>")(param_name)

    stmt = (
        select(
            param_value.label("param_value"),
            func.avg(JudgeScore.master_composite_score).label("avg_mcs"),
            func.stddev(JudgeScore.master_composite_score).label("score_variance"),
            func.count(EvaluationRun.id).label("run_count"),
        )
        .join(JudgeScore, JudgeScore.run_id == EvaluationRun.id)
        .join(Question, EvaluationRun.question_id == Question.id)
        .join(Subtopic, Question.subtopic_id == Subtopic.id)
        .where(
            EvaluationRun.model_id == db_model.id,
            EvaluationRun.status == "completed",
        )
        .group_by(param_value)
    )

    if subtopic:
        stmt = stmt.where(Subtopic.name.ilike(f"%{subtopic}%"))

    result = await db.execute(stmt)
    rows = result.all()

    param_results = []
    best_val, best_score = None, -1.0

    for row in rows:
        val = row.param_value
        if val is None:
            continue

        avg_mcs = row.avg_mcs or 0.0
        variance = row.score_variance or 0.0

        param_results.append({
            "param_name": param_name,
            "param_value": val,
            "avg_mcs": round(avg_mcs, 2),
            "score_variance": round(variance, 4),
            "run_count": row.run_count,
        })

        if avg_mcs > best_score:
            best_score = avg_mcs
            best_val = val

    # Compute sensitivity tier (Section 3.3)
    if param_results:
        scores = [r["avg_mcs"] for r in param_results]
        score_range = max(scores) - min(scores) if len(scores) > 1 else 0.0
        sensitivity = "HIGH" if score_range > 15 else ("MEDIUM" if score_range > 7 else "LOW")
    else:
        sensitivity = "UNKNOWN"

    response: dict = {
        "model_id": model_id,
        "param_name": param_name,
        "results": sorted(param_results, key=lambda r: r["param_value"]),
        "recommended_value": best_val,
        "sensitivity": sensitivity,
    }
    if sweep_id:
        response["sweep_id"] = sweep_id
        response["sweep_status"] = "dispatched — poll /eval/results/summary for progress"
    else:
        response["note"] = (
            "Set trigger_sweep=true to dispatch a Celery sweep, "
            "or run /eval/generate with varying hyperparams to populate this report."
        )
    return response


@router.get(
    "/eval/prompts/compare",
    summary="Compare prompting strategies for a model",
)
async def prompts_compare(
    model_id: str = Query(..., description="Logical model ID"),
    subtopic: Optional[str] = Query(None),
    trigger_runs: bool = Query(
        False,
        description="Dispatch Celery jobs to evaluate all 9 strategies if not yet run",
    ),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Compare all prompting strategies that have been evaluated for the specified
    model, returning accuracy lift over zero-shot, consistency variance, and
    token efficiency metrics (Section 4.4).

    When trigger_runs=True, dispatches a prompt_comparison_task via Celery and
    persists a PromptStrategyComparison metadata record (Phase 3 checklist).
    """
    from config import FAST_SWEEP_SAMPLE_FRACTION
    from models.schemas import GenerateRequest

    ALL_STRATEGIES = [
        "zero-shot", "one-shot", "few-shot", "cot",
        "few-shot-cot", "self-consistency", "role-prompting",
        "least-to-most", "react",
    ]

    model_result = await db.execute(
        select(LLMModel).where(LLMModel.model_id == model_id)
    )
    db_model = model_result.scalar_one_or_none()
    if db_model is None:
        raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")

    # ── Persist PromptStrategyComparison record ───────────────────────────────
    comparison_id = None
    if trigger_runs:
        comp_record = PromptStrategyComparison(
            model_id=db_model.id,
            strategies=ALL_STRATEGIES,
            status="dispatched",
        )
        db.add(comp_record)
        await db.flush()
        comparison_id = comp_record.id

        try:
            from tasks import prompt_comparison_task
            prompt_comparison_task.delay(
                model_id=model_id,
                strategies=ALL_STRATEGIES,
            )
            comp_record.status = "running"
        except Exception:
            comp_record.status = "dispatch_failed"
        await db.flush()

    # ── Aggregate existing runs ───────────────────────────────────────────────
    stmt = (
        select(
            EvaluationRun.prompt_strategy,
            func.avg(JudgeScore.master_composite_score).label("avg_mcs"),
            func.stddev(JudgeScore.master_composite_score).label("score_stddev"),
            func.avg(EvaluationRun.input_tokens + EvaluationRun.output_tokens).label("avg_tokens"),
            func.avg(JudgeScore.format_compliance_score).label("avg_format"),
            func.count(EvaluationRun.id).label("run_count"),
        )
        .join(JudgeScore, JudgeScore.run_id == EvaluationRun.id)
        .join(Question, EvaluationRun.question_id == Question.id)
        .join(Subtopic, Question.subtopic_id == Subtopic.id)
        .where(
            EvaluationRun.model_id == db_model.id,
            EvaluationRun.status == "completed",
        )
    )
    if subtopic:
        stmt = stmt.where(Subtopic.name.ilike(f"%{subtopic}%"))

    stmt = stmt.group_by(EvaluationRun.prompt_strategy)
    result = await db.execute(stmt)
    rows = result.all()

    if not rows:
        resp: dict = {
            "model_id": model_id,
            "strategies": [],
            "note": "No evaluated runs found. Set trigger_runs=true to dispatch evaluations.",
        }
        if comparison_id:
            resp["comparison_id"] = comparison_id
        return resp

    zero_shot_mcs = next(
        (float(r.avg_mcs) for r in rows if r.prompt_strategy == "zero-shot"), None
    )

    strategies = []
    best_strategy, best_mcs = None, -1.0
    for row in rows:
        avg_mcs = float(row.avg_mcs or 0.0)
        lift = ((avg_mcs - zero_shot_mcs) / max(zero_shot_mcs, 1e-6)
                if zero_shot_mcs else 0.0)
        # Cast to float explicitly: SQLAlchemy returns avg() of integer columns as
        # decimal.Decimal on PostgreSQL, which cannot be divided by a Python float.
        avg_tokens = float(row.avg_tokens or 1.0)
        token_efficiency = lift / max(avg_tokens / 1000, 0.001) if lift > 0 else 0.0

        entry = {
            "strategy": row.prompt_strategy,
            "avg_mcs": round(avg_mcs, 2),
            "accuracy_lift_vs_zeroshot": round(lift, 4),
            "consistency_sigma": round(float(row.score_stddev or 0.0), 4),
            "avg_tokens": round(avg_tokens, 0),
            "token_efficiency": round(token_efficiency, 4),
            "format_compliance_rate": round(float(row.avg_format or 0.0), 4),
            "run_count": row.run_count,
        }
        strategies.append(entry)
        if avg_mcs > best_mcs:
            best_mcs = avg_mcs
            best_strategy = row.prompt_strategy

    response: dict = {
        "model_id": model_id,
        "strategies": sorted(strategies, key=lambda s: s["avg_mcs"], reverse=True),
        "recommended_strategy": best_strategy,
        "zero_shot_baseline_mcs": round(zero_shot_mcs or 0.0, 2),
    }
    if comparison_id:
        response["comparison_id"] = comparison_id
    return response


@router.get(
    "/eval/leaderboard",
    response_model=LeaderboardResponse,
    summary="Global leaderboard across all evaluation axes",
)
async def leaderboard(
    sort_by: str = Query("mcs_score", description="mcs_score|elo_rating|win_rate|db_correctness"),
    db: AsyncSession = Depends(get_db),
) -> LeaderboardResponse:
    """
    Return the global leaderboard sorted by the specified axis.  Each entry
    includes the Master Composite Score, Elo rating, contest win rate, best
    and worst DBMS topics, hallucination rate, and average latency.
    """
    valid_sort = {"mcs_score", "elo_rating", "win_rate", "db_correctness", "llm_quality"}
    if sort_by not in valid_sort:
        sort_by = "mcs_score"

    stmt = (
        select(Leaderboard, LLMModel)
        .join(LLMModel, Leaderboard.model_id == LLMModel.id)
        .order_by(
            getattr(Leaderboard, sort_by).desc().nullslast()
            if sort_by != "win_rate"
            else Leaderboard.win_rate.desc().nullslast()
        )
    )
    result = await db.execute(stmt)
    rows = result.all()

    # Total contest count
    from models.db_models import ContestResult
    contest_count_result = await db.execute(
        select(func.count()).select_from(ContestResult)
    )
    total_contests = contest_count_result.scalar() or 0

    # Total runs
    run_count_result = await db.execute(
        select(func.count()).select_from(EvaluationRun)
    )
    total_runs = run_count_result.scalar() or 0

    entries = [
        LeaderboardEntry(
            rank=idx + 1,
            model_id=m.model_id,
            display_name=m.display_name,
            provider=m.provider,
            mcs_score=lb.mcs_score,
            db_correctness=lb.db_correctness,
            llm_quality=lb.llm_quality,
            elo_rating=lb.elo_rating,
            contest_wins=lb.contest_wins,
            contest_total=lb.contest_total,
            win_rate=lb.win_rate,
            best_prompt_strategy=lb.best_prompt_strategy,
            best_topic=lb.best_topic,
            worst_topic=lb.worst_topic,
            avg_latency_ms=lb.avg_latency_ms,
            hallucination_rate=lb.hallucination_rate,
            last_updated=lb.last_updated,
        )
        for idx, (lb, m) in enumerate(rows)
    ]

    return LeaderboardResponse(
        total_models=len(entries),
        total_contests=total_contests,
        total_runs=total_runs,
        entries=entries,
        generated_at=datetime.utcnow(),
    )


@router.post(
    "/eval/leaderboard/refresh",
    summary="Recompute leaderboard aggregates for all models",
)
async def refresh_leaderboard(
    db: AsyncSession = Depends(get_db),
) -> dict:
    """
    Recompute and upsert the leaderboard row for every model that has at
    least one completed + scored evaluation run.

    Call this after a bulk evaluation run to ensure the leaderboard reflects
    the latest results.  The operation is idempotent — safe to call multiple
    times.
    """
    from evaluators.eval_service import _refresh_leaderboard_for_model

    # Find all distinct model UUIDs with completed, scored runs
    result = await db.execute(
        select(EvaluationRun.model_id)
        .join(JudgeScore, JudgeScore.run_id == EvaluationRun.id)
        .where(EvaluationRun.status == "completed")
        .distinct()
    )
    model_ids = [r[0] for r in result.all()]

    refreshed = []
    for model_id in model_ids:
        await _refresh_leaderboard_for_model(db, model_id)
        # Resolve display name for the response
        m_result = await db.execute(
            select(LLMModel).where(LLMModel.id == model_id)
        )
        m = m_result.scalar_one_or_none()
        if m:
            refreshed.append(m.model_id)

    return {
        "status": "ok",
        "models_refreshed": len(refreshed),
        "model_ids": refreshed,
    }


# ── /eval/export/json ────────────────────────────────────────────────────────
@router.get("/eval/export/json", summary="Export all results as JSON")
async def export_json(
    model_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Export all evaluation results with scores as a downloadable JSON file."""
    stmt = (
        select(EvaluationRun, JudgeScore, LLMModel, Question)
        .join(JudgeScore, JudgeScore.run_id == EvaluationRun.id, isouter=True)
        .join(LLMModel, EvaluationRun.model_id == LLMModel.id)
        .join(Question, EvaluationRun.question_id == Question.id)
        .where(EvaluationRun.status == "completed")
    )
    if model_id:
        stmt = stmt.where(LLMModel.model_id == model_id)

    result = await db.execute(stmt)
    rows = result.all()

    data = [
        {
            "run_id": run.id,
            "model_id": m.model_id,
            "question_id": q.id,
            "question_type": q.question_type,
            "prompt_strategy": run.prompt_strategy,
            "mcs": score.master_composite_score if score else None,
            "db_correctness": score.db_correctness_score if score else None,
            "llm_quality": score.llm_quality_score if score else None,
            "hallucination_rate": score.hallucination_rate if score else None,
            "judge_score_0_10": score.judge_score_0_10 if score else None,
            "total_latency_ms": run.total_latency_ms,
            "cost_usd": run.cost_usd,
        }
        for run, score, m, q in rows
    ]

    json_bytes = json.dumps(data, indent=2, default=str).encode()
    return StreamingResponse(
        iter([json_bytes]),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=eval_results.json"},
    )


# ── /eval/export/csv ─────────────────────────────────────────────────────────
@router.get("/eval/export/csv", summary="Export all results as CSV")
async def export_csv(
    model_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Export all evaluation results with scores as a downloadable CSV file."""
    stmt = (
        select(EvaluationRun, JudgeScore, LLMModel, Question)
        .join(JudgeScore, JudgeScore.run_id == EvaluationRun.id, isouter=True)
        .join(LLMModel, EvaluationRun.model_id == LLMModel.id)
        .join(Question, EvaluationRun.question_id == Question.id)
        .where(EvaluationRun.status == "completed")
    )
    if model_id:
        stmt = stmt.where(LLMModel.model_id == model_id)

    result = await db.execute(stmt)
    rows = result.all()

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "run_id", "model_id", "question_id", "question_type",
        "prompt_strategy", "mcs", "db_correctness", "llm_quality",
        "hallucination_rate", "judge_score_0_10", "total_latency_ms", "cost_usd",
    ])
    writer.writeheader()
    for run, score, m, q in rows:
        writer.writerow({
            "run_id": run.id,
            "model_id": m.model_id,
            "question_id": q.id,
            "question_type": q.question_type,
            "prompt_strategy": run.prompt_strategy,
            "mcs": score.master_composite_score if score else "",
            "db_correctness": score.db_correctness_score if score else "",
            "llm_quality": score.llm_quality_score if score else "",
            "hallucination_rate": score.hallucination_rate if score else "",
            "judge_score_0_10": score.judge_score_0_10 if score else "",
            "total_latency_ms": run.total_latency_ms or "",
            "cost_usd": run.cost_usd or "",
        })

    csv_bytes = output.getvalue().encode()
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=eval_results.csv"},
    )
