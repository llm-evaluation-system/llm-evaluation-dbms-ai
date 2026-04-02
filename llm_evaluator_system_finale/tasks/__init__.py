"""
tasks/celery_app.py — Celery application and async task definitions.

Handles all background batch operations:
  - Batch evaluation (dispatch generation jobs across models × questions × strategies)
  - Hyperparameter grid sweeps
  - Prompt strategy comparison runs
  - Self-consistency sampling

All tasks are designed to be idempotent: re-queuing the same task with the
same parameters is safe and will not duplicate database records.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from celery import Celery

from config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND, DEFAULT_HYPERPARAMS, HYPERPARAM_GRID

celery_app = Celery(
    "llm_evaluator",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)


def _run_async(coro):
    """Run an async coroutine synchronously within a Celery task."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(name="tasks.generate_answer", bind=True, max_retries=3)
def generate_answer_task(
    self,
    model_id: str,
    question_id: str,
    prompt_strategy: str,
    hyperparams: Optional[dict] = None,
    eval_version: str = "v1",
):
    """
    Celery task: Generate one model answer for one question.
    Corresponds to a single /eval/generate call dispatched asynchronously.
    """
    async def _inner():
        from database import AsyncSessionLocal
        from evaluators.eval_service import run_generation

        async with AsyncSessionLocal() as session:
            async with session.begin():
                run = await run_generation(
                    session=session,
                    model_id=model_id,
                    question_id=question_id,
                    prompt_strategy=prompt_strategy,
                    hyperparams=hyperparams or DEFAULT_HYPERPARAMS,
                    eval_version=eval_version,
                )
                return {
                    "run_id": run.id,
                    "status": run.status,
                    "model_id": model_id,
                    "question_id": question_id,
                }

    try:
        result = _run_async(_inner())
        
        # Hand the baton to the Judge LLM (scoring only; contests are
        # triggered in a dedicated post-generation sweep by marathon_runner.py
        # to ensure BOTH models have answered before comparing them).
        judge_score_task.delay(run_id=result["run_id"])
        
        return result
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)


@celery_app.task(name="tasks.judge_score_task", bind=True, max_retries=3)
def judge_score_task(self, run_id: str, force_rescore: bool = False):
    """
    Celery task: Score an existing EvaluationRun with the Judge LLM.
    """
    async def _inner():
        from database import AsyncSessionLocal
        from evaluators.eval_service import run_judge_score

        async with AsyncSessionLocal() as session:
            async with session.begin():
                score = await run_judge_score(
                    session=session,
                    run_id=run_id,
                    force_rescore=force_rescore,
                )
                return {
                    "score_id": score.id,
                    "run_id": run_id,
                    "mcs": score.master_composite_score,
                }

    try:
        return _run_async(_inner())
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)


@celery_app.task(name="tasks.contest_task", bind=True, max_retries=12)
def contest_task(self, question_id: str, run_ids: Optional[list[str]] = None, required_model_ids: Optional[list[str]] = None):
    """
    Celery task: Run the Judge LLM pairwise contest for one question.

    Readiness guard: before calling the Judge, verifies that every required
    model has a completed, scored EvaluationRun for this question under the
    default zero-shot / default-hyperparam config.  If not all answers are
    ready yet, the task retries (up to 12 times, 30s apart = ~6 minutes).
    This prevents the contest from firing before the second model finishes
    generating its answer — the root cause of 0 contests in the original run.
    """
    async def _check_readiness():
        """Return True when all required models have a scored run for question_id."""
        from database import AsyncSessionLocal
        from models.db_models import EvaluationRun, JudgeScore, LLMModel
        from sqlalchemy import select, and_

        models_to_check = required_model_ids or ["llama-3.1-70b", "gpt-4o"]

        async with AsyncSessionLocal() as session:
            for model_id_str in models_to_check:
                # Resolve model UUID
                m_res = await session.execute(
                    select(LLMModel).where(LLMModel.model_id == model_id_str)
                )
                db_model = m_res.scalar_one_or_none()
                if db_model is None:
                    return False

                # Check for at least one completed+scored run
                run_res = await session.execute(
                    select(EvaluationRun.id)
                    .join(JudgeScore, JudgeScore.run_id == EvaluationRun.id)
                    .where(
                        and_(
                            EvaluationRun.question_id == question_id,
                            EvaluationRun.model_id == db_model.id,
                            EvaluationRun.status == "completed",
                            EvaluationRun.prompt_strategy == "zero-shot",
                        )
                    )
                    .limit(1)
                )
                if run_res.scalar_one_or_none() is None:
                    return False
        return True

    async def _inner():
        from database import AsyncSessionLocal
        from evaluators.eval_service import run_contest

        async with AsyncSessionLocal() as session:
            async with session.begin():
                result = await run_contest(
                    session=session,
                    question_id=question_id,
                    run_ids=run_ids,
                )
                return {
                    "contest_id": result["contest_id"],
                    "ranked_model_ids": result["ranked_model_ids"],
                }

    try:
        if not _run_async(_check_readiness()):
            # Not all model answers are scored yet — retry after 30 seconds
            raise self.retry(
                exc=ValueError("Not all models scored yet — retrying"),
                countdown=30,
            )
        return _run_async(_inner())
    except self.MaxRetriesExceededError:
        # Log but don't crash — contest simply won't be recorded
        return {"error": "contest_task exceeded max retries", "question_id": question_id}
    except Exception as exc:
        if "Not all models scored yet" in str(exc):
            raise
        raise self.retry(exc=exc, countdown=10)


@celery_app.task(name="tasks.batch_eval_task")
def batch_eval_task(
    model_ids: list[str],
    question_ids: Optional[list[str]],
    prompt_strategies: list[str],
    hyperparams: dict,
    run_judge: bool = True,
    run_contest_after: bool = True,
):
    """
    Celery task: Dispatch a full batch evaluation across all combinations of
    models × questions × strategies.  Chains generation → scoring → contest.

    Returns a summary of dispatched task IDs.
    """
    async def _get_all_question_ids():
        if question_ids:
            return question_ids
        from database import AsyncSessionLocal
        from models.db_models import Question
        from sqlalchemy import select

        async with AsyncSessionLocal() as session:
            result = await session.execute(select(Question.id))
            return [r[0] for r in result.all()]

    all_qids = _run_async(_get_all_question_ids())

    task_ids = []
    # Generate answers
    for model_id in model_ids:
        for qid in all_qids:
            for strategy in prompt_strategies:
                task = generate_answer_task.delay(
                    model_id=model_id,
                    question_id=qid,
                    prompt_strategy=strategy,
                    hyperparams=hyperparams,
                )
                task_ids.append(task.id)

    return {
        "batch_size": len(task_ids),
        "task_ids": task_ids,
        "models": model_ids,
        "question_count": len(all_qids),
        "strategies": prompt_strategies,
    }


@celery_app.task(name="tasks.hyperparam_sweep_task")
def hyperparam_sweep_task(
    model_id: str,
    sweep_param: str,
    param_values: Optional[list] = None,
    question_sample_ids: Optional[list[str]] = None,
):
    """
    Celery task: Run a single-axis hyperparameter sweep for the specified model.
    Iterates over all values of sweep_param, holding all others at defaults.
    """
    values = param_values or HYPERPARAM_GRID.get(sweep_param, [])
    task_ids = []

    for val in values:
        hp = {**DEFAULT_HYPERPARAMS, sweep_param: val}
        task = batch_eval_task.delay(
            model_ids=[model_id],
            question_ids=question_sample_ids,
            prompt_strategies=["zero-shot"],
            hyperparams=hp,
            run_judge=True,
            run_contest_after=False,
        )
        task_ids.append({"param_value": val, "task_id": task.id})

    return {
        "model_id": model_id,
        "sweep_param": sweep_param,
        "values_tested": values,
        "sweep_tasks": task_ids,
    }


@celery_app.task(name="tasks.prompt_comparison_task")
def prompt_comparison_task(
    model_id: str,
    strategies: list[str],
    question_sample_ids: Optional[list[str]] = None,
):
    """
    Celery task: Run a prompt strategy comparison for the specified model.
    Each strategy is evaluated against the question sample.
    """
    task = batch_eval_task.delay(
        model_ids=[model_id],
        question_ids=question_sample_ids,
        prompt_strategies=strategies,
        hyperparams=DEFAULT_HYPERPARAMS,
        run_judge=True,
        run_contest_after=False,
    )
    return {"model_id": model_id, "strategies": strategies, "task_id": task.id}