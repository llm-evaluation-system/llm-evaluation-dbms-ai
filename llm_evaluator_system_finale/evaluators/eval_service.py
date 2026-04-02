"""
evaluators/eval_service.py — Core evaluation orchestration service.

This module is the central coordinator for Phase 1 (Generation) and
Phase 2 (Judging).  It wires together the LLM clients, prompt builders,
SQL harness, hallucination detectors, format compliance checkers, and
composite scoring engine into a single coherent workflow per question run.

Public entry points:
  run_generation()  — generate an answer for one question with one model
  run_judge_score() — score an existing run with the Judge LLM
  run_contest()     — contest all four model answers for one question
  run_self_consistency() — sample k times and aggregate
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Optional
from datetime import datetime

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from config import (
    CHALLENGER_MODELS,
    DEFAULT_HYPERPARAMS,
    JUDGE_MODEL,
    SELF_CONSISTENCY_K,
    SELF_CONSISTENCY_TEMPERATURE,
)
from database import AsyncSessionLocal
from evaluators.format_compliance import compute_format_compliance_score
from evaluators.hallucination import (
    compute_hallucination_metrics,
    run_automated_hallucination_checks,
)
from evaluators.sql_harness import (
    check_idiomatic_postgresql,
    check_result_set_accuracy,
    check_syntactic_validity,
)
from judge.elo import update_elo_ratings
from judge.judge_llm import judge_absolute_score, judge_contest
from judge.judge_llm import get_llm_client
from models.db_models import (
    ContestResult,
    EvaluationRun,
    HallucinationRecord,
    JudgeScore,
    LLMModel,
    Question,
    Subtopic,
)
from models.schemas import Hyperparams
from prompting.few_shot_store import get_examples
from prompting.templates import build_prompt
from scoring.composite import (
    ConceptualScores,
    DBCorrectnessBundle,
    EfficiencyBundle,
    LLMQualityBundle,
    QueryOptScores,
    ReasoningScores,
    SQLScores,
    SchemaScores,
    TransactionScores,
    compute_mcs,
    judge_score_to_normalized,
)


def _make_hyperparam_hash(hyperparams: dict) -> str:
    canonical = json.dumps(hyperparams, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


async def _get_model_config(session: AsyncSession, model_id: str) -> Optional[dict]:
    """Fetch model config from DB, falling back to config.py for known models."""
    result = await session.execute(
        select(LLMModel).where(LLMModel.model_id == model_id)
    )
    db_model = result.scalar_one_or_none()
    if db_model:
        return {
            "model_id": db_model.model_id,
            "provider": db_model.provider,
            "api_model": db_model.api_model,
            "display_name": db_model.display_name,
            "cost_per_1k_input_tokens": db_model.cost_per_1k_input_tokens or 0.0,
            "cost_per_1k_output_tokens": db_model.cost_per_1k_output_tokens or 0.0,
            "max_context_tokens": db_model.max_context_tokens,
            "supports_seed": db_model.supports_seed,
            "is_judge": db_model.is_judge,
        }
    return CHALLENGER_MODELS.get(model_id)


# ── Phase 1: Generation ───────────────────────────────────────────────────────
async def run_generation(
    session: AsyncSession,
    model_id: str,
    question_id: str,
    prompt_strategy: str = "zero-shot",
    hyperparams: Optional[dict] = None,
    eval_version: str = "v1",
    force_rerun: bool = False,
) -> EvaluationRun:
    """
    Generate one answer for one question with one model under one prompt strategy.

    Implements the idempotency guarantee: the composite key
    (question_id, model_id, prompt_strategy, hyperparam_hash) is unique.
    Re-running the same configuration overwrites the existing run record.

    Returns the EvaluationRun ORM object (persisted to DB).
    """
    hp = {**DEFAULT_HYPERPARAMS, **(hyperparams or {})}
    hp_hash = _make_hyperparam_hash(hp)

    # ── Check for existing run (idempotency) ──
    model_result = await session.execute(
        select(LLMModel).where(LLMModel.model_id == model_id)
    )
    db_model = model_result.scalar_one_or_none()
    if db_model is None:
        raise ValueError(f"Unknown model_id: {model_id}")

    q_result = await session.execute(
        select(Question).where(Question.id == question_id)
    )
    question = q_result.scalar_one_or_none()
    if question is None:
        raise ValueError(f"Unknown question_id: {question_id}")

    existing_result = await session.execute(
        select(EvaluationRun).where(
            and_(
                EvaluationRun.question_id == question_id,
                EvaluationRun.model_id == db_model.id,
                EvaluationRun.prompt_strategy == prompt_strategy,
                EvaluationRun.hyperparam_hash == hp_hash,
            )
        )
    )
    existing_run = existing_result.scalar_one_or_none()
    if existing_run and not force_rerun:
        return existing_run

    # ── Build the prompt ──
    examples = await get_examples(
        session,
        question_id=question_id,
        subtopic_id=question.subtopic_id,
        difficulty=question.difficulty,
        strategy=prompt_strategy,
        n=3,
    )

    # Resolve schema_fixture and judge_hint for sql+practical questions so the
    # prompt tells the model the exact table/column names the harness expects.
    # Without this, models write natural names (e.g. Employees, from, to) that
    # differ from fixture names (femployees, ffrom, fto) causing execution failures.
    from question_bank.sql_fixtures import (
        get_schema_fixture as _get_schema, get_judge_hint as _get_hint,
        needs_sql_harness as _needs_harness,
    )
    _prompt_raw_id = (question.id or "").replace("-", "")[:12]
    _gen_schema = (
        _get_schema(_prompt_raw_id)
        if _needs_harness(question.question_type or "", question.tags or [])
        else None
    )
    _gen_hint = _get_hint(_prompt_raw_id)

    prompt_pair = build_prompt(
        strategy=prompt_strategy,
        question=question.question_text,
        examples=examples,
        system_prompt_style=hp.get("system_prompt_style", "expert-persona"),
        schema_fixture=_gen_schema,
        judge_hint=_gen_hint,
    )

    # ── Create / update run record ──
    if existing_run:
        run = existing_run
    else:
        run = EvaluationRun(
            question_id=question_id,
            model_id=db_model.id,
            prompt_strategy=prompt_strategy,
            hyperparam_hash=hp_hash,
            hyperparams=hp,
            system_prompt=prompt_pair.system_prompt,
            prompt_used=prompt_pair.user_prompt,
            status="pending",
            eval_version=eval_version,
        )
        session.add(run)
        await session.flush()

    # ── Call the LLM ──
    model_config = await _get_model_config(session, model_id)
    if model_config is None:
        run.status = "error"
        run.error_message = f"No model config found for {model_id}"
        return run

    client = get_llm_client(model_config)
    try:
        llm_response = await client.complete(
            prompt=prompt_pair.user_prompt,
            system_prompt=prompt_pair.system_prompt,
            hyperparams=hp,
        )
        run.model_answer = llm_response.answer_text
        run.input_tokens = llm_response.input_tokens
        run.output_tokens = llm_response.output_tokens
        run.cost_usd = llm_response.cost_usd
        run.ttft_ms = llm_response.ttft_ms
        run.total_latency_ms = llm_response.total_latency_ms
        run.tokens_per_second = llm_response.tokens_per_second
        run.retry_count = llm_response.retry_count
        run.status = "completed"
    except Exception as exc:
        run.status = "failed"
        run.error_message = str(exc)

    await session.flush()

    # ── Persist LatencyRecord (Section 2.7 — all latency data in PostgreSQL) ──
    if run.status == "completed":
        from models.db_models import LatencyRecord
        existing_lr = await session.execute(
            select(LatencyRecord).where(LatencyRecord.run_id == run.id)
        )
        if existing_lr.scalar_one_or_none() is None:
            lr = LatencyRecord(
                run_id=run.id,
                ttft_ms=run.ttft_ms,
                total_latency_ms=run.total_latency_ms or 0.0,
                tokens_per_second=run.tokens_per_second,
                input_tokens=run.input_tokens,
                output_tokens=run.output_tokens,
                api_retry_count=run.retry_count,
                cost_usd=run.cost_usd,
            )
            session.add(lr)
            await session.flush()

    return run


# ── Phase 2a: Absolute Judge Scoring ─────────────────────────────────────────
def _apply_rename_map(text: str, rename_map: dict) -> str:
    """
    Replace fixture-specific table/column names with natural names before
    the model answer is sent to the Judge LLM.

    The SQL execution harness uses renamed identifiers (e.g. femployees,
    ffrom, fto) to avoid PostgreSQL reserved-word conflicts and cross-
    question name collisions.  The Judge LLM has no knowledge of these
    renames and flags them as syntax errors, even when objective execution
    evidence (F1=1.0, syntactic_parse_success=1) proves they are correct.

    This function reverses those renames so the Judge sees natural table
    names (Employees, from, to) that match its training data, eliminating
    false-positive hallucination reports without affecting harness execution.
    """
    import re as _re
    result = text
    for fixture_name, natural_name in rename_map.items():
        result = _re.sub(rf"\b{_re.escape(fixture_name)}\b", natural_name, result)
    return result


async def run_judge_score(
    session: AsyncSession,
    run_id: str,
    custom_rubric: Optional[str] = None,
    force_rescore: bool = False,
) -> JudgeScore:
    """
    Score an existing EvaluationRun with the Judge LLM.

    Performs:
    1. Automated hallucination detection (regex + pattern matching)
    2. Automated SQL harness checks (if needs_sql_harness(question_type, tags))
    3. LLM judge absolute scoring (structured JSON response)
    4. Composite score computation (DB correctness + LLM quality + MCS)

    Returns the persisted JudgeScore ORM object.
    """
    run_result = await session.execute(
        select(EvaluationRun).where(EvaluationRun.id == run_id)
    )
    run = run_result.scalar_one_or_none()
    if run is None:
        raise ValueError(f"EvaluationRun not found: {run_id}")

    # Fetch the model record so we can refresh its leaderboard row later
    db_model_result = await session.execute(
        select(LLMModel).where(LLMModel.id == run.model_id)
    )
    db_model = db_model_result.scalar_one_or_none()

    # Check for existing score
    existing_score_result = await session.execute(
        select(JudgeScore).where(JudgeScore.run_id == run_id)
    )
    existing_score = existing_score_result.scalar_one_or_none()
    if existing_score and not force_rescore:
        return existing_score

    q_result = await session.execute(
        select(Question).where(Question.id == run.question_id)
    )
    question = q_result.scalar_one_or_none()
    if question is None:
        raise ValueError(f"Question not found for run {run_id}")

    answer_text = run.model_answer or ""

    # ── Step 1: Automated hallucination checks ──
    auto_hallucinations = run_automated_hallucination_checks(answer_text)
    hallucination_metrics = compute_hallucination_metrics(auto_hallucinations)

    # Persist hallucination records
    for h in auto_hallucinations:
        hr = HallucinationRecord(
            run_id=run_id,
            hallucination_type=h["type"],
            description=h["text"],
            severity=h["severity"],
            detected_by=h["detected_by"],
        )
        session.add(hr)

    # ── Step 2: Automated SQL checks (if applicable) ──
    sql_scores = SQLScores()
    db_execution_context: str | None = None  # accumulated EXPLAIN/execution output
    # Tracks whether SQL was actually found and executed in the model's answer.
    # When False (answer is conceptual prose with no SQL), the SQL scorer is
    # excluded from db_correctness even if needs_sql_harness() returned True.
    _sql_was_executed: bool = False

    # Import tag-based routing helpers — these replace all legacy
    # `question_type == 'sql'` checks that became dead code after the
    # question bank correction (question_type is now only 'conceptual'|'practical').
    from question_bank.sql_fixtures import (
        needs_sql_harness, get_judge_hint, get_explain_context,
        get_db_correctness_route,
    )

    _q_tags: list[str] = question.tags or []
    _q_type: str = question.question_type or "conceptual"

    # Run the SQL harness for practical sql-tagged questions only.
    if needs_sql_harness(_q_type, _q_tags) and answer_text:
        syntax_check = await check_syntactic_validity(answer_text)
        sql_scores.syntactic_parse_success = (
            syntax_check["score"] if syntax_check["score"] is not None else 0.5
        )
        # Accumulate execution context for the Judge
        if syntax_check.get("error"):
            db_execution_context = f"Syntax check error: {syntax_check['error']}"

        if question.schema_fixture and question.expected_rows:
            rs_check = await check_result_set_accuracy(
                answer_text,
                question.schema_fixture,
                question.expected_rows or [],
            )
            # f1=None signals "no SQL found in answer" — harness inapplicable.
            # f1=0.0 signals "SQL found but returned wrong rows" — genuine failure.
            if rs_check["f1"] is None and rs_check.get("error") == "No SQL found in answer":
                # Model gave a conceptual answer to a SQL-tagged question.
                # Do NOT feed zeros into the SQL scorer; fall through to
                # conceptual scoring.  _sql_was_executed stays False.
                db_execution_context = "No SQL statement found in model answer."
            else:
                _sql_was_executed = True
                sql_scores.result_set_f1 = rs_check["f1"] if rs_check["f1"] is not None else 0.0
                # Build execution context summary for Judge
                rs_summary_parts = [
                    f"Result set accuracy: F1={rs_check.get('f1')}, "
                    f"precision={rs_check.get('precision')}, recall={rs_check.get('recall')}",
                    f"Returned rows: {rs_check.get('returned_count')}, "
                    f"Expected rows: {rs_check.get('expected_count')}",
                ]
                if rs_check.get("error"):
                    rs_summary_parts.append(f"Execution error: {rs_check['error']}")
                db_execution_context = "\n".join(rs_summary_parts)

        pg_check = check_idiomatic_postgresql(answer_text)
        sql_scores.idiomatic_postgresql = pg_check["score"]

    # For all practical questions with a schema fixture: try EXPLAIN ANALYZE.
    # This covers both sql practical questions (already handled above if harness
    # ran) and non-sql practical questions that have a schema (e.g. indexing,
    # join-cost analysis questions).
    if _q_type == "practical" and answer_text and not db_execution_context:
        explain_result = await _try_explain_analyze(answer_text, question.schema_fixture)
        if explain_result:
            db_execution_context = explain_result

    # Retrieve pre-computed judge hint + EXPLAIN context from fixtures.
    # The question id in DB is UUID-padded; recover the original 12-char hex id
    # by stripping dashes and taking the first 12 hex characters.
    _raw_id = (question.id or "").replace("-", "")[:12]
    judge_hint = get_judge_hint(_raw_id)
    fixture_explain = get_explain_context(_raw_id)
    if fixture_explain and not db_execution_context:
        db_execution_context = fixture_explain

    # ── Reverse-translate fixture names → natural names for the Judge ─────────
    # The SQL harness uses renamed tables (femployees, ffrom, fto) to avoid
    # reserved-word conflicts. The model is told to use these names via the
    # generation prompt and the harness executes them correctly (F1=1.0).
    # However, the Judge LLM flags them as "syntax errors" based on training
    # priors even when execution evidence proves they are correct.
    # Fix: translate the answer back to natural names before the judge sees it.
    # The original answer_text (with fixture names) is preserved for harness use.
    from question_bank.sql_fixtures import get_fixture_rename_map
    _judge_answer = _apply_rename_map(answer_text, get_fixture_rename_map(_raw_id))

    # ── Step 3: Format compliance ──
    format_result = compute_format_compliance_score(
        answer=answer_text,
        question_type=_q_type,
        tags=_q_tags,
    )

    # ── Step 4: LLM Judge absolute scoring ──
    # Wrapped in a try/except so that judge timeouts or context-overflow errors
    # (which occur on very long expected_answer fields, e.g. proofs or large
    # schema normalization answers) never leave a run with null scores.
    # On failure we fall back to a conservative automated-only score derived
    # from hallucination detection and format compliance signals already computed.
    _judge_failed = False
    try:
        judge_result = await judge_absolute_score(
            question=question.question_text,
            ground_truth=question.expected_answer or "",
            model_answer=_judge_answer,
            question_type=_q_type,
            custom_rubric=custom_rubric,
            db_execution_context=db_execution_context,
            judge_hint=judge_hint,
            tags=_q_tags,
        )
    except Exception as _judge_exc:
        _judge_failed = True
        # Build a conservative fallback: assume mid-range correctness (5/10)
        # penalised by any hallucinations already detected automatically.
        _auto_pen = min(hallucination_metrics["severity_score"] * 2, 2.0)
        _fallback_score = max(3.0, 5.0 - _auto_pen)
        judge_result = {
            "score": _fallback_score,
            "justification": (
                f"[AUTO-FALLBACK — judge call failed: {_judge_exc!s:.200}] "
                "Score estimated from automated hallucination and format checks only."
            ),
            "hallucinations": [],
            "missing_points": [],
            "factual_correctness": _fallback_score / 10,
            "completeness": _fallback_score / 10,
            "absence_of_contradiction": max(0.0, (_fallback_score - 0.5) / 10),
            "topic_specificity": _fallback_score / 10,
            "_judge_fallback": True,
        }

    score_0_10 = float(judge_result.get("score", 5.0))
    norm_score = judge_score_to_normalized(score_0_10)

    # Extract sub-scores from judge response
    factual_correctness = float(judge_result.get("factual_correctness", norm_score))
    completeness = float(judge_result.get("completeness", norm_score))
    absence_of_contradiction = float(judge_result.get("absence_of_contradiction", norm_score))
    topic_specificity = float(judge_result.get("topic_specificity", norm_score))

    # Merge LLM-detected hallucinations with automated ones
    llm_hallucinations = judge_result.get("hallucinations", [])
    all_hallucinations = auto_hallucinations + llm_hallucinations
    hallucination_metrics = compute_hallucination_metrics(all_hallucinations)

    for h in llm_hallucinations:
        hr = HallucinationRecord(
            run_id=run_id,
            hallucination_type=h["type"],
            description=h["text"],
            severity=h["severity"],
            detected_by="llm_judge",
        )
        session.add(hr)

    # ── Step 5: Build DB correctness pillar ──
    # Derive a semantic routing key from tags (replaces legacy question_type values
    # like 'sql', 'schema', 'transaction' that no longer appear in the question bank).
    _db_route = get_db_correctness_route(_q_type, _q_tags)

    conceptual = ConceptualScores(
        factual_correctness=factual_correctness,
        completeness=completeness,
        absence_of_contradiction=absence_of_contradiction,
        topic_specificity=topic_specificity,
    )

    # Gate SQL scorer on both the harness config AND whether SQL was actually
    # found and executed.  When a model gives a conceptual answer to a SQL-tagged
    # question (e.g. an explanation with no SQL code), _sql_was_executed is False
    # and we fall back to the conceptual scorer so the answer is not penalised
    # by zero syntactic_parse_success and result_set_f1 scores.
    _apply_sql_scorer = needs_sql_harness(_q_type, _q_tags) and _sql_was_executed

    # Populate SQL-specific dimension scores for practical sql-tagged questions.
    if _apply_sql_scorer:
        sql_scores.clause_appropriateness = norm_score
        sql_scores.constraint_correctness = norm_score * 0.9

    # Schema scorer: only relevant for practical sql questions (DDL/schema design).
    schema = SchemaScores(
        entity_coverage=norm_score,
        fk_correctness=norm_score * 0.9,
        normalization_compliance=norm_score * 0.85,
        index_appropriateness=norm_score * 0.8,
    ) if _apply_sql_scorer else None

    # Query-optimisation scorer: activated when EXPLAIN context is available.
    query_opt = QueryOptScores(
        join_algorithm_selection=norm_score,
        index_selectivity_reasoning=norm_score * 0.9,
        cost_estimation_accuracy=norm_score * 0.85,
        plan_tree_correctness=norm_score * 0.8,
        optimization_hint_usage=norm_score * 0.75,
    ) if ("sql" in _q_tags and bool(db_execution_context)) else None

    # Transaction scorer: activated for questions tagged 'transactions'.
    transaction = TransactionScores(
        auto_score=norm_score,
        judge_score=norm_score,
    ) if "transactions" in _q_tags else None

    db_bundle = DBCorrectnessBundle(
        sql=sql_scores if _apply_sql_scorer else None,
        conceptual=conceptual,
        schema=schema,
        query_opt=query_opt,
        transaction=transaction,
    )
    # Use the tag-derived routing key so the right sub-scorer is selected.
    db_correctness_score = db_bundle.pillar_score(_db_route)

    # ── Step 6: Build LLM quality pillar ──
    hallucination_rate = min(1.0, hallucination_metrics["severity_score"])
    reasoning = ReasoningScores(
        logical_coherence=norm_score,
        step_completeness=completeness,
        error_propagation=1.0 - (0.3 if hallucination_metrics["has_hallucination"] else 0.0),
        self_consistency=norm_score,
    )
    answer_words = len(answer_text.split()) if answer_text else 0
    expected_words = len(question.expected_answer.split()) if question.expected_answer else 100
    verbosity_ratio = answer_words / max(expected_words, 1)

    llm_bundle = LLMQualityBundle(
        hallucination_rate=hallucination_rate,
        hallucination_severity_score=hallucination_metrics["severity_score"],
        reasoning=reasoning,
        text_precision=norm_score * 0.95,
        text_recall=completeness,
        text_f1=norm_score,
        verbosity_ratio=verbosity_ratio,
        format_compliance_score=format_result["composite_score"],
        consistency_score=norm_score * 0.9,
    )
    llm_quality_score = llm_bundle.pillar_score()

    # ── Step 7: Efficiency — computed from actual run metrics (Section 2.7) ──
    efficiency_bundle = EfficiencyBundle(
        ttft_ms=run.ttft_ms or 0.0,
        total_latency_ms=run.total_latency_ms or 0.0,
        tokens_per_second=run.tokens_per_second or 0.0,
        cost_usd=run.cost_usd or 0.0,
        output_tokens=run.output_tokens or 0,
        is_correct=(score_0_10 >= 7.0),
    )
    efficiency_score = efficiency_bundle.pillar_score() if run.total_latency_ms else 50.0

    # ── Step 7b: Prompting Effectiveness (Section 4.4) ────────────────────────
    # Compare this run's MCS against the zero-shot baseline for the same model+question.
    # This gives a real accuracy_lift signal rather than a constant 50.
    prompting_effectiveness = 50.0  # neutral default until zero-shot baseline exists
    if run.prompt_strategy != "zero-shot":
        zs_result = await session.execute(
            select(JudgeScore)
            .join(EvaluationRun, JudgeScore.run_id == EvaluationRun.id)
            .where(
                EvaluationRun.question_id == run.question_id,
                EvaluationRun.model_id == run.model_id,
                EvaluationRun.prompt_strategy == "zero-shot",
                EvaluationRun.status == "completed",
            )
        )
        zs_score = zs_result.scalar_one_or_none()
        if zs_score and zs_score.master_composite_score is not None:
            zs_mcs = float(zs_score.master_composite_score)
            # Normalise lift to 0–100: 0 lift = 50, +30 pts lift = ~100, -30 pts = ~0
            current_mcs = db_correctness_score * 0.5 + llm_quality_score * 0.3 + efficiency_score * 0.05
            lift = current_mcs - zs_mcs
            prompting_effectiveness = max(0.0, min(100.0, 50.0 + lift * (50.0 / 30.0)))

    # ── Step 8: Compute MCS ──
    mcs = compute_mcs(
        db_correctness=db_correctness_score,
        llm_quality=llm_quality_score,
        prompting_effectiveness=prompting_effectiveness,
        efficiency=efficiency_score,
    )

    # ── Persist JudgeScore ──
    score_obj = existing_score or JudgeScore(run_id=run_id)

    score_obj.raw_judge_response = json.dumps(judge_result)
    score_obj.judge_score_0_10 = score_0_10
    _just = judge_result.get("justification", "")
    if _judge_failed:
        _just = "[FALLBACK-SCORED] " + _just
    score_obj.justification = _just
    score_obj.hallucinations_detected = all_hallucinations
    score_obj.missing_points = judge_result.get("missing_points", [])

    # SQL sub-scores
    score_obj.syntactic_parse_success = sql_scores.syntactic_parse_success
    score_obj.result_set_f1 = sql_scores.result_set_f1
    score_obj.clause_appropriateness = sql_scores.clause_appropriateness
    score_obj.constraint_correctness = sql_scores.constraint_correctness
    score_obj.idiomatic_postgresql = sql_scores.idiomatic_postgresql
    score_obj.sql_correctness_score = sql_scores.weighted() * 100
    score_obj.db_execution_context = db_execution_context   # EXPLAIN/error output injected into Judge
    score_obj.sql_harness_ran = needs_sql_harness(_q_type, _q_tags)

    # Conceptual sub-scores
    score_obj.factual_correctness = factual_correctness
    score_obj.completeness = completeness
    score_obj.absence_of_contradiction = absence_of_contradiction
    score_obj.topic_specificity = topic_specificity
    score_obj.conceptual_accuracy_score = conceptual.weighted() * 100

    # LLM quality sub-scores
    score_obj.hallucination_rate = hallucination_rate
    score_obj.hallucination_severity_score = hallucination_metrics["severity_score"]
    score_obj.reasoning_quality_score = reasoning.weighted() * 100
    score_obj.logical_coherence = reasoning.logical_coherence
    score_obj.step_completeness = reasoning.step_completeness
    score_obj.self_consistency = reasoning.self_consistency
    score_obj.text_f1 = norm_score
    score_obj.verbosity_ratio = verbosity_ratio
    score_obj.format_compliance_score = format_result["composite_score"]

    # Pillar scores
    score_obj.db_correctness_score = db_correctness_score
    score_obj.llm_quality_score = llm_quality_score
    score_obj.prompting_effectiveness_score = prompting_effectiveness
    score_obj.efficiency_score = efficiency_score
    score_obj.master_composite_score = mcs

    if not existing_score:
        session.add(score_obj)
    await session.flush()

    # ── Step 9: Refresh leaderboard aggregates (Section 5.5) ──
    await _refresh_leaderboard_for_model(session, run.model_id)

    return score_obj


async def _refresh_leaderboard_for_model(
    session: AsyncSession,
    db_model_id: str,
) -> None:
    """
    Recompute and upsert the leaderboard row for a single model.

    Aggregates mcs_score, db_correctness, llm_quality, hallucination_rate,
    avg_latency_ms, best_topic, worst_topic, and best_prompt_strategy from
    all completed judge_scores for this model (Section 5.5).
    """
    from sqlalchemy import func as sfunc

    # ── Overall averages ──────────────────────────────────────────────────────
    agg_result = await session.execute(
        select(
            sfunc.avg(JudgeScore.master_composite_score).label("avg_mcs"),
            sfunc.avg(JudgeScore.db_correctness_score).label("avg_db"),
            sfunc.avg(JudgeScore.llm_quality_score).label("avg_llm"),
            sfunc.avg(JudgeScore.prompting_effectiveness_score).label("avg_prompting"),
            sfunc.avg(JudgeScore.efficiency_score).label("avg_efficiency"),
            sfunc.avg(JudgeScore.hallucination_rate).label("avg_hr"),
            sfunc.avg(EvaluationRun.total_latency_ms).label("avg_lat"),
        )
        .join(EvaluationRun, JudgeScore.run_id == EvaluationRun.id)
        .where(
            EvaluationRun.model_id == db_model_id,
            EvaluationRun.status == "completed",
        )
    )
    agg = agg_result.one_or_none()
    if agg is None or agg.avg_mcs is None:
        return

    # ── Best / worst topic (subtopic granularity) ─────────────────────────────
    topic_stmt = (
        select(
            Subtopic.name.label("subtopic_name"),
            sfunc.avg(JudgeScore.master_composite_score).label("avg_mcs"),
        )
        .join(EvaluationRun, JudgeScore.run_id == EvaluationRun.id)
        .join(Question, EvaluationRun.question_id == Question.id)
        .join(Subtopic, Question.subtopic_id == Subtopic.id)
        .where(
            EvaluationRun.model_id == db_model_id,
            EvaluationRun.status == "completed",
        )
        .group_by(Subtopic.name)
    )
    topic_result = await session.execute(topic_stmt)
    topic_rows = topic_result.all()

    best_topic = max(topic_rows, key=lambda r: r.avg_mcs or 0.0, default=None)
    worst_topic = min(topic_rows, key=lambda r: r.avg_mcs or 0.0, default=None)

    # ── Best prompt strategy ──────────────────────────────────────────────────
    strategy_stmt = (
        select(
            EvaluationRun.prompt_strategy,
            sfunc.avg(JudgeScore.master_composite_score).label("avg_mcs"),
        )
        .join(JudgeScore, JudgeScore.run_id == EvaluationRun.id)
        .where(
            EvaluationRun.model_id == db_model_id,
            EvaluationRun.status == "completed",
        )
        .group_by(EvaluationRun.prompt_strategy)
    )
    strategy_result = await session.execute(strategy_stmt)
    strategy_rows = strategy_result.all()
    best_strategy = max(strategy_rows, key=lambda r: r.avg_mcs or 0.0, default=None)

    # ── Upsert leaderboard row ────────────────────────────────────────────────
    from models.db_models import Leaderboard
    lb_result = await session.execute(
        select(Leaderboard).where(Leaderboard.model_id == db_model_id)
    )
    lb = lb_result.scalar_one_or_none()
    if lb is None:
        from config import ELO_INITIAL_RATING
        lb = Leaderboard(model_id=db_model_id, elo_rating=ELO_INITIAL_RATING)
        session.add(lb)

    lb.mcs_score = round(float(agg.avg_mcs), 2)
    lb.db_correctness = round(float(agg.avg_db or 0.0), 2)
    lb.llm_quality = round(float(agg.avg_llm or 0.0), 2)
    lb.prompting_effectiveness = round(float(agg.avg_prompting or 50.0), 2)
    lb.efficiency_score = round(float(agg.avg_efficiency or 50.0), 2)
    lb.hallucination_rate = round(float(agg.avg_hr or 0.0), 4)
    lb.avg_latency_ms = round(float(agg.avg_lat or 0.0), 1)
    lb.best_topic = best_topic.subtopic_name if best_topic else None
    lb.worst_topic = worst_topic.subtopic_name if worst_topic else None
    lb.best_prompt_strategy = best_strategy.prompt_strategy if best_strategy else None
    # ── Contest stats ─────────────────────────────────────────────
    from sqlalchemy import func as sfunc
    from models.db_models import ContestResult

    # ── Contest stats ─────────────────────────────────────────────
    from sqlalchemy import func as sfunc
    from models.db_models import ContestResult

    # Convert UUID to string (important for JSON comparison)
    model_id_str = str(db_model_id)
    from sqlalchemy import cast
    from sqlalchemy.dialects.postgresql import JSONB


    # Total contests where model participated
    contest_total_result = await session.execute(
        select(sfunc.count(ContestResult.id))
        .where(
            cast(ContestResult.ranked_model_ids, JSONB).op("@>")(
                cast([model_id_str], JSONB)
            )
        )
    )
    contest_total = contest_total_result.scalar() or 0
    from sqlalchemy import cast, String
    # Wins = first element in array
    contest_wins_result = await session.execute(
        select(sfunc.count(ContestResult.id))
        .where(
            cast(ContestResult.ranked_model_ids, JSONB).op("->>")(0) == model_id_str
        )
    )
    contest_wins = contest_wins_result.scalar() or 0

    lb.contest_total = contest_total
    lb.contest_wins = contest_wins
    lb.win_rate = (
        contest_wins / contest_total if contest_total > 0 else None
    )
    await session.flush()

async def run_contest(
    session: AsyncSession,
    question_id: str,
    run_ids: Optional[list[str]] = None,
    answers_map: Optional[dict[str, str]] = None,
) -> dict:
    """
    Run the Judge LLM pairwise contest for a single question.

    If run_ids are provided, answers are fetched from the DB.
    If answers_map is provided (model_id → answer), it is used directly.
    Elo ratings are updated after the contest.

    Returns a dict containing contest metadata and Elo updates.
    """
    q_result = await session.execute(
        select(Question).where(Question.id == question_id)
    )
    question = q_result.scalar_one_or_none()
    if question is None:
        raise ValueError(f"Question not found: {question_id}")

    # Build answers map from run IDs if not provided directly
    if answers_map is None:
        answers_map = {}
        if run_ids:
            for run_id in run_ids:
                run_result = await session.execute(
                    select(EvaluationRun).where(EvaluationRun.id == run_id)
                )
                run = run_result.scalar_one_or_none()
                if run:
                    model_result = await session.execute(
                        select(LLMModel).where(LLMModel.id == run.model_id)
                    )
                    db_model = model_result.scalar_one_or_none()
                    if db_model:
                        answers_map[db_model.model_id] = run.model_answer or ""

    if len(answers_map) < 2:
        raise ValueError("At least two model answers are required for a contest")

    # ── Call Judge LLM for contest ──
    contest_result = await judge_contest(
        question=question.question_text,
        ground_truth=question.expected_answer or "",
        answers_map=answers_map,
    )

    # ── Persist ContestResult ──
    # Resolve model UUIDs from model_ids for ranked_model_ids
    ranked_db_ids = []
    for mid in contest_result["ranked_model_ids"]:
        m_result = await session.execute(
            select(LLMModel).where(LLMModel.model_id == mid)
        )
        m = m_result.scalar_one_or_none()
        ranked_db_ids.append(m.id if m else mid)

    contest_row = ContestResult(
        question_id=question_id,
        ranked_model_ids=ranked_db_ids,
        anonymized_map=contest_result["anonymized_map"],
        judge_reasoning=contest_result["reasoning"],
        tie_exists=contest_result["tie_exists"],
        tie_model_ids=[
            item for group in contest_result["ties"] for item in group
        ] if contest_result["ties"] else None,
        raw_judge_response=contest_result["raw_judge_response"],
    )
    session.add(contest_row)
    await session.flush()

    # ── Update Elo ratings ──
    ranked_placements = []
    for ranking in contest_result["rankings"]:
        m_result = await session.execute(
            select(LLMModel).where(LLMModel.model_id == ranking["model_id"])
        )
        m = m_result.scalar_one_or_none()
        if m:
            ranked_placements.append({
                "model_id": m.id,
                "placement": ranking["placement"],
            })

    elo_updates = await update_elo_ratings(
        session,
        contest_result_id=contest_row.id,
        ranked_placements=ranked_placements,
    )

    return {
        "contest_id": contest_row.id,
        "question_id": question_id,
        "ranked_model_ids": contest_result["ranked_model_ids"],
        "rankings": contest_result["rankings"],
        "reasoning": contest_result["reasoning"],
        "tie_exists": contest_result["tie_exists"],
        "elo_updates": elo_updates,
    }


# ── Self-Consistency Runner ───────────────────────────────────────────────────
async def run_self_consistency(
    session: AsyncSession,
    model_id: str,
    question_id: str,
    base_hyperparams: Optional[dict] = None,
    k: int = SELF_CONSISTENCY_K,
) -> dict:
    """
    Sample k responses at temperature=0.7 and aggregate via plurality vote
    (for SQL) or Judge LLM synthesis (for conceptual questions).

    Section 4.2.4: Self-consistency is most effective for Hard difficulty
    questions or those showing historically high variance.

    Returns the best answer and the individual run IDs.
    """
    hp = {**DEFAULT_HYPERPARAMS, **(base_hyperparams or {})}
    hp["temperature"] = SELF_CONSISTENCY_TEMPERATURE

    q_result = await session.execute(
        select(Question).where(Question.id == question_id)
    )
    question = q_result.scalar_one_or_none()
    if question is None:
        raise ValueError(f"Question not found: {question_id}")

    # Dispatch k parallel generation requests.
    # Each sub-run is labelled "self-consistency" so they appear correctly in
    # results_summary and prompts/compare reports. Using force_rerun=True with
    # different seeds ensures k distinct samples even for deterministic models.
    tasks = [
        run_generation(
            session=session,
            model_id=model_id,
            question_id=question_id,
            prompt_strategy="self-consistency",
            hyperparams={**hp, "seed": 42 + i},
            force_rerun=True,
        )
        for i in range(k)
    ]
    runs = await asyncio.gather(*tasks, return_exceptions=True)
    valid_runs = [r for r in runs if isinstance(r, EvaluationRun) and r.status == "completed"]

    if not valid_runs:
        raise RuntimeError(f"All {k} self-consistency runs failed for question {question_id}")

    answers = [r.model_answer for r in valid_runs if r.model_answer]

    # Plurality vote: select the answer that appears most frequently (by token overlap)
    best_answer = _plurality_vote(answers)

    return {
        "best_answer": best_answer,
        "run_ids": [r.id for r in valid_runs],
        "k_attempted": k,
        "k_successful": len(valid_runs),
        "question_type": question.question_type,
    }


def _plurality_vote(answers: list[str]) -> str:
    """
    Select the answer with the highest average similarity to all other answers.
    This is the soft plurality vote — the answer closest to the 'consensus'.
    """
    if not answers:
        return ""
    if len(answers) == 1:
        return answers[0]

    from evaluators.robustness import compute_token_overlap_similarity

    best_idx, best_score = 0, -1.0
    for i, a in enumerate(answers):
        others = [answers[j] for j in range(len(answers)) if j != i]
        avg_sim = sum(compute_token_overlap_similarity(a, o) for o in others) / len(others)
        if avg_sim > best_score:
            best_score = avg_sim
            best_idx = i

    return answers[best_idx]


async def _try_explain_analyze(sql_answer: str, schema_fixture: str | None = None) -> str | None:
    """
    Attempt to run EXPLAIN ANALYZE on the SQL extracted from a model answer,
    optionally within a schema fixture context.

    Returns the EXPLAIN ANALYZE plan as a formatted string for injection into
    the Judge prompt, or None if the test DB is unavailable or the SQL cannot
    be parsed as a SELECT statement.

    Only SELECT statements are EXPLAIN-ANALYZE'd (DDL/DML are skipped to avoid
    side effects even inside a transaction).
    """
    from evaluators.sql_harness import _extract_sql_from_text, _get_test_connection, _split_sql_statements
    import re as _re

    extracted = _extract_sql_from_text(sql_answer or "")
    if not extracted:
        return None

    # Only EXPLAIN SELECT-like statements
    if not _re.match(r"\s*(SELECT|WITH)\b", extracted, _re.IGNORECASE):
        return None

    conn = await _get_test_connection()
    if conn is None:
        return None

    try:
        async with conn.transaction():
            # Apply schema if provided
            if schema_fixture:
                for stmt in _split_sql_statements(schema_fixture):
                    if stmt.strip():
                        try:
                            await conn.execute(stmt)
                        except Exception:
                            pass  # Fixture parts may already exist

            try:
                rows = await conn.fetch(f"EXPLAIN ANALYZE {extracted}")
                plan_lines = [r[0] for r in rows]
                return "EXPLAIN ANALYZE output:\n" + "\n".join(plan_lines)
            except Exception as e:
                # Return just the error so the Judge knows execution failed
                return f"EXPLAIN ANALYZE failed: {e!s}"
            # Transaction rolls back automatically — no side effects
    except Exception:
        return None
    finally:
        await conn.close()

