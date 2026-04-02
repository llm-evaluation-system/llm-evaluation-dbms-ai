"""
models/db_models.py — SQLAlchemy ORM table definitions.

Covers every table referenced in the specification:
  questions, topics, models, evaluation_runs, judge_scores,
  contest_results, leaderboard, hyperparameter_runs,
  prompt_strategy_runs, perturbation_results, elo_history,
  few_shot_examples, hallucination_records, latency_records.
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer,
    String, Text, JSON, UniqueConstraint, Index,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from database import Base


def new_uuid() -> str:
    return str(uuid.uuid4())


# ── Topics / Subtopics ────────────────────────────────────────────────────────
class Topic(Base):
    """Top-level DBMS topic (e.g. FOUNDATIONS, TRANSACTION MANAGEMENT)."""
    __tablename__ = "topics"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    name = Column(String(128), nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    subtopics = relationship("Subtopic", back_populates="topic", cascade="all, delete-orphan")


class Subtopic(Base):
    """A specific subtopic within a topic (e.g. INTRODUCTION TO DATABASE DESIGN)."""
    __tablename__ = "subtopics"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    topic_id = Column(UUID(as_uuid=False), ForeignKey("topics.id"), nullable=False)
    name = Column(String(256), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    topic = relationship("Topic", back_populates="subtopics")
    questions = relationship("Question", back_populates="subtopic", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint("topic_id", "name", name="uq_subtopic_topic_name"),
    )


# ── Question Bank ─────────────────────────────────────────────────────────────
class Question(Base):
    """
    A single question from the DBMS textbook question bank.

    question_type is one of exactly two values after the question-bank
    correction:
      - 'conceptual' : explanation / definition / analysis — no executable
                       SQL artifact expected from the challenger.
      - 'practical'  : the challenger must produce a concrete artifact
                       (SQL DDL/DML, RA expressions, algorithm traces, etc.).

    Content-specific routing (SQL harness, SQL rubric, format checks) is
    driven by the `tags` column, NOT by question_type. Use the helpers in
    question_bank.sql_fixtures (needs_sql_harness, needs_sql_rubric, …)
    for all routing decisions.
    """
    __tablename__ = "questions"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    subtopic_id = Column(UUID(as_uuid=False), ForeignKey("subtopics.id"), nullable=False)
    exercise_number = Column(String(32), nullable=True)   # e.g. "Exercise 5.1"
    question_text = Column(Text, nullable=False)
    expected_answer = Column(Text, nullable=True)
    question_type = Column(
        String(32), nullable=False, default="conceptual"
    )  # conceptual | practical  (content routing is via tags, not this field)
    difficulty = Column(String(16), nullable=False, default="medium")  # easy | medium | hard
    schema_fixture = Column(Text, nullable=True)   # DDL for the test schema (SQL questions)
    expected_rows = Column(JSON, nullable=True)    # Expected result set as JSON array
    tags = Column(JSON, nullable=True)             # List of keyword tags
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    subtopic = relationship("Subtopic", back_populates="questions")
    runs = relationship("EvaluationRun", back_populates="question", cascade="all, delete-orphan")
    few_shot_examples = relationship("FewShotExample", back_populates="question")

    __table_args__ = (
        Index("ix_questions_subtopic_id", "subtopic_id"),
        Index("ix_questions_question_type", "question_type"),
        Index("ix_questions_difficulty", "difficulty"),
    )


# ── Model Registry ────────────────────────────────────────────────────────────
class LLMModel(Base):
    """
    Registry of all LLM models participating in the evaluation:
    four challenger models + one judge model.
    """
    __tablename__ = "llm_models"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    model_id = Column(String(64), nullable=False, unique=True)   # logical key used in API calls
    display_name = Column(String(128), nullable=False)
    provider = Column(String(64), nullable=False)                 # openai | anthropic | google | groq
    api_model = Column(String(128), nullable=False)               # exact string passed to the API
    is_judge = Column(Boolean, nullable=False, default=False)
    cost_per_1k_input_tokens = Column(Float, nullable=True)
    cost_per_1k_output_tokens = Column(Float, nullable=True)
    max_context_tokens = Column(Integer, nullable=True)
    supports_seed = Column(Boolean, nullable=False, default=False)
    model_version = Column(String(64), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    runs = relationship("EvaluationRun", back_populates="model", cascade="all, delete-orphan")
    leaderboard = relationship("Leaderboard", back_populates="model", uselist=False)
    elo_history = relationship("EloHistory", back_populates="model", cascade="all, delete-orphan")


# ── Evaluation Runs ───────────────────────────────────────────────────────────
class EvaluationRun(Base):
    """
    One generation run: a single model answering a single question under a
    specific prompt strategy and hyperparameter configuration.

    The composite key (question_id, model_id, prompt_strategy, hyperparam_hash)
    is unique; re-running the same configuration overwrites the existing record.
    """
    __tablename__ = "evaluation_runs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    question_id = Column(UUID(as_uuid=False), ForeignKey("questions.id"), nullable=False)
    model_id = Column(UUID(as_uuid=False), ForeignKey("llm_models.id"), nullable=False)
    prompt_strategy = Column(String(64), nullable=False)   # zero-shot | few-shot | cot | …
    hyperparam_hash = Column(String(64), nullable=False)   # SHA-256 of serialised hyperparams
    hyperparams = Column(JSON, nullable=False)             # Full hyperparam snapshot
    model_answer = Column(Text, nullable=True)             # Raw text response
    prompt_used = Column(Text, nullable=True)              # Full prompt sent to model
    system_prompt = Column(Text, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    cost_usd = Column(Float, nullable=True)
    ttft_ms = Column(Float, nullable=True)                 # Time-to-first-token (ms)
    total_latency_ms = Column(Float, nullable=True)
    tokens_per_second = Column(Float, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    eval_version = Column(String(32), nullable=False, default="v1")
    run_timestamp = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String(32), nullable=False, default="pending")  # pending | completed | failed | error
    error_message = Column(Text, nullable=True)

    question = relationship("Question", back_populates="runs")
    model = relationship("LLMModel", back_populates="runs")
    judge_score = relationship("JudgeScore", back_populates="run", uselist=False, cascade="all, delete-orphan")
    hallucination_records = relationship("HallucinationRecord", back_populates="run", cascade="all, delete-orphan")

    __table_args__ = (
        UniqueConstraint(
            "question_id", "model_id", "prompt_strategy", "hyperparam_hash",
            name="uq_run_composite_key"
        ),
        Index("ix_eval_runs_model_id", "model_id"),
        Index("ix_eval_runs_question_id", "question_id"),
        Index("ix_eval_runs_prompt_strategy", "prompt_strategy"),
        Index("ix_eval_runs_status", "status"),
    )


# ── Judge Scores ──────────────────────────────────────────────────────────────
class JudgeScore(Base):
    """
    Absolute score assigned by the Judge LLM to a single evaluation run.
    Covers both DB-correctness dimensions and LLM-quality dimensions.
    """
    __tablename__ = "judge_scores"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    run_id = Column(UUID(as_uuid=False), ForeignKey("evaluation_runs.id"), nullable=False, unique=True)

    # ── Raw judge output ──
    raw_judge_response = Column(Text, nullable=True)
    judge_score_0_10 = Column(Float, nullable=True)       # Overall 0–10 score from judge
    justification = Column(Text, nullable=True)
    hallucinations_detected = Column(JSON, nullable=True)  # List of hallucinated items
    missing_points = Column(JSON, nullable=True)

    # ── SQL / DB Correctness (Section 1) ──
    syntactic_parse_success = Column(Float, nullable=True)
    result_set_f1 = Column(Float, nullable=True)
    clause_appropriateness = Column(Float, nullable=True)
    constraint_correctness = Column(Float, nullable=True)
    idiomatic_postgresql = Column(Float, nullable=True)
    sql_correctness_score = Column(Float, nullable=True)   # Weighted composite
    db_execution_context = Column(Text, nullable=True)     # EXPLAIN/error output injected into Judge
    sql_harness_ran = Column(Boolean, nullable=False, default=False)  # True when harness ran

    # ── Conceptual Accuracy ──
    factual_correctness = Column(Float, nullable=True)
    completeness = Column(Float, nullable=True)
    absence_of_contradiction = Column(Float, nullable=True)
    topic_specificity = Column(Float, nullable=True)
    conceptual_accuracy_score = Column(Float, nullable=True)

    # ── Schema Design ──
    entity_coverage = Column(Float, nullable=True)
    fk_correctness = Column(Float, nullable=True)
    normalization_compliance = Column(Float, nullable=True)
    index_appropriateness = Column(Float, nullable=True)
    schema_design_score = Column(Float, nullable=True)

    # ── Query Plan / Optimization ──
    join_algorithm_selection = Column(Float, nullable=True)
    index_selectivity_reasoning = Column(Float, nullable=True)
    cost_estimation_accuracy = Column(Float, nullable=True)
    plan_tree_correctness = Column(Float, nullable=True)
    optimization_hint_usage = Column(Float, nullable=True)
    query_optimization_score = Column(Float, nullable=True)

    # ── Transaction / Concurrency ──
    serializability_correctness = Column(Float, nullable=True)
    deadlock_detection_score = Column(Float, nullable=True)
    isolation_level_score = Column(Float, nullable=True)
    aries_trace_score = Column(Float, nullable=True)
    transaction_score = Column(Float, nullable=True)

    # ── LLM Quality (Section 2) ──
    hallucination_rate = Column(Float, nullable=True)
    hallucination_severity_score = Column(Float, nullable=True)
    reasoning_quality_score = Column(Float, nullable=True)
    logical_coherence = Column(Float, nullable=True)
    step_completeness = Column(Float, nullable=True)
    error_propagation = Column(Float, nullable=True)
    self_consistency = Column(Float, nullable=True)
    text_precision = Column(Float, nullable=True)
    text_recall = Column(Float, nullable=True)
    text_f1 = Column(Float, nullable=True)
    verbosity_ratio = Column(Float, nullable=True)
    format_compliance_score = Column(Float, nullable=True)

    # ── Composite Pillar Scores ──
    db_correctness_score = Column(Float, nullable=True)      # 0–100
    llm_quality_score = Column(Float, nullable=True)         # 0–100
    prompting_effectiveness_score = Column(Float, nullable=True)  # 0–100
    efficiency_score = Column(Float, nullable=True)          # 0–100
    master_composite_score = Column(Float, nullable=True)    # MCS 0–100

    scored_at = Column(DateTime(timezone=True), server_default=func.now())

    run = relationship("EvaluationRun", back_populates="judge_score")

    __table_args__ = (
        Index("ix_judge_scores_run_id", "run_id"),
    )


# ── Contest Results ────────────────────────────────────────────────────────────
class ContestResult(Base):
    """
    Pairwise / tournament contest result for a single question.
    All four model answers are judged simultaneously; this table captures
    the judge's full ranking and per-model placements.
    """
    __tablename__ = "contest_results"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    question_id = Column(UUID(as_uuid=False), ForeignKey("questions.id"), nullable=False)
    ranked_model_ids = Column(JSON, nullable=False)   # [1st, 2nd, 3rd, 4th] model UUIDs
    anonymized_map = Column(JSON, nullable=False)     # {"A": model_uuid, "B": …}
    judge_reasoning = Column(Text, nullable=True)
    tie_exists = Column(Boolean, nullable=False, default=False)
    tie_model_ids = Column(JSON, nullable=True)       # Model UUIDs that tied
    raw_judge_response = Column(Text, nullable=True)
    contest_timestamp = Column(DateTime(timezone=True), server_default=func.now())

    question = relationship("Question")

    __table_args__ = (
        Index("ix_contest_results_question_id", "question_id"),
    )


# ── Leaderboard ───────────────────────────────────────────────────────────────
class Leaderboard(Base):
    """
    Aggregate leaderboard row per model — updated after each contest and run.
    Mirrors the schema defined in Section 5.5 of the specification.
    """
    __tablename__ = "leaderboard"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    model_id = Column(UUID(as_uuid=False), ForeignKey("llm_models.id"), nullable=False, unique=True)
    mcs_score = Column(Float, nullable=True)
    db_correctness = Column(Float, nullable=True)
    llm_quality = Column(Float, nullable=True)
    prompting_effectiveness = Column(Float, nullable=True)
    efficiency_score = Column(Float, nullable=True)
    elo_rating = Column(Integer, nullable=False, default=1200)
    contest_wins = Column(Integer, nullable=False, default=0)
    contest_total = Column(Integer, nullable=False, default=0)
    win_rate = Column(Float, nullable=True)
    best_prompt_strategy = Column(String(64), nullable=True)
    best_topic = Column(String(256), nullable=True)
    worst_topic = Column(String(256), nullable=True)
    avg_latency_ms = Column(Float, nullable=True)
    hallucination_rate = Column(Float, nullable=True)
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    model = relationship("LLMModel", back_populates="leaderboard")

    __table_args__ = (
        Index("ix_leaderboard_elo_rating", "elo_rating"),
        Index("ix_leaderboard_mcs_score", "mcs_score"),
    )


# ── Elo History ───────────────────────────────────────────────────────────────
class EloHistory(Base):
    """Time-series record of Elo rating changes per model per contest."""
    __tablename__ = "elo_history"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    model_id = Column(UUID(as_uuid=False), ForeignKey("llm_models.id"), nullable=False)
    contest_result_id = Column(UUID(as_uuid=False), ForeignKey("contest_results.id"), nullable=True)
    old_rating = Column(Integer, nullable=False)
    new_rating = Column(Integer, nullable=False)
    delta = Column(Integer, nullable=False)
    placement = Column(Integer, nullable=True)   # 1st / 2nd / 3rd / 4th
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())

    model = relationship("LLMModel", back_populates="elo_history")
    contest_result = relationship("ContestResult")

    __table_args__ = (
        Index("ix_elo_history_model_id", "model_id"),
    )


# ── Hallucination Records ─────────────────────────────────────────────────────
class HallucinationRecord(Base):
    """
    Individual hallucination instance detected within an evaluation run.
    Severity tiers: critical | high | medium | low.
    """
    __tablename__ = "hallucination_records"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    run_id = Column(UUID(as_uuid=False), ForeignKey("evaluation_runs.id"), nullable=False)
    hallucination_type = Column(String(64), nullable=False)  # fabricated_function | wrong_attribution | …
    description = Column(Text, nullable=False)
    severity = Column(String(16), nullable=False)             # critical | high | medium | low
    detected_by = Column(String(32), nullable=False)          # regex | pg_catalog | llm_judge | ground_truth
    detected_at = Column(DateTime(timezone=True), server_default=func.now())

    run = relationship("EvaluationRun", back_populates="hallucination_records")

    __table_args__ = (
        Index("ix_hallucination_run_id", "run_id"),
        Index("ix_hallucination_severity", "severity"),
    )


# ── Latency Records ───────────────────────────────────────────────────────────
class LatencyRecord(Base):
    """Per-request latency and throughput telemetry."""
    __tablename__ = "latency_records"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    run_id = Column(UUID(as_uuid=False), ForeignKey("evaluation_runs.id"), nullable=False, unique=True)
    ttft_ms = Column(Float, nullable=True)
    total_latency_ms = Column(Float, nullable=False)
    tokens_per_second = Column(Float, nullable=True)
    input_tokens = Column(Integer, nullable=True)
    output_tokens = Column(Integer, nullable=True)
    api_retry_count = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Float, nullable=True)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_latency_run_id", "run_id"),
    )


# ── Few-Shot Example Store ────────────────────────────────────────────────────
class FewShotExample(Base):
    """
    Pre-computed few-shot example bank.
    Examples are stratified by subtopic and difficulty and referenced by the
    prompt builder.  A leakage guard ensures a question cannot appear as its
    own few-shot example.
    """
    __tablename__ = "few_shot_examples"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    question_id = Column(UUID(as_uuid=False), ForeignKey("questions.id"), nullable=False)
    subtopic_id = Column(UUID(as_uuid=False), ForeignKey("subtopics.id"), nullable=False)
    difficulty = Column(String(16), nullable=False)
    prompt_strategy = Column(String(64), nullable=False)   # few-shot | cot-few-shot
    example_question = Column(Text, nullable=False)
    example_answer = Column(Text, nullable=False)          # For CoT: includes reasoning steps
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    question = relationship("Question", back_populates="few_shot_examples")

    __table_args__ = (
        Index("ix_few_shot_subtopic_difficulty", "subtopic_id", "difficulty"),
    )


# ── Perturbation Results ──────────────────────────────────────────────────────
class PerturbationResult(Base):
    """
    Robustness benchmark: similarity between answers to original and perturbed
    questions.  Cosine similarity of sentence embeddings is stored here.
    """
    __tablename__ = "perturbation_results"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    original_run_id = Column(UUID(as_uuid=False), ForeignKey("evaluation_runs.id"), nullable=False)
    perturbed_run_id = Column(UUID(as_uuid=False), ForeignKey("evaluation_runs.id"), nullable=False)
    perturbation_type = Column(String(64), nullable=False)
    perturbed_question = Column(Text, nullable=False)
    cosine_similarity = Column(Float, nullable=True)
    consistency_pass = Column(Boolean, nullable=True)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        Index("ix_perturbation_original_run", "original_run_id"),
    )


# ── Hyperparameter Sweep Metadata ─────────────────────────────────────────────
class HyperparamSweep(Base):
    """
    Groups a set of evaluation runs under a single sweep experiment so that
    the /eval/hyperparams/compare endpoint can reconstruct the full grid.
    """
    __tablename__ = "hyperparam_sweeps"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    model_id = Column(UUID(as_uuid=False), ForeignKey("llm_models.id"), nullable=False)
    sweep_param = Column(String(64), nullable=False)     # Which parameter is being swept
    param_grid = Column(JSON, nullable=False)             # Full grid specification
    sample_size = Column(Integer, nullable=True)
    status = Column(String(32), nullable=False, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_hyperparam_sweep_model_id", "model_id"),
    )


# ── Prompt Strategy Comparison Metadata ──────────────────────────────────────
class PromptStrategyComparison(Base):
    """
    Groups a set of evaluation runs under a single prompt-strategy comparison
    experiment so the /eval/prompts/compare endpoint can aggregate results.
    """
    __tablename__ = "prompt_strategy_comparisons"

    id = Column(UUID(as_uuid=False), primary_key=True, default=new_uuid)
    model_id = Column(UUID(as_uuid=False), ForeignKey("llm_models.id"), nullable=False)
    strategies = Column(JSON, nullable=False)   # List of strategy names compared
    sample_size = Column(Integer, nullable=True)
    status = Column(String(32), nullable=False, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    __table_args__ = (
        Index("ix_prompt_comparison_model_id", "model_id"),
    )
