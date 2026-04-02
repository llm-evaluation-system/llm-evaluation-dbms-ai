"""
models/schemas.py — Pydantic v2 request / response schemas.

These are the data-transfer objects used by FastAPI endpoints.  Every field
carries explicit validation and a docstring so the auto-generated OpenAPI spec
is self-documenting.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# ── Common Helpers ────────────────────────────────────────────────────────────
class OKResponse(BaseModel):
    ok: bool = True
    message: str = "Success"


# ── Hyperparameters ───────────────────────────────────────────────────────────
class Hyperparams(BaseModel):
    """LLM inference hyperparameters for one generation request."""
    temperature: float = Field(0.3, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling threshold")
    max_tokens: int = Field(1024, ge=64, le=8192, description="Maximum output tokens")
    top_k: int = Field(-1, ge=-1, description="Top-K sampling (-1 = disabled)")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")
    system_prompt_style: str = Field(
        "expert-persona",
        description="minimal | role-based | expert-persona"
    )
    seed: Optional[int] = Field(None, description="Deterministic seed (model must support it)")

    @field_validator("system_prompt_style")
    @classmethod
    def validate_prompt_style(cls, v: str) -> str:
        allowed = {"minimal", "role-based", "expert-persona"}
        if v not in allowed:
            raise ValueError(f"system_prompt_style must be one of {allowed}")
        return v


# ── /eval/generate ────────────────────────────────────────────────────────────
class GenerateRequest(BaseModel):
    """POST /eval/generate — submit a question to a challenger model."""
    model_id: str = Field(..., description="Challenger model identifier (e.g. 'gpt-4o')")
    question_id: str = Field(..., description="UUID of the question to answer")
    prompt_strategy: str = Field(
        "zero-shot",
        description=(
            "zero-shot | one-shot | few-shot | cot | few-shot-cot | "
            "self-consistency | role-prompting | least-to-most | react"
        )
    )
    hyperparams: Hyperparams = Field(default_factory=Hyperparams)
    async_run: bool = Field(False, description="If True, enqueue as Celery task and return task_id")

    @field_validator("prompt_strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        allowed = {
            "zero-shot", "one-shot", "few-shot", "cot",
            "few-shot-cot", "self-consistency", "role-prompting",
            "least-to-most", "react",
        }
        if v not in allowed:
            raise ValueError(f"prompt_strategy must be one of {allowed}")
        return v


class GenerateResponse(BaseModel):
    run_id: str
    model_id: str
    question_id: str
    prompt_strategy: str
    status: str
    model_answer: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    total_latency_ms: Optional[float] = None
    task_id: Optional[str] = None   # Populated when async_run=True


# ── /eval/judge/score ─────────────────────────────────────────────────────────
class JudgeScoreRequest(BaseModel):
    """POST /eval/judge/score — judge scores a single model's answer."""
    model_id: str = Field(..., description="Challenger model identifier")
    question_id: str = Field(..., description="UUID of the question")
    run_id: Optional[str] = Field(
        None,
        description="If provided, judge scores the existing run; otherwise model_answer is required"
    )
    model_answer: Optional[str] = Field(None, description="The model's text answer (if run_id not given)")
    ground_truth: Optional[str] = Field(None, description="Override ground truth (defaults to DB value)")
    scoring_rubric: Optional[str] = Field(None, description="Custom rubric override")
    force_rescore: bool = Field(False, description="Overwrite existing judge score if present")


class SQLExecutionDetails(BaseModel):
    """
    Sub-scores from the automated SQL execution harness.
    Only populated for practical sql-tagged questions where the harness ran.
    All fields are None when the harness was not activated.
    """
    harness_ran: bool = False
    syntactic_parse_success: Optional[float] = None   # 0-1: did SQL parse in PG?
    result_set_f1: Optional[float] = None             # 0-1: F1 vs expected_rows
    idiomatic_postgresql: Optional[float] = None      # 0-1: PG idiom quality
    db_execution_context: Optional[str] = None        # EXPLAIN/error output sent to Judge


class JudgeScoreResponse(BaseModel):
    score_id: str
    run_id: str
    judge_score_0_10: float
    justification: str
    hallucinations_detected: list[dict]
    missing_points: list[str]
    db_correctness_score: float
    llm_quality_score: float
    master_composite_score: float
    scored_at: datetime
    prompting_effectiveness_score: float
    efficiency_score: float
    sql_execution_details: Optional[SQLExecutionDetails] = None  # populated for sql questions


# ── /eval/judge/contest ───────────────────────────────────────────────────────
class ContestRequest(BaseModel):
    """POST /eval/judge/contest — judge picks best answer among all models."""
    question_id: str = Field(..., description="UUID of the question")
    run_ids: Optional[list[str]] = Field(
        None,
        description="List of run UUIDs to contest (defaults to most recent run per model)"
    )
    answers_map: Optional[dict[str, str]] = Field(
        None,
        description="model_id → answer text (alternative to run_ids)"
    )
    ground_truth: Optional[str] = Field(None, description="Ground truth override")

    @model_validator(mode="after")
    def validate_inputs(self) -> "ContestRequest":
        if self.run_ids is None and self.answers_map is None:
            raise ValueError("Either run_ids or answers_map must be provided")
        return self


class ContestResponse(BaseModel):
    contest_id: str
    question_id: str
    ranked_model_ids: list[str]          # [1st-place model_id, …]
    ranking_with_scores: list[dict]       # [{model_id, placement, justification}]
    tie_exists: bool
    tie_model_ids: Optional[list[str]]
    judge_reasoning: str
    elo_updates: list[dict]              # [{model_id, old_rating, new_rating, delta}]


# ── /eval/results/summary ─────────────────────────────────────────────────────
class ResultsSummaryRequest(BaseModel):
    model_id: Optional[str] = None
    topic: Optional[str] = None
    subtopic: Optional[str] = None
    prompt_type: Optional[str] = None
    difficulty: Optional[str] = None
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)


class ModelTopicScore(BaseModel):
    model_id: str
    model_display_name: str
    subtopic: str
    prompt_strategy: str
    question_count: int
    avg_mcs: float
    avg_db_correctness: float
    avg_llm_quality: float
    avg_hallucination_rate: float
    avg_latency_ms: float


class ResultsSummaryResponse(BaseModel):
    total: int
    results: list[ModelTopicScore]
    generated_at: datetime


# ── /eval/hyperparams/compare ─────────────────────────────────────────────────
class HyperparamCompareRequest(BaseModel):
    """GET /eval/hyperparams/compare — compare one model across param grid."""
    model_id: str
    param_grid: dict[str, list[Any]] = Field(
        ...,
        description="Dict of hyperparam name → list of values to compare"
    )
    subtopic: Optional[str] = None
    sample_size: Optional[int] = Field(None, ge=1)
    trigger_sweep: bool = Field(False, description="If True, dispatch new sweep jobs via Celery")


class ParamValueScore(BaseModel):
    param_name: str
    param_value: Any
    avg_mcs: float
    avg_db_correctness: float
    avg_llm_quality: float
    score_variance: float
    optimal_range_width: Optional[float] = None
    degradation_rate: Optional[float] = None
    run_count: int


class HyperparamCompareResponse(BaseModel):
    model_id: str
    sweep_id: Optional[str]
    results: list[ParamValueScore]
    recommended_config: dict[str, Any]
    sensitivity_summary: dict[str, str]   # param → HIGH | MEDIUM | LOW


# ── /eval/prompts/compare ─────────────────────────────────────────────────────
class PromptCompareRequest(BaseModel):
    model_id: str
    strategies: list[str] = Field(
        ...,
        description="List of prompt strategies to compare",
        min_length=2
    )
    subtopic: Optional[str] = None
    sample_size: Optional[int] = Field(None, ge=1)
    trigger_runs: bool = Field(False, description="If True, dispatch generation jobs via Celery")


class StrategyScore(BaseModel):
    strategy: str
    avg_mcs: float
    accuracy_lift_vs_zeroshot: float
    consistency_sigma: float
    token_efficiency: float
    reasoning_depth: Optional[float] = None
    format_compliance_rate: float
    run_count: int


class PromptCompareResponse(BaseModel):
    model_id: str
    comparison_id: Optional[str]
    results: list[StrategyScore]
    recommended_strategy_per_subtopic: dict[str, str]


# ── /eval/leaderboard ─────────────────────────────────────────────────────────
class LeaderboardEntry(BaseModel):
    rank: int
    model_id: str
    display_name: str
    provider: str
    mcs_score: Optional[float]
    db_correctness: Optional[float]
    llm_quality: Optional[float]
    elo_rating: int
    contest_wins: int
    contest_total: int
    win_rate: Optional[float]
    best_prompt_strategy: Optional[str]
    best_topic: Optional[str]
    worst_topic: Optional[str]
    avg_latency_ms: Optional[float]
    hallucination_rate: Optional[float]
    last_updated: Optional[datetime]


class LeaderboardResponse(BaseModel):
    total_models: int
    total_contests: int
    total_runs: int
    entries: list[LeaderboardEntry]
    generated_at: datetime


# ── Question Bank Schemas ─────────────────────────────────────────────────────
class QuestionSchema(BaseModel):
    id: str
    subtopic: str
    topic: str
    exercise_number: Optional[str]
    question_text: str
    expected_answer: Optional[str]
    question_type: str
    difficulty: str
    tags: Optional[list[str]]


class QuestionListResponse(BaseModel):
    total: int
    questions: list[QuestionSchema]


# ── Model Registry Schemas ────────────────────────────────────────────────────
class ModelSchema(BaseModel):
    model_id: str
    display_name: str
    provider: str
    api_model: str
    is_judge: bool
    max_context_tokens: Optional[int]
    supports_seed: bool


# ── Batch Evaluation Request ──────────────────────────────────────────────────
class BatchEvalRequest(BaseModel):
    """Trigger evaluation of multiple questions / models at once."""
    model_ids: list[str] = Field(..., description="Challenger model IDs to evaluate")
    question_ids: Optional[list[str]] = Field(
        None,
        description="Specific question UUIDs; if None, runs the full question bank"
    )
    prompt_strategies: list[str] = Field(
        default_factory=lambda: ["zero-shot", "few-shot-cot"],
        description="Strategies to run for each model-question pair"
    )
    hyperparams: Hyperparams = Field(default_factory=Hyperparams)
    run_judge: bool = Field(True, description="Auto-run judge scoring after generation")
    run_contest: bool = Field(True, description="Auto-run contest after all models answer")


class BatchEvalResponse(BaseModel):
    batch_id: str
    total_jobs: int
    task_ids: list[str]
    estimated_completion_seconds: Optional[float]


# ── Export Schemas ────────────────────────────────────────────────────────────
class ExportRequest(BaseModel):
    format: str = Field("json", description="json | csv")
    model_id: Optional[str] = None
    subtopic: Optional[str] = None
    include_raw_answers: bool = Field(False)
