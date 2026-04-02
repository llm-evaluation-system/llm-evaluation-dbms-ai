"""
config.py — Centralised configuration for the LLM Evaluator System.

All environment variables, model identifiers, scoring weights, and system
constants are defined here so that the rest of the codebase depends only on
this module and never on raw os.environ calls.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/llm_evaluator"
)
DATABASE_URL_SYNC: str = os.getenv(
    "DATABASE_URL_SYNC",
    "postgresql://postgres:postgres@localhost:5432/llm_evaluator"
)

# Sandboxed test database used for SQL execution harness
TEST_DATABASE_URL: str = os.getenv(
    "TEST_DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/llm_evaluator_test"
)

# ── LLM API Keys ──────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")

# ── Model Registry ────────────────────────────────────────────────────────────
# Four challenger models + one judge model.
# Update model identifiers here as new versions are released.

CHALLENGER_MODELS: dict[str, dict] = {
    "gpt-4o": {
        "provider": "openai",
        "display_name": "GPT-4o",
        "api_model": "gpt-4o",
        "cost_per_1k_input_tokens": 0.0025,
        "cost_per_1k_output_tokens": 0.010,
        "max_context_tokens": 128000,
        "supports_seed": True,
    },
    "claude-3-5-sonnet": {
        "provider": "anthropic",
        "display_name": "Claude 3.5 Sonnet",
        "api_model": "claude-3-5-sonnet-20241022",
        "cost_per_1k_input_tokens": 0.003,
        "cost_per_1k_output_tokens": 0.015,
        "max_context_tokens": 200000,
        "supports_seed": False,
    },
    "gemini-1.5-pro": {
        "provider": "google",
        "display_name": "Gemini 1.5 Pro",
        "api_model": "gemini-1.5-pro",
        "cost_per_1k_input_tokens": 0.00125,
        "cost_per_1k_output_tokens": 0.005,
        "max_context_tokens": 1000000,
        "supports_seed": False,
    },
    "llama-3.1-70b": {
        "provider": "groq",
        "display_name": "Llama 3.1 70B",
        # "api_model": "llama-3.1-70b-versatile",
        # "api_model": "llama3-70b-8192",
        "api_model": "llama-3.3-70b-versatile",
        "cost_per_1k_input_tokens": 0.00059,
        "cost_per_1k_output_tokens": 0.00079,
        "max_context_tokens": 131072,
        "supports_seed": False,
    },
}

JUDGE_MODEL: dict = {
    "model_id": "gpt-4o",
    "provider": "openai",
    "api_model": "gpt-4o",
    "display_name": "GPT-4o (Judge)",
}

# ── Hyperparameter Grid ───────────────────────────────────────────────────────
DEFAULT_HYPERPARAMS: dict = {
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 1024,
    "top_k": -1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "system_prompt_style": "expert-persona",
    "seed": None,
}

HYPERPARAM_GRID: dict[str, list] = {
    "temperature": [0.0, 0.3, 0.7, 1.0],
    "top_p": [0.7, 0.85, 0.95, 1.0],
    "max_tokens": [256, 512, 1024, 2048],
    "top_k": [10, 40, 80, -1],
    "presence_penalty": [0.0, 0.5, 1.0],
    "frequency_penalty": [0.0, 0.5, 1.0],
    "system_prompt_style": ["minimal", "role-based", "expert-persona"],
    "seed": [42, 137, 999],
}

# ── Scoring Weights ───────────────────────────────────────────────────────────
# Master Composite Score (MCS) pillar weights
MCS_WEIGHTS: dict[str, float] = {
    "db_correctness": 0.50,
    "llm_quality": 0.30,
    "prompting_effectiveness": 0.15,
    "efficiency": 0.05,
}

# SQL Syntactic & Semantic sub-weights
SQL_WEIGHTS: dict[str, float] = {
    "syntactic_parse_success": 0.15,
    "result_set_accuracy": 0.30,
    "clause_appropriateness": 0.20,
    "constraint_correctness": 0.15,
    "idiomatic_postgresql": 0.20,
}

# Conceptual Accuracy sub-weights
CONCEPTUAL_WEIGHTS: dict[str, float] = {
    "factual_correctness": 0.35,
    "completeness": 0.25,
    "absence_of_contradiction": 0.20,
    "topic_specificity": 0.20,
}

# Schema Design sub-weights
SCHEMA_WEIGHTS: dict[str, float] = {
    "entity_coverage": 0.30,
    "fk_correctness": 0.25,
    "normalization_compliance": 0.25,
    "index_appropriateness": 0.20,
}

# Query Optimization sub-weights
QUERY_OPT_WEIGHTS: dict[str, float] = {
    "join_algorithm_selection": 0.25,
    "index_selectivity_reasoning": 0.25,
    "cost_estimation_accuracy": 0.20,
    "plan_tree_correctness": 0.15,
    "optimization_hint_usage": 0.15,
}

# Transaction / Concurrency sub-weights (auto vs judge split)
TRANSACTION_AUTO_WEIGHT: float = 0.60
TRANSACTION_JUDGE_WEIGHT: float = 0.40

# LLM Quality sub-weights
LLM_QUALITY_WEIGHTS: dict[str, float] = {
    "hallucination_rate_inverted": 0.25,
    "reasoning_quality": 0.25,
    "precision_recall_f1": 0.20,
    "format_compliance": 0.15,
    "consistency": 0.15,
}

# Reasoning quality sub-weights
REASONING_WEIGHTS: dict[str, float] = {
    "logical_coherence": 0.30,
    "step_completeness": 0.25,
    "error_propagation": 0.20,
    "self_consistency": 0.25,
}

# Format compliance sub-weights
FORMAT_WEIGHTS: dict[str, float] = {
    "json_output_validity": 0.25,
    "sql_code_block": 0.20,
    "numbered_steps": 0.20,
    "schema_table_formatting": 0.20,
    "length_constraints": 0.15,
}

# Prompting effectiveness sub-weights
PROMPTING_WEIGHTS: dict[str, float] = {
    "accuracy_lift_over_zeroshot": 0.35,
    "consistency_low_variance": 0.20,
    "token_efficiency": 0.15,
    "reasoning_depth": 0.15,
    "format_compliance_rate": 0.15,
}

# Efficiency pillar sub-weights
EFFICIENCY_WEIGHTS: dict[str, float] = {
    "latency_score_inverted": 0.40,
    "token_efficiency": 0.35,
    "cost_per_correct_answer": 0.25,
}

# Hallucination severity tiers (for weighted severity score)
HALLUCINATION_SEVERITY: dict[str, float] = {
    "critical": 3.0,
    "high": 2.0,
    "medium": 1.0,
    "low": 0.5,
}

# ── Verbosity Penalty Thresholds ─────────────────────────────────────────────
VERBOSITY_HIGH_RATIO: float = 3.0      # ratio > 3.0 → 10% penalty
VERBOSITY_LOW_RATIO: float = 0.5       # ratio < 0.5 → 15% penalty
VERBOSITY_HIGH_PENALTY: float = 0.10
VERBOSITY_LOW_PENALTY: float = 0.15

# ── Elo Rating System ─────────────────────────────────────────────────────────
ELO_INITIAL_RATING: int = 1200
ELO_K_FACTOR: int = 32

# ── Self-Consistency ──────────────────────────────────────────────────────────
SELF_CONSISTENCY_K: int = 5           # Number of samples for self-consistency
SELF_CONSISTENCY_TEMPERATURE: float = 0.7

# ── Latency Targets ──────────────────────────────────────────────────────────
TTFT_TARGET_SECONDS: float = 1.5      # Time-to-first-token target
API_RETRY_RATE_TARGET: float = 0.02   # < 2% retry rate target

# ── API Configuration ─────────────────────────────────────────────────────────
API_MAX_RETRIES: int = 3
API_RETRY_BACKOFF_BASE: float = 2.0   # Exponential backoff base
API_TIMEOUT_SECONDS: float = 120.0

# ── Question Bank ─────────────────────────────────────────────────────────────
QUESTION_BANK_EXCEL_PATH: str = os.getenv(
    "QUESTION_BANK_PATH",
    "/mnt/project/Database_Project_EVALUATIONGroup_4.xlsx"
)
QUESTION_BANK_JSON_PATH: str = "data/question_bank.json"
QUESTION_BANK_SHEET_NAME: str = "Question Bank"

# Difficulty tiers used for stratified sampling
DIFFICULTY_TIERS: list[str] = ["easy", "medium", "hard"]

# Stratified sample size for fast hyperparameter sweep (20% of bank)
FAST_SWEEP_SAMPLE_FRACTION: float = 0.20
FAST_SWEEP_MIN_QUESTIONS: int = 5

# ── Application Metadata ──────────────────────────────────────────────────────
APP_TITLE: str = "Master LLM Evaluator System"
APP_VERSION: str = "1.0.0"
APP_DESCRIPTION: str = (
    "Comprehensive Evaluation Benchmarking Framework for "
    "Database Management Systems Question Answering"
)

# ── CORS ──────────────────────────────────────────────────────────────────────
CORS_ORIGINS: list[str] = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:8080"
).split(",")

# ── Celery ────────────────────────────────────────────────────────────────────
CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
