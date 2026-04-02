"""
scoring/composite.py — Master Composite Score (MCS) computation engine.

Implements the full scoring formula from Section 5.1:

  MCS = 0.50 × DB_Correctness + 0.30 × LLM_Quality
        + 0.15 × Prompting_Effectiveness + 0.05 × Efficiency

Each pillar is itself a weighted combination of sub-scores.  This module
provides functions to:
  - Compute DB_Correctness from individual evaluator outputs.
  - Compute LLM_Quality from judge scores and automated checks.
  - Compute Prompting_Effectiveness for a given strategy comparison.
  - Compute Efficiency from latency and cost metrics.
  - Combine all four pillars into the final MCS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from config import (
    CONCEPTUAL_WEIGHTS,
    EFFICIENCY_WEIGHTS,
    ELO_INITIAL_RATING,
    LLM_QUALITY_WEIGHTS,
    MCS_WEIGHTS,
    PROMPTING_WEIGHTS,
    QUERY_OPT_WEIGHTS,
    REASONING_WEIGHTS,
    SCHEMA_WEIGHTS,
    SQL_WEIGHTS,
    TRANSACTION_AUTO_WEIGHT,
    TRANSACTION_JUDGE_WEIGHT,
    TTFT_TARGET_SECONDS,
    VERBOSITY_HIGH_PENALTY,
    VERBOSITY_HIGH_RATIO,
    VERBOSITY_LOW_PENALTY,
    VERBOSITY_LOW_RATIO,
)


# ── Sub-score containers ──────────────────────────────────────────────────────
@dataclass
class SQLScores:
    syntactic_parse_success: float = 0.0   # 0 or 1 (automated)
    result_set_f1: float = 0.0             # F1 of expected vs returned rows (automated)
    clause_appropriateness: float = 0.0    # LLM judge 0–1
    constraint_correctness: float = 0.0    # LLM judge 0–1
    idiomatic_postgresql: float = 0.0      # LLM judge + regex 0–1

    def weighted(self) -> float:
        w = SQL_WEIGHTS
        return clamp(
            w["syntactic_parse_success"] * self.syntactic_parse_success
            + w["result_set_accuracy"] * self.result_set_f1
            + w["clause_appropriateness"] * self.clause_appropriateness
            + w["constraint_correctness"] * self.constraint_correctness
            + w["idiomatic_postgresql"] * self.idiomatic_postgresql
        )


@dataclass
class ConceptualScores:
    factual_correctness: float = 0.0
    completeness: float = 0.0
    absence_of_contradiction: float = 0.0
    topic_specificity: float = 0.0

    def weighted(self) -> float:
        w = CONCEPTUAL_WEIGHTS
        return clamp(
            w["factual_correctness"] * self.factual_correctness
            + w["completeness"] * self.completeness
            + w["absence_of_contradiction"] * self.absence_of_contradiction
            + w["topic_specificity"] * self.topic_specificity
        )


@dataclass
class SchemaScores:
    entity_coverage: float = 0.0
    fk_correctness: float = 0.0
    normalization_compliance: float = 0.0
    index_appropriateness: float = 0.0

    def weighted(self) -> float:
        w = SCHEMA_WEIGHTS
        return clamp(
            w["entity_coverage"] * self.entity_coverage
            + w["fk_correctness"] * self.fk_correctness
            + w["normalization_compliance"] * self.normalization_compliance
            + w["index_appropriateness"] * self.index_appropriateness
        )


@dataclass
class QueryOptScores:
    join_algorithm_selection: float = 0.0
    index_selectivity_reasoning: float = 0.0
    cost_estimation_accuracy: float = 0.0
    plan_tree_correctness: float = 0.0
    optimization_hint_usage: float = 0.0

    def weighted(self) -> float:
        w = QUERY_OPT_WEIGHTS
        return clamp(
            w["join_algorithm_selection"] * self.join_algorithm_selection
            + w["index_selectivity_reasoning"] * self.index_selectivity_reasoning
            + w["cost_estimation_accuracy"] * self.cost_estimation_accuracy
            + w["plan_tree_correctness"] * self.plan_tree_correctness
            + w["optimization_hint_usage"] * self.optimization_hint_usage
        )


@dataclass
class TransactionScores:
    auto_score: float = 0.0    # 60% weight — deterministic auto-grading
    judge_score: float = 0.0   # 40% weight — LLM judge

    def weighted(self) -> float:
        return clamp(
            TRANSACTION_AUTO_WEIGHT * self.auto_score
            + TRANSACTION_JUDGE_WEIGHT * self.judge_score
        )


@dataclass
class DBCorrectnessBundle:
    """Container for all DB-correctness pillar sub-scores."""
    sql: Optional[SQLScores] = None
    conceptual: Optional[ConceptualScores] = None
    schema: Optional[SchemaScores] = None
    query_opt: Optional[QueryOptScores] = None
    transaction: Optional[TransactionScores] = None

    def pillar_score(self, question_type: str) -> float:
        """
        Route to the correct sub-scorer based on question type, then
        scale to 0–100.

        After the question-bank correction, question_type is ONLY
        'conceptual' or 'practical'.  The granular content routes
        ('sql', 'transaction', etc.) are now passed as the routing key
        from get_db_correctness_route() in sql_fixtures.py, not from
        question.question_type directly.

        This method therefore accepts BOTH the legacy fine-grained keys
        (for backward compatibility if anything still passes them) and
        the new coarse-grained values.
        """
        type_map = {
            # Legacy fine-grained keys (kept for backward compat)
            "sql":                self.sql,
            "schema":             self.schema,
            "relational_algebra": self.conceptual,
            "query_optimization": self.query_opt,
            "transaction":        self.transaction,
            "normalization":      self.conceptual,
            "warehousing":        self.conceptual,
            # New routing keys from get_db_correctness_route()
            "conceptual":         self.conceptual,
            "practical":          self.conceptual,  # practical without sql tag → conceptual scorer
        }
        scorer = type_map.get(question_type, self.conceptual)
        if scorer is None:
            # Fall back to conceptual scorer if the specific one is not populated
            scorer = self.conceptual
        if scorer is None:
            return 0.0
        return scorer.weighted() * 100.0


# ── LLM Quality Pillar ────────────────────────────────────────────────────────
@dataclass
class ReasoningScores:
    logical_coherence: float = 0.0
    step_completeness: float = 0.0
    error_propagation: float = 1.0   # 1.0 = no error propagation (good)
    self_consistency: float = 0.0

    def weighted(self) -> float:
        w = REASONING_WEIGHTS
        return clamp(
            w["logical_coherence"] * self.logical_coherence
            + w["step_completeness"] * self.step_completeness
            + w["error_propagation"] * self.error_propagation
            + w["self_consistency"] * self.self_consistency
        )


@dataclass
class LLMQualityBundle:
    hallucination_rate: float = 0.0        # 0–1 (0 = no hallucinations → good)
    hallucination_severity_score: float = 0.0
    reasoning: Optional[ReasoningScores] = None
    text_precision: float = 0.0
    text_recall: float = 0.0
    text_f1: float = 0.0
    verbosity_ratio: float = 1.0           # 1.0 = median length
    format_compliance_score: float = 0.0   # 0–1
    consistency_score: float = 0.0         # cosine similarity across perturbations

    def pillar_score(self) -> float:
        """Compute LLM Quality pillar score (0–100)."""
        hallucination_inverted = clamp(1.0 - self.hallucination_rate)

        reasoning_score = self.reasoning.weighted() if self.reasoning else 0.5
        text_f1 = self.text_f1 or _compute_f1(self.text_precision, self.text_recall)

        # Apply verbosity penalty
        verbosity_multiplier = 1.0
        if self.verbosity_ratio > VERBOSITY_HIGH_RATIO:
            verbosity_multiplier = 1.0 - VERBOSITY_HIGH_PENALTY
        elif self.verbosity_ratio < VERBOSITY_LOW_RATIO:
            verbosity_multiplier = 1.0 - VERBOSITY_LOW_PENALTY

        w = LLM_QUALITY_WEIGHTS
        raw = (
            w["hallucination_rate_inverted"] * hallucination_inverted
            + w["reasoning_quality"] * reasoning_score
            + w["precision_recall_f1"] * text_f1 * verbosity_multiplier
            + w["format_compliance"] * self.format_compliance_score
            + w["consistency"] * self.consistency_score
        )
        return clamp(raw) * 100.0


# ── Prompting Effectiveness Pillar ────────────────────────────────────────────
@dataclass
class PromptingBundle:
    accuracy_lift_over_zeroshot: float = 0.0    # Δ score vs zero-shot baseline (normalised 0–1)
    consistency_sigma: float = 0.0              # std dev across runs (inverted)
    token_efficiency: float = 0.0               # Δ score / Δ token count (normalised)
    reasoning_depth: float = 0.0                # CoT quality (LLM judge)
    format_compliance_rate: float = 0.0         # % of runs that comply with format

    def pillar_score(self) -> float:
        # Consistency: lower sigma is better → invert
        consistency_score = clamp(1.0 - min(self.consistency_sigma, 1.0))

        w = PROMPTING_WEIGHTS
        raw = (
            w["accuracy_lift_over_zeroshot"] * clamp(self.accuracy_lift_over_zeroshot)
            + w["consistency_low_variance"] * consistency_score
            + w["token_efficiency"] * clamp(self.token_efficiency)
            + w["reasoning_depth"] * clamp(self.reasoning_depth)
            + w["format_compliance_rate"] * clamp(self.format_compliance_rate)
        )
        return clamp(raw) * 100.0


# ── Efficiency Pillar ─────────────────────────────────────────────────────────
@dataclass
class EfficiencyBundle:
    ttft_ms: float = 0.0           # Time-to-first-token in milliseconds
    total_latency_ms: float = 0.0
    tokens_per_second: float = 0.0
    cost_usd: float = 0.0
    output_tokens: int = 0
    is_correct: bool = False        # Whether the answer was correct (for cost-per-correct)

    def pillar_score(self) -> float:
        # TTFT score: 1.0 at 0ms → 0.0 at 3× target
        ttft_target_ms = TTFT_TARGET_SECONDS * 1000
        ttft_score = clamp(1.0 - (self.ttft_ms / (ttft_target_ms * 3.0)))

        # Token efficiency: normalise TPS (assume 200 TPS is excellent)
        tps_score = clamp(self.tokens_per_second / 200.0)

        # Cost score: lower is better; scale so $0.001 = 1.0, $0.01 = 0.5
        cost_score = clamp(1.0 - (self.cost_usd / 0.02)) if self.cost_usd > 0 else 0.5

        w = EFFICIENCY_WEIGHTS
        raw = (
            w["latency_score_inverted"] * ttft_score
            + w["token_efficiency"] * tps_score
            + w["cost_per_correct_answer"] * cost_score
        )
        return clamp(raw) * 100.0


# ── Master Composite Score ────────────────────────────────────────────────────
def compute_mcs(
    db_correctness: float,
    llm_quality: float,
    prompting_effectiveness: float = 50.0,
    efficiency: float = 50.0,
) -> float:
    """
    Compute the Master Composite Score from the four pillar scores (each 0–100).
    MCS = 0.50 × DB + 0.30 × LLM + 0.15 × Prompting + 0.05 × Efficiency
    """
    w = MCS_WEIGHTS
    mcs = (
        w["db_correctness"] * db_correctness
        + w["llm_quality"] * llm_quality
        + w["prompting_effectiveness"] * prompting_effectiveness
        + w["efficiency"] * efficiency
    )
    return clamp(mcs, 0.0, 100.0)


# ── Hallucination Metrics ─────────────────────────────────────────────────────
def compute_hallucination_rate(
    responses_with_hallucination: int,
    total_responses: int,
) -> float:
    """HR = (# responses with ≥1 hallucination) / total × 100%."""
    if total_responses == 0:
        return 0.0
    return (responses_with_hallucination / total_responses) * 100.0


def compute_hallucination_severity_score(
    hallucinations: list[dict],
    severity_weights: Optional[dict[str, float]] = None,
) -> float:
    """
    Compute the Hallucination Severity Score (HSS) as a weighted sum
    normalised by total possible severity mass.
    """
    from config import HALLUCINATION_SEVERITY
    weights = severity_weights or HALLUCINATION_SEVERITY

    if not hallucinations:
        return 0.0

    total_weight = sum(
        weights.get(h.get("severity", "low"), 0.5)
        for h in hallucinations
    )
    max_possible = len(hallucinations) * weights.get("critical", 3.0)
    if max_possible == 0:
        return 0.0
    return min(total_weight / max_possible, 1.0)


# ── Helpers ───────────────────────────────────────────────────────────────────
def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _compute_f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def judge_score_to_normalized(score_0_10: float) -> float:
    """Convert a 0–10 judge score to a 0–1 normalised value."""
    return clamp(score_0_10 / 10.0)


def compute_verbosity_ratio(answer_word_count: int, median_expected_word_count: int) -> float:
    """Compute the verbosity ratio as answer / median expected word count."""
    if median_expected_word_count <= 0:
        return 1.0
    return answer_word_count / median_expected_word_count
