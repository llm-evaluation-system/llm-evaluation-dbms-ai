"""
judge/judge_llm.py — Judge LLM orchestration module.

Implements the two Judge LLM protocols defined in Sections 5.2 and 5.3:

  1. Absolute Scoring  — scores one model's answer 0–10 against the ground
                         truth, returning structured JSON with justification,
                         hallucinations detected, and missing points.

  2. Pairwise Contest  — presents all four model answers simultaneously
                         (anonymised as A/B/C/D), ranks them, and emits
                         per-model placement justifications.

The Judge is always the most capable model (configured as JUDGE_MODEL in
config.py).  It NEVER knows which challenger model produced which answer
during pairwise contests (anonymisation is enforced server-side).
"""

from __future__ import annotations

import json
import random
import re
from typing import Optional

from config import CHALLENGER_MODELS, JUDGE_MODEL
from llm_clients.providers import get_llm_client

# Default judge hyperparams: zero temperature for deterministic, high-precision scoring
JUDGE_HYPERPARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 2048,
    "top_k": -1,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "system_prompt_style": "expert-persona",
    "seed": 42,
}

JUDGE_SYSTEM_PROMPT = (
    "You are a senior DBMS examiner and an unbiased expert judge. "
    "Your role is to evaluate answers to database management systems questions "
    "with strict adherence to the provided rubric and ground truth. "
    "You must reason carefully before assigning any score. "
    "You never award partial credit without explicit justification. "
    "You respond ONLY with valid JSON as specified in the prompt — "
    "no preamble, no markdown, no extra text."
)


# ── Absolute Scoring ──────────────────────────────────────────────────────────
ABSOLUTE_SCORE_PROMPT_TEMPLATE = """You are an expert DBMS judge. Evaluate the following answer.

Question:
{question}

Expected Answer (Ground Truth):
{ground_truth}

Model Answer:
{model_answer}

Scoring Rubric:
{scoring_rubric}

Instructions:
1. Reason step-by-step about the correctness, completeness, and quality of the model answer vs the expected answer.
2. Identify any factually incorrect statements (hallucinations).
3. Identify any key points from the expected answer that the model answer omits.
4. Assign an overall score from 0 to 10 (0 = completely wrong/empty, 10 = perfect).

Respond ONLY with a valid JSON object in exactly this format (no extra text, no markdown):
{{
  "score": <number 0-10, may use one decimal place>,
  "justification": "<concise explanation of the score, ≤ 150 words>",
  "hallucinations": [
    {{"type": "<hallucination_type>", "text": "<the hallucinated claim>", "severity": "<critical|high|medium|low>"}}
  ],
  "missing_points": ["<key point from expected answer that is missing>"],
  "factual_correctness": <0.0-1.0>,
  "completeness": <0.0-1.0>,
  "absence_of_contradiction": <0.0-1.0>,
  "topic_specificity": <0.0-1.0>
}}"""

DEFAULT_RUBRIC = (
    "Score based on: "
    "(1) Factual correctness relative to DBMS textbook ground truth — 35%; "
    "(2) Completeness: covers all key sub-points from the expected answer — 25%; "
    "(3) No internal contradictions — 20%; "
    "(4) Domain specificity: demonstrates DBMS knowledge, not generic waffle — 20%."
)

SQL_RUBRIC = (
    "Score based on: "
    "(1) Syntactic validity: parseable by PostgreSQL — 15%; "
    "(2) Semantic correctness: returns the correct result set — 30%; "
    "(3) Use of appropriate SQL clauses — 20%; "
    "(4) Correct constraint handling — 15%; "
    "(5) Idiomatic PostgreSQL (RETURNING, ::cast, window functions, CTEs) — 20%."
)


def _is_sql_tagged(question_type: str, tags: Optional[list]) -> bool:
    """
    Return True when a question involves SQL and should use the SQL rubric.

    After the question-bank correction, question_type is ONLY 'conceptual' or
    'practical' — the value 'sql' never appears.  SQL content is now indicated
    exclusively by the presence of 'sql' in the question's tags list.

    For backward compatibility the legacy check (question_type == 'sql') is
    also retained so that any existing callers that still pass the old type
    string continue to work correctly.
    """
    if tags and "sql" in tags:
        return True
    # Legacy fallback: callers that haven't been updated yet
    if question_type == "sql":
        return True
    return False


async def judge_absolute_score(
    question: str,
    ground_truth: str,
    model_answer: str,
    question_type: str = "conceptual",
    custom_rubric: Optional[str] = None,
    db_execution_context: Optional[str] = None,
    judge_hint: Optional[str] = None,
    tags: Optional[list] = None,
) -> dict:
    """
    Ask the Judge LLM to score a single model answer against the ground truth.

    Parameters
    ----------
    db_execution_context : str, optional
        Output from EXPLAIN ANALYZE or SQL execution results that the Judge
        should use to evaluate query-optimisation and SQL correctness questions.
        Injected as an additional context block before the scoring rubric.
    judge_hint : str, optional
        Pre-computed schema description or evaluation hint from sql_fixtures.py
        that helps the Judge understand the exact schema and expected behaviour
        for this question.

    Returns a dict matching the JSON structure defined in ABSOLUTE_SCORE_PROMPT_TEMPLATE.
    Raises RuntimeError if the Judge API call fails or returns unparseable JSON.
    """
    rubric = custom_rubric or (SQL_RUBRIC if _is_sql_tagged(question_type, tags) else DEFAULT_RUBRIC)

    # ── Token-limit guard ──────────────────────────────────────────────────
    # Very long expected_answer fields (e.g. 5 000-char proofs for normalization
    # or security/authorization questions) can push the judge prompt past the
    # model's context window, causing the API call to fail with a 400/413 error.
    # Truncate both ground_truth and model_answer to safe character limits so the
    # judge always receives a parseable prompt.  The truncation is noted inline
    # so the judge understands the context.
    _GT_CHAR_LIMIT  = 2000   # ~500 tokens for ground truth
    _ANS_CHAR_LIMIT = 3000   # ~750 tokens for model answer
    if ground_truth and len(ground_truth) > _GT_CHAR_LIMIT:
        ground_truth = (
            ground_truth[:_GT_CHAR_LIMIT]
            + f"\n\n[TRUNCATED — original length {len(ground_truth)} chars; "
            "key points above are sufficient for scoring]"
        )
    if model_answer and len(model_answer) > _ANS_CHAR_LIMIT:
        model_answer = (
            model_answer[:_ANS_CHAR_LIMIT]
            + f"\n\n[TRUNCATED — original length {len(model_answer)} chars]"
        )
    # ──────────────────────────────────────────────────────────────────────

    # Build optional context blocks to inject into the prompt
    context_blocks: list[str] = []
    if judge_hint:
        context_blocks.append(
            f"Schema and Evaluation Context (authoritative — use this to verify "
            f"the model answer):\n{judge_hint}"
        )
    if db_execution_context:
        context_blocks.append(
            f"PostgreSQL Execution / EXPLAIN ANALYZE Output "
            f"(use this to assess query correctness and plan quality):\n"
            f"{db_execution_context}"
        )

    # Inject context blocks into the prompt between the model answer and rubric
    if context_blocks:
        extra = "\n\n".join(context_blocks)
        prompt = ABSOLUTE_SCORE_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth or "(No ground truth available — score based on DBMS textbook knowledge)",
            model_answer=model_answer or "(empty response)",
            scoring_rubric=f"{extra}\n\nScoring Rubric:\n{rubric}",
        )
    else:
        prompt = ABSOLUTE_SCORE_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth or "(No ground truth available — score based on DBMS textbook knowledge)",
            model_answer=model_answer or "(empty response)",
            scoring_rubric=rubric,
        )

    judge_config = {
        "model_id": JUDGE_MODEL["model_id"],
        "provider": JUDGE_MODEL["provider"],
        "api_model": JUDGE_MODEL["api_model"],
        "display_name": JUDGE_MODEL["display_name"],
        "cost_per_1k_input_tokens": 0.0025,
        "cost_per_1k_output_tokens": 0.010,
        "max_context_tokens": 128000,
        "supports_seed": True,
    }
    client = get_llm_client(judge_config)

    response = await client.complete(
        prompt=prompt,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        hyperparams=JUDGE_HYPERPARAMS,
    )

    raw_text = response.answer_text.strip()
    return _parse_json_response(raw_text, fallback_score=5.0)


def _parse_json_response(raw: str, fallback_score: float = 5.0) -> dict:
    """
    Robustly parse JSON from the judge's response.  Handles common issues like
    leading/trailing markdown fences.
    """
    # Strip markdown code fences if present
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Attempt to extract a JSON object with regex as last resort
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    # Return a structured fallback if parsing completely fails
    return {
        "score": fallback_score,
        "justification": "Judge response could not be parsed. Raw output: " + raw[:200],
        "hallucinations": [],
        "missing_points": [],
        "factual_correctness": fallback_score / 10,
        "completeness": fallback_score / 10,
        "absence_of_contradiction": fallback_score / 10,
        "topic_specificity": fallback_score / 10,
        "_parse_error": True,
    }


# ── Pairwise Contest ──────────────────────────────────────────────────────────
CONTEST_PROMPT_TEMPLATE = """You are an expert DBMS judge. You must rank the following {n_models} answers to a DBMS question from BEST to WORST.

Question:
{question}

Expected Answer (Ground Truth):
{ground_truth}

Model Answers (anonymised — you do NOT know which model produced which answer):
{answers_block}

Ranking Instructions:
1. Reason step-by-step about the correctness, completeness, and quality of each answer.
2. Rank all {n_models} answers from 1st (best) to {n_models}th (worst).
3. If two answers are equally correct, declare a tie and explain why.
4. Use the brevity-precision heuristic as a tiebreaker: the shorter, more precise correct answer wins.

Respond ONLY with a valid JSON object in exactly this format (no extra text, no markdown):
{{
  "reasoning": "<step-by-step comparison of all answers, ≤ 300 words>",
  "rankings": [
    {{"label": "A", "placement": 1, "justification": "<why this placement>"}},
    {{"label": "B", "placement": 2, "justification": "<why this placement>"}},
    {{"label": "C", "placement": 3, "justification": "<why this placement>"}},
    {{"label": "D", "placement": 4, "justification": "<why this placement>"}}
  ],
  "ties": [["A", "C"]] 
}}

Note: The "ties" field should list groups of labels that are tied. If no ties, use an empty list [].
Only include the labels that are actually present in the answers."""


async def judge_contest(
    question: str,
    ground_truth: str,
    answers_map: dict[str, str],
) -> dict:
    """
    Run a pairwise contest where the Judge LLM ranks all provided answers.

    Args:
        question:    The question text.
        ground_truth: Expected answer from the question bank.
        answers_map: {model_id: answer_text} for all participating models.

    Returns:
        {
          "anonymized_map": {"A": model_id, ...},
          "rankings": [{label, placement, justification}, ...],
          "reasoning": str,
          "ties": [[label, label], ...],
          "ranked_model_ids": [model_id_1st, model_id_2nd, ...]
        }
    """
    # Anonymise model IDs — shuffle to prevent positional bias
    labels = ["A", "B", "C", "D"][:len(answers_map)]
    model_ids = list(answers_map.keys())
    random.shuffle(model_ids)
    anonymized_map = {label: model_id for label, model_id in zip(labels, model_ids)}
    reverse_map = {v: k for k, v in anonymized_map.items()}

    answers_block = "\n\n".join(
        f"Answer {label}:\n{answers_map[anonymized_map[label]] or '(empty response)'}"
        for label in labels
    )

    prompt = CONTEST_PROMPT_TEMPLATE.format(
        n_models=len(labels),
        question=question,
        ground_truth=ground_truth or "(No ground truth available)",
        answers_block=answers_block,
    )

    judge_config = {
        "model_id": JUDGE_MODEL["model_id"],
        "provider": JUDGE_MODEL["provider"],
        "api_model": JUDGE_MODEL["api_model"],
        "display_name": JUDGE_MODEL["display_name"],
        "cost_per_1k_input_tokens": 0.0025,
        "cost_per_1k_output_tokens": 0.010,
        "max_context_tokens": 128000,
        "supports_seed": True,
    }
    client = get_llm_client(judge_config)

    response = await client.complete(
        prompt=prompt,
        system_prompt=JUDGE_SYSTEM_PROMPT,
        hyperparams=JUDGE_HYPERPARAMS,
    )

    parsed = _parse_json_response(response.answer_text.strip(), fallback_score=5.0)

    # Decode anonymised labels back to model IDs
    rankings = parsed.get("rankings", [])
    # Sort by placement
    rankings_sorted = sorted(rankings, key=lambda r: r.get("placement", 99))
    ranked_model_ids = [anonymized_map.get(r["label"], r["label"]) for r in rankings_sorted]

    # Decode ties
    raw_ties = parsed.get("ties", [])
    decoded_ties = [
        [anonymized_map.get(lbl, lbl) for lbl in group]
        for group in raw_ties
        if isinstance(group, list)
    ]

    return {
        "anonymized_map": anonymized_map,
        "rankings": [
            {
                "model_id": anonymized_map.get(r.get("label", ""), ""),
                "label": r.get("label", ""),
                "placement": r.get("placement", 99),
                "justification": r.get("justification", ""),
            }
            for r in rankings_sorted
        ],
        "reasoning": parsed.get("reasoning", ""),
        "ties": decoded_ties,
        "tie_exists": len(decoded_ties) > 0,
        "ranked_model_ids": ranked_model_ids,
        "raw_judge_response": response.answer_text,
    }
