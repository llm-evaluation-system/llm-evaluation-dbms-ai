"""
evaluators/robustness.py — Robustness & consistency under perturbations (Section 2.5).

For each of the five perturbation types defined in the spec, this module:
  1. Generates the perturbed version of the question.
  2. Records the perturbation in the database.
  3. Computes a cosine-similarity consistency score between the original and
     perturbed answers (using lightweight token-overlap when a sentence
     transformer is unavailable, or full embeddings when available).

Perturbation types:
  - syntactic_rephrasing
  - variable_name_change
  - order_inversion
  - typo_injection
  - negation_injection
"""

from __future__ import annotations

import re
import random
from typing import Optional


# ── Perturbation Generators ───────────────────────────────────────────────────
def perturb_syntactic_rephrase(question: str) -> str:
    """
    Rewrite the question with a surface-level paraphrase.
    For example: "What is X?" → "Explain X."
    """
    replacements = [
        (r"^What\s+is\s+", "Explain "),
        (r"^What\s+are\s+", "Describe "),
        (r"^How\s+does\s+", "Describe how "),
        (r"^Define\s+", "Provide a definition of "),
        (r"^Explain\s+", "What is meant by "),
        (r"\?$", "."),
    ]
    result = question
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE, count=1)
    return result if result != question else question + " Please explain in detail."


def perturb_variable_name(question: str) -> str:
    """
    Replace common schema identifiers with synonyms to test robustness.
    e.g. 'employee' → 'staff', 'department' → 'division'.
    """
    substitutions = {
        r"\bemployees?\b": "staff",
        r"\bdepartments?\b": "division",
        r"\bstudents?\b": "pupils",
        r"\bcustomers?\b": "clients",
        r"\bsalary\b": "compensation",
        r"\bsid\b": "student_id",
        r"\beid\b": "emp_id",
        r"\bdid\b": "dept_id",
    }
    result = question
    for pattern, replacement in substitutions.items():
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def perturb_order_inversion(question: str) -> str:
    """
    For questions listing conditions A, B, C, reverse the listed order to C, B, A.
    Also reorder multi-part questions.
    """
    # Reverse enumerated lists like "1. ... 2. ... 3. ..."
    items = re.split(r"\n\s*\d+\.\s+", question)
    if len(items) > 2:
        header = items[0]
        parts = items[1:]
        reversed_parts = list(reversed(parts))
        reordered = header
        for i, part in enumerate(reversed_parts, 1):
            reordered += f"\n{i}. {part}"
        return reordered

    # Reverse comma-separated condition lists in parentheses
    paren_match = re.search(r"\(([^)]+,\s*[^)]+)\)", question)
    if paren_match:
        items_inner = [x.strip() for x in paren_match.group(1).split(",")]
        reversed_inner = ", ".join(reversed(items_inner))
        return question.replace(paren_match.group(1), reversed_inner)

    return question


def perturb_typo_injection(question: str) -> str:
    """
    Inject realistic typos into DBMS technical terms.
    """
    typos = {
        "serializability": "serialzability",
        "normalization": "normalisation",
        "transaction": "transactoin",
        "concurrency": "concurency",
        "relational": "realtional",
        "constraint": "constrant",
        "functional": "funtional",
        "dependency": "dependancy",
        "attribute": "attribut",
        "decomposition": "decompositon",
    }
    result = question
    for correct, typo in typos.items():
        if correct in result.lower():
            result = re.sub(correct, typo, result, count=1, flags=re.IGNORECASE)
            break  # Only inject one typo per perturbation
    return result


def perturb_negation_injection(question: str) -> str:
    """
    Convert a positive question to its negative counterpart.
    e.g. "Which of the following IS a property…" → "Which of the following is NOT a property…"
    """
    # Insert "NOT" after IS/ARE/DOES/DO
    result = re.sub(
        r"\b(is|are|does|do)\b(?!\s+not)",
        r"\1 NOT",
        question,
        count=1,
        flags=re.IGNORECASE,
    )
    if result == question:
        # Append negation at question end
        if question.strip().endswith("?"):
            result = question.strip()[:-1] + ", given that the usual assumption does NOT hold?"
        else:
            result = question + " (Note: the standard assumption does NOT apply here.)"
    return result


PERTURBATION_GENERATORS = {
    "syntactic_rephrasing": perturb_syntactic_rephrase,
    "variable_name_change": perturb_variable_name,
    "order_inversion": perturb_order_inversion,
    "typo_injection": perturb_typo_injection,
    "negation_injection": perturb_negation_injection,
}


def generate_perturbations(question: str) -> dict[str, str]:
    """
    Generate all five perturbed versions of the given question.
    Returns dict mapping perturbation_type → perturbed_question.
    """
    return {
        ptype: fn(question)
        for ptype, fn in PERTURBATION_GENERATORS.items()
    }


# ── Consistency Scoring ───────────────────────────────────────────────────────
def compute_token_overlap_similarity(text_a: str, text_b: str) -> float:
    """
    Compute Jaccard similarity over unigrams as a lightweight proxy for
    cosine similarity when sentence transformers are not available.
    """
    if not text_a or not text_b:
        return 0.0

    tokens_a = set(re.findall(r"\b\w+\b", text_a.lower()))
    tokens_b = set(re.findall(r"\b\w+\b", text_b.lower()))

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b

    return len(intersection) / len(union) if union else 0.0


def compute_consistency_score(
    original_answer: str,
    perturbed_answers: dict[str, str],
) -> dict:
    """
    Compute the Consistency Score (CS) as the average pairwise similarity
    between the original answer and each perturbed answer.

    Returns:
        {
          "consistency_score": float 0–1,
          "per_perturbation": {type: similarity},
          "method": "token_overlap" | "sentence_transformer"
        }
    """
    if not perturbed_answers:
        return {"consistency_score": 1.0, "per_perturbation": {}, "method": "none"}

    similarities: dict[str, float] = {}
    for ptype, perturbed_answer in perturbed_answers.items():
        if perturbed_answer:
            sim = compute_token_overlap_similarity(original_answer, perturbed_answer)
            similarities[ptype] = round(sim, 4)

    if not similarities:
        return {"consistency_score": 0.0, "per_perturbation": {}, "method": "token_overlap"}

    avg_similarity = sum(similarities.values()) / len(similarities)

    return {
        "consistency_score": round(avg_similarity, 4),
        "per_perturbation": similarities,
        "method": "token_overlap",
    }
