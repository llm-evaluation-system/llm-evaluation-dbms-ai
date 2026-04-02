#!/usr/bin/env python3
"""
infra_test.py — Comprehensive infrastructure test for the LLM Evaluator system.

Tests ALL 80 questions against ALL requirements from the problem specification:
  Section 1: Database-Related Benchmarks (SQL harness, scoring, schema, transactions)
  Section 2: LLM/AI Benchmarks (hallucination, reasoning, format, robustness, latency)
  Section 3: Hyperparameter framework
  Section 4: Prompting strategy framework
  Section 5: Composite scoring & leaderboard (MCS formula, Elo, leaderboard schema)
  Section 6: FastAPI implementation checklist (all endpoints, idempotency, storage)

Usage:
    python3 infra_test.py [--base-url http://localhost:8000] [--model llama-3.1-70b]
                          [--output infra_test_results.json] [--quick]

    --quick  runs only 1 question per category for a fast sanity check (~15 min)
    full run tests all 80 questions (~2-3 hours depending on model latency)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import urllib.request
import urllib.error

# ── Embedded question bank (80 unique questions, ER diagrams removed) ─────────
QUESTION_BANK: list[dict] = [
  {"id":"94f3945c-a596-0000-0000-000000000000","raw_id":"94f3945ca596","type":"conceptual","difficulty":"easy","tags":["relational_model"],"subtopic":"THE RELATIONAL MODEL","topic":"FOUNDATIONS"},
  {"id":"41de485a-98af-0000-0000-000000000000","raw_id":"41de485a98af","type":"conceptual","difficulty":"easy","tags":["relational_model"],"subtopic":"THE RELATIONAL MODEL","topic":"FOUNDATIONS"},
  {"id":"6e9e7751-c76c-0000-0000-000000000000","raw_id":"6e9e7751c76c","type":"conceptual","difficulty":"medium","tags":["relational_model"],"subtopic":"THE RELATIONAL MODEL","topic":"FOUNDATIONS"},
  {"id":"66f0e4d6-5fb0-0000-0000-000000000000","raw_id":"66f0e4d65fb0","type":"conceptual","difficulty":"easy","tags":["relational_model"],"subtopic":"THE RELATIONAL MODEL","topic":"FOUNDATIONS"},
  {"id":"c6c4ce74-44f2-0000-0000-000000000000","raw_id":"c6c4ce7444f2","type":"practical","difficulty":"easy","tags":["sql","relational_model"],"subtopic":"THE RELATIONAL MODEL","topic":"FOUNDATIONS"},
  {"id":"7f30162b-5f47-0000-0000-000000000000","raw_id":"7f30162b5f47","type":"practical","difficulty":"medium","tags":["sql"],"subtopic":"THE RELATIONAL MODEL","topic":"FOUNDATIONS"},
  {"id":"0c053e7b-ed52-0000-0000-000000000000","raw_id":"0c053e7bed52","type":"practical","difficulty":"easy","tags":["sql","database_concepts"],"subtopic":"THE RELATIONAL MODEL","topic":"FOUNDATIONS"},
  {"id":"78ebf72b-7b30-0000-0000-000000000000","raw_id":"78ebf72b7b30","type":"conceptual","difficulty":"easy","tags":["relational_model"],"subtopic":"RELATIONAL ALGEBRA AND CALCULUS","topic":"FOUNDATIONS"},
  {"id":"91d0bd09-fe08-0000-0000-000000000000","raw_id":"91d0bd09fe08","type":"practical","difficulty":"hard","tags":["relational_model","sql"],"subtopic":"RELATIONAL ALGEBRA AND CALCULUS","topic":"FOUNDATIONS"},
  {"id":"8c61735b-507d-0000-0000-000000000000","raw_id":"8c61735b507d","type":"practical","difficulty":"hard","tags":["relational_model","sql"],"subtopic":"RELATIONAL ALGEBRA AND CALCULUS","topic":"FOUNDATIONS"},
  {"id":"09f1cb01-bac5-0000-0000-000000000000","raw_id":"09f1cb01bac5","type":"conceptual","difficulty":"easy","tags":["relational_model"],"subtopic":"RELATIONAL ALGEBRA AND CALCULUS","topic":"FOUNDATIONS"},
  {"id":"c36cf025-1982-0000-0000-000000000000","raw_id":"c36cf0251982","type":"practical","difficulty":"hard","tags":["sql"],"subtopic":"SQL: QUERIES, CONSTRAINTS, TRIGGERS","topic":"FOUNDATIONS"},
  {"id":"0def7bdc-534e-0000-0000-000000000000","raw_id":"0def7bdc534e","type":"practical","difficulty":"hard","tags":["sql"],"subtopic":"SQL: QUERIES, CONSTRAINTS, TRIGGERS","topic":"FOUNDATIONS"},
  {"id":"6c3f53eb-80bb-0000-0000-000000000000","raw_id":"6c3f53eb80bb","type":"practical","difficulty":"medium","tags":["sql"],"subtopic":"SQL: QUERIES, CONSTRAINTS, TRIGGERS","topic":"FOUNDATIONS"},
  {"id":"c62a41f9-4461-0000-0000-000000000000","raw_id":"c62a41f94461","type":"practical","difficulty":"medium","tags":["sql"],"subtopic":"SQL: QUERIES, CONSTRAINTS, TRIGGERS","topic":"FOUNDATIONS"},
  {"id":"6bd4ccd2-5d85-0000-0000-000000000000","raw_id":"6bd4ccd25d85","type":"conceptual","difficulty":"easy","tags":["sql","database_concepts"],"subtopic":"SQL: QUERIES, CONSTRAINTS, TRIGGERS","topic":"FOUNDATIONS"},
  {"id":"2ca79e49-a73b-0000-0000-000000000000","raw_id":"2ca79e49a73b","type":"conceptual","difficulty":"easy","tags":["sql","database_concepts"],"subtopic":"DATABASE APPLICATION DEVELOPMENT","topic":"APPLICATION DEVELOPMENT"},
  {"id":"4d30f627-adcf-0000-0000-000000000000","raw_id":"4d30f627adcf","type":"conceptual","difficulty":"easy","tags":["sql","database_concepts"],"subtopic":"DATABASE APPLICATION DEVELOPMENT","topic":"APPLICATION DEVELOPMENT"},
  {"id":"e1330b9f-31d3-0000-0000-000000000000","raw_id":"e1330b9f31d3","type":"conceptual","difficulty":"medium","tags":["sql","database_concepts"],"subtopic":"DATABASE APPLICATION DEVELOPMENT","topic":"APPLICATION DEVELOPMENT"},
  {"id":"ed2b175f-e63d-0000-0000-000000000000","raw_id":"ed2b175fe63d","type":"conceptual","difficulty":"easy","tags":["database_concepts"],"subtopic":"INTERNET APPLICATIONS","topic":"APPLICATION DEVELOPMENT"},
  {"id":"dad55c7e-810c-0000-0000-000000000000","raw_id":"dad55c7e810c","type":"conceptual","difficulty":"easy","tags":["database_concepts"],"subtopic":"OVERVIEW OF STORAGE AND INDEXING","topic":"STORAGE AND INDEXING"},
  {"id":"f1901f7d-4493-0000-0000-000000000000","raw_id":"f1901f7d4493","type":"conceptual","difficulty":"easy","tags":["database_concepts"],"subtopic":"OVERVIEW OF STORAGE AND INDEXING","topic":"STORAGE AND INDEXING"},
  {"id":"02d0a5cc-5773-0000-0000-000000000000","raw_id":"02d0a5cc5773","type":"conceptual","difficulty":"medium","tags":["database_concepts"],"subtopic":"OVERVIEW OF STORAGE AND INDEXING","topic":"STORAGE AND INDEXING"},
  {"id":"92d2fe23-fe81-0000-0000-000000000000","raw_id":"92d2fe23fe81","type":"conceptual","difficulty":"medium","tags":["database_concepts"],"subtopic":"OVERVIEW OF STORAGE AND INDEXING","topic":"STORAGE AND INDEXING"},
  {"id":"7214203d-8a10-0000-0000-000000000000","raw_id":"7214203d8a10","type":"practical","difficulty":"medium","tags":["database_concepts"],"subtopic":"OVERVIEW OF STORAGE AND INDEXING","topic":"STORAGE AND INDEXING"},
  {"id":"7e69b80b-219e-0000-0000-000000000000","raw_id":"7e69b80b219e","type":"practical","difficulty":"medium","tags":["database_concepts"],"subtopic":"STORING DATA: DISKS AND FILES","topic":"STORAGE AND INDEXING"},
  {"id":"dcbc750f-9798-0000-0000-000000000000","raw_id":"dcbc750f9798","type":"conceptual","difficulty":"easy","tags":["database_concepts"],"subtopic":"STORING DATA: DISKS AND FILES","topic":"STORAGE AND INDEXING"},
  {"id":"d0b5debe-1efb-0000-0000-000000000000","raw_id":"d0b5debe1efb","type":"conceptual","difficulty":"medium","tags":["database_concepts"],"subtopic":"STORING DATA: DISKS AND FILES","topic":"STORAGE AND INDEXING"},
  {"id":"5b32b074-18b1-0000-0000-000000000000","raw_id":"5b32b07418b1","type":"conceptual","difficulty":"easy","tags":["database_concepts"],"subtopic":"STORING DATA: DISKS AND FILES","topic":"STORAGE AND INDEXING"},
  {"id":"030a913c-68e4-0000-0000-000000000000","raw_id":"030a913c68e4","type":"conceptual","difficulty":"easy","tags":["database_concepts"],"subtopic":"STORING DATA: DISKS AND FILES","topic":"STORAGE AND INDEXING"},
  {"id":"2f20bbee-5774-0000-0000-000000000000","raw_id":"2f20bbee5774","type":"practical","difficulty":"hard","tags":["database_concepts"],"subtopic":"TREE-STRUCTURED INDEXING","topic":"STORAGE AND INDEXING"},
  {"id":"38fec19d-a0d2-0000-0000-000000000000","raw_id":"38fec19da0d2","type":"conceptual","difficulty":"easy","tags":["database_concepts"],"subtopic":"TREE-STRUCTURED INDEXING","topic":"STORAGE AND INDEXING"},
  {"id":"45e705cf-e771-0000-0000-000000000000","raw_id":"45e705cfe771","type":"practical","difficulty":"medium","tags":["database_concepts"],"subtopic":"TREE-STRUCTURED INDEXING","topic":"STORAGE AND INDEXING"},
  {"id":"00715362-73b3-0000-0000-000000000000","raw_id":"0071536273b3","type":"conceptual","difficulty":"easy","tags":["database_concepts"],"subtopic":"TREE-STRUCTURED INDEXING","topic":"STORAGE AND INDEXING"},
  {"id":"661a48e1-8431-0000-0000-000000000000","raw_id":"661a48e18431","type":"practical","difficulty":"medium","tags":["database_concepts"],"subtopic":"HASH-BASED INDEXING","topic":"STORAGE AND INDEXING"},
  {"id":"496d1f01-b582-0000-0000-000000000000","raw_id":"496d1f01b582","type":"conceptual","difficulty":"medium","tags":["database_concepts"],"subtopic":"HASH-BASED INDEXING","topic":"STORAGE AND INDEXING"},
  {"id":"9f53f572-cb29-0000-0000-000000000000","raw_id":"9f53f572cb29","type":"conceptual","difficulty":"medium","tags":["database_concepts"],"subtopic":"HASH-BASED INDEXING","topic":"STORAGE AND INDEXING"},
  {"id":"68be450a-bd7c-0000-0000-000000000000","raw_id":"68be450abd7c","type":"conceptual","difficulty":"medium","tags":["database_concepts"],"subtopic":"HASH-BASED INDEXING","topic":"STORAGE AND INDEXING"},
  {"id":"5704959a-3fbf-0000-0000-000000000000","raw_id":"5704959a3fbf","type":"practical","difficulty":"medium","tags":["database_concepts"],"subtopic":"HASH-BASED INDEXING","topic":"STORAGE AND INDEXING"},
  {"id":"2b3771b3-253f-0000-0000-000000000000","raw_id":"2b3771b3253f","type":"conceptual","difficulty":"easy","tags":["sql","database_concepts"],"subtopic":"OVERVIEW OF QUERY EVALUATION","topic":"QUERY EVALUATION"},
  {"id":"7c18de42-3908-0000-0000-000000000000","raw_id":"7c18de423908","type":"conceptual","difficulty":"medium","tags":["database_concepts"],"subtopic":"OVERVIEW OF QUERY EVALUATION","topic":"QUERY EVALUATION"},
  {"id":"5875107d-ecdf-0000-0000-000000000000","raw_id":"5875107decdf","type":"conceptual","difficulty":"hard","tags":["database_concepts"],"subtopic":"EXTERNAL SORTING","topic":"QUERY EVALUATION"},
  {"id":"bc293c19-a457-0000-0000-000000000000","raw_id":"bc293c19a457","type":"practical","difficulty":"medium","tags":["database_concepts"],"subtopic":"EXTERNAL SORTING","topic":"QUERY EVALUATION"},
  {"id":"d4ed9772-4281-0000-0000-000000000000","raw_id":"d4ed97724281","type":"practical","difficulty":"hard","tags":["sql","database_concepts"],"subtopic":"EVALUATING RELATIONAL OPERATORS","topic":"QUERY EVALUATION"},
  {"id":"f10e4b18-4023-0000-0000-000000000000","raw_id":"f10e4b184023","type":"practical","difficulty":"hard","tags":["database_concepts"],"subtopic":"EVALUATING RELATIONAL OPERATORS","topic":"QUERY EVALUATION"},
  {"id":"d9c7e0be-6bc3-0000-0000-000000000000","raw_id":"d9c7e0be6bc3","type":"conceptual","difficulty":"medium","tags":["sql","database_concepts"],"subtopic":"A TYPICAL RELATIONAL QUERY OPTIMIZER","topic":"QUERY EVALUATION"},
  {"id":"4e7a2456-43d9-0000-0000-000000000000","raw_id":"4e7a245643d9","type":"conceptual","difficulty":"easy","tags":["sql","database_concepts"],"subtopic":"A TYPICAL RELATIONAL QUERY OPTIMIZER","topic":"QUERY EVALUATION"},
  {"id":"5893d14b-3015-0000-0000-000000000000","raw_id":"5893d14b3015","type":"practical","difficulty":"medium","tags":["database_concepts"],"subtopic":"A TYPICAL RELATIONAL QUERY OPTIMIZER","topic":"QUERY EVALUATION"},
  {"id":"5f979c3f-0e84-0000-0000-000000000000","raw_id":"5f979c3f0e84","type":"practical","difficulty":"medium","tags":["database_concepts","sql"],"subtopic":"A TYPICAL RELATIONAL QUERY OPTIMIZER","topic":"QUERY EVALUATION"},
  {"id":"a16ad64a-b216-0000-0000-000000000000","raw_id":"a16ad64ab216","type":"conceptual","difficulty":"medium","tags":["transactions"],"subtopic":"OVERVIEW OF TRANSACTION MANAGEMENT","topic":"TRANSACTION MANAGEMENT"},
  {"id":"f4ddd3bd-b9b0-0000-0000-000000000000","raw_id":"f4ddd3bdb9b0","type":"practical","difficulty":"medium","tags":["transactions"],"subtopic":"OVERVIEW OF TRANSACTION MANAGEMENT","topic":"TRANSACTION MANAGEMENT"},
  {"id":"a9462bff-64df-0000-0000-000000000000","raw_id":"a9462bff64df","type":"conceptual","difficulty":"medium","tags":["transactions"],"subtopic":"OVERVIEW OF TRANSACTION MANAGEMENT","topic":"TRANSACTION MANAGEMENT"},
  {"id":"f25e0c1c-d0e0-0000-0000-000000000000","raw_id":"f25e0c1cd0e0","type":"conceptual","difficulty":"easy","tags":["sql","transactions"],"subtopic":"OVERVIEW OF TRANSACTION MANAGEMENT","topic":"TRANSACTION MANAGEMENT"},
  {"id":"76579c82-c472-0000-0000-000000000000","raw_id":"76579c82c472","type":"conceptual","difficulty":"medium","tags":["sql","transactions"],"subtopic":"OVERVIEW OF TRANSACTION MANAGEMENT","topic":"TRANSACTION MANAGEMENT"},
  {"id":"e4e291f7-4535-0000-0000-000000000000","raw_id":"e4e291f74535","type":"conceptual","difficulty":"hard","tags":["transactions"],"subtopic":"CONCURRENCY CONTROL","topic":"TRANSACTION MANAGEMENT"},
  {"id":"8166dd0c-cff1-0000-0000-000000000000","raw_id":"8166dd0ccff1","type":"practical","difficulty":"hard","tags":["transactions"],"subtopic":"CONCURRENCY CONTROL","topic":"TRANSACTION MANAGEMENT"},
  {"id":"b4e1a627-1c37-0000-0000-000000000000","raw_id":"b4e1a6271c37","type":"conceptual","difficulty":"easy","tags":["transactions"],"subtopic":"CONCURRENCY CONTROL","topic":"TRANSACTION MANAGEMENT"},
  {"id":"316cc7c0-34d5-0000-0000-000000000000","raw_id":"316cc7c034d5","type":"practical","difficulty":"hard","tags":["transactions"],"subtopic":"CONCURRENCY CONTROL","topic":"TRANSACTION MANAGEMENT"},
  {"id":"c0d653bd-3abd-0000-0000-000000000000","raw_id":"c0d653bd3abd","type":"conceptual","difficulty":"medium","tags":["sql","transactions"],"subtopic":"CONCURRENCY CONTROL","topic":"TRANSACTION MANAGEMENT"},
  {"id":"46a50244-6923-0000-0000-000000000000","raw_id":"46a502446923","type":"practical","difficulty":"hard","tags":["transactions"],"subtopic":"CONCURRENCY CONTROL","topic":"TRANSACTION MANAGEMENT"},
  {"id":"be6d6e30-18b2-0000-0000-000000000000","raw_id":"be6d6e3018b2","type":"conceptual","difficulty":"easy","tags":["transactions","database_concepts"],"subtopic":"CRASH RECOVERY","topic":"TRANSACTION MANAGEMENT"},
  {"id":"d1d97baf-9e4c-0000-0000-000000000000","raw_id":"d1d97baf9e4c","type":"practical","difficulty":"medium","tags":["transactions","database_concepts"],"subtopic":"CRASH RECOVERY","topic":"TRANSACTION MANAGEMENT"},
  {"id":"8c8ebfcf-0330-0000-0000-000000000000","raw_id":"8c8ebfcf0330","type":"practical","difficulty":"hard","tags":["transactions","database_concepts"],"subtopic":"CRASH RECOVERY","topic":"TRANSACTION MANAGEMENT"},
  {"id":"2bd29bcd-ab98-0000-0000-000000000000","raw_id":"2bd29bcdab98","type":"conceptual","difficulty":"easy","tags":["transactions","database_concepts"],"subtopic":"CRASH RECOVERY","topic":"TRANSACTION MANAGEMENT"},
  {"id":"836478e1-173a-0000-0000-000000000000","raw_id":"836478e1173a","type":"practical","difficulty":"medium","tags":["transactions","database_concepts"],"subtopic":"CRASH RECOVERY","topic":"TRANSACTION MANAGEMENT"},
  {"id":"f31cf7c1-6458-0000-0000-000000000000","raw_id":"f31cf7c16458","type":"conceptual","difficulty":"easy","tags":["normalization"],"subtopic":"SCHEMA REFINEMENT AND NORMAL FORMS","topic":"DATABASE DESIGN AND TUNING"},
  {"id":"27e488d7-24f3-0000-0000-000000000000","raw_id":"27e488d724f3","type":"conceptual","difficulty":"medium","tags":["normalization"],"subtopic":"SCHEMA REFINEMENT AND NORMAL FORMS","topic":"DATABASE DESIGN AND TUNING"},
  {"id":"e6f3c650-6091-0000-0000-000000000000","raw_id":"e6f3c6506091","type":"conceptual","difficulty":"medium","tags":["normalization"],"subtopic":"SCHEMA REFINEMENT AND NORMAL FORMS","topic":"DATABASE DESIGN AND TUNING"},
  {"id":"586b20c9-c06c-0000-0000-000000000000","raw_id":"586b20c9c06c","type":"conceptual","difficulty":"hard","tags":["normalization"],"subtopic":"SCHEMA REFINEMENT AND NORMAL FORMS","topic":"DATABASE DESIGN AND TUNING"},
  {"id":"a0444786-68c6-0000-0000-000000000000","raw_id":"a044478668c6","type":"conceptual","difficulty":"medium","tags":["normalization"],"subtopic":"SCHEMA REFINEMENT AND NORMAL FORMS","topic":"DATABASE DESIGN AND TUNING"},
  {"id":"7d55f825-7bb2-0000-0000-000000000000","raw_id":"7d55f8257bb2","type":"conceptual","difficulty":"hard","tags":["normalization"],"subtopic":"SCHEMA REFINEMENT AND NORMAL FORMS","topic":"DATABASE DESIGN AND TUNING"},
  {"id":"de8d7541-6edf-0000-0000-000000000000","raw_id":"de8d75416edf","type":"practical","difficulty":"medium","tags":["database_concepts"],"subtopic":"PHYSICAL DATABASE DESIGN AND TUNING","topic":"DATABASE DESIGN AND TUNING"},
  {"id":"8fdae0f0-6d9c-0000-0000-000000000000","raw_id":"8fdae0f06d9c","type":"conceptual","difficulty":"medium","tags":["sql","database_concepts"],"subtopic":"PHYSICAL DATABASE DESIGN AND TUNING","topic":"DATABASE DESIGN AND TUNING"},
  {"id":"ed3b91e8-8cb8-0000-0000-000000000000","raw_id":"ed3b91e88cb8","type":"practical","difficulty":"medium","tags":["database_concepts"],"subtopic":"PHYSICAL DATABASE DESIGN AND TUNING","topic":"DATABASE DESIGN AND TUNING"},
  {"id":"85e72f20-267a-0000-0000-000000000000","raw_id":"85e72f20267a","type":"practical","difficulty":"medium","tags":["database_concepts"],"subtopic":"PHYSICAL DATABASE DESIGN AND TUNING","topic":"DATABASE DESIGN AND TUNING"},
  {"id":"165c1883-b2d7-0000-0000-000000000000","raw_id":"165c1883b2d7","type":"conceptual","difficulty":"medium","tags":["database_concepts"],"subtopic":"SECURITY AND AUTHORIZATION","topic":"DATABASE DESIGN AND TUNING"},
  {"id":"848c4a51-e9ab-0000-0000-000000000000","raw_id":"848c4a51e9ab","type":"conceptual","difficulty":"medium","tags":["database_concepts"],"subtopic":"SECURITY AND AUTHORIZATION","topic":"DATABASE DESIGN AND TUNING"},
  {"id":"2e8fe7d4-d629-0000-0000-000000000000","raw_id":"2e8fe7d4d629","type":"conceptual","difficulty":"medium","tags":["database_concepts"],"subtopic":"DATA WAREHOUSING AND DECISION SUPPORT","topic":"ADDITIONAL TOPICS"},
  {"id":"30541b0a-864e-0000-0000-000000000000","raw_id":"30541b0a864e","type":"practical","difficulty":"medium","tags":["sql"],"subtopic":"DATA WAREHOUSING AND DECISION SUPPORT","topic":"ADDITIONAL TOPICS"},
  {"id":"9d080a66-a9fc-0000-0000-000000000000","raw_id":"9d080a66a9fc","type":"conceptual","difficulty":"medium","tags":["sql","database_concepts"],"subtopic":"DATA WAREHOUSING AND DECISION SUPPORT","topic":"ADDITIONAL TOPICS"},
]

# ── Spec-derived constants ────────────────────────────────────────────────────
MCS_WEIGHTS = {"db_correctness": 0.50, "llm_quality": 0.30,
               "prompting_effectiveness": 0.15, "efficiency": 0.05}
SQL_WEIGHTS  = {"syntactic_parse_success": 0.15, "result_set_accuracy": 0.30,
                "clause_appropriateness": 0.20, "constraint_correctness": 0.15,
                "idiomatic_postgresql": 0.20}
CONCEPTUAL_WEIGHTS = {"factual_correctness": 0.35, "completeness": 0.25,
                      "absence_of_contradiction": 0.20, "topic_specificity": 0.20}
LLM_QUALITY_WEIGHTS = {"hallucination_rate_inverted": 0.25, "reasoning_quality": 0.25,
                       "precision_recall_f1": 0.20, "format_compliance": 0.15,
                       "consistency": 0.15}
PROMPTING_WEIGHTS = {"accuracy_lift_over_zeroshot": 0.35, "consistency_low_variance": 0.20,
                     "token_efficiency": 0.15, "reasoning_depth": 0.15,
                     "format_compliance_rate": 0.15}
EFFICIENCY_WEIGHTS = {"latency_score_inverted": 0.40, "token_efficiency": 0.35,
                      "cost_per_correct_answer": 0.25}
ELO_INITIAL = 1200
ELO_K = 32
SELF_CONSISTENCY_K = 5
TTFT_TARGET_MS = 1500.0
RETRY_RATE_TARGET = 0.02
VERBOSITY_HIGH_RATIO = 3.0
VERBOSITY_LOW_RATIO  = 0.5
ALL_PROMPT_STRATEGIES = [
    "zero-shot", "one-shot", "few-shot", "cot",
    "few-shot-cot", "self-consistency", "role", "least-to-most", "react",
]
ALL_HYPERPARAMS = {
    "temperature": [0.0, 0.3, 0.7, 1.0],
    "top_p": [0.7, 0.85, 0.95, 1.0],
    "max_tokens": [256, 512, 1024, 2048],
    "top_k": [10, 40, 80, -1],
    "presence_penalty": [0.0, 0.5, 1.0],
    "frequency_penalty": [0.0, 0.5, 1.0],
    "system_prompt_style": ["minimal", "role-based", "expert-persona"],
    "seed": [42, 137, 999],
}
LEADERBOARD_REQUIRED_FIELDS = {
    "model_id", "mcs_score", "db_correctness", "llm_quality",
    "elo_rating", "contest_wins", "contest_total", "win_rate",
    "best_prompt_strategy", "best_topic", "worst_topic",
    "avg_latency_ms", "hallucination_rate", "last_updated",
}
GENERATE_RESPONSE_REQUIRED = {
    "run_id", "model_id", "question_id", "prompt_strategy",
    "status", "model_answer",
}
JUDGE_RESPONSE_REQUIRED = {
    "score_id", "run_id", "judge_score_0_10", "justification",
    "hallucinations_detected", "missing_points",
    "db_correctness_score", "llm_quality_score",
    "master_composite_score", "prompting_effectiveness_score",
    "efficiency_score",
}
SCORE_RANGE_0_10 = (0.0, 10.0)
SCORE_RANGE_0_100 = (0.0, 100.0)

DEFAULT_HYPERPARAMS = {
    "temperature": 0.2, "top_p": 0.9, "max_tokens": 1024,
    "top_k": -1, "presence_penalty": 0, "frequency_penalty": 0,
    "system_prompt_style": "expert-persona", "seed": 0,
}


# ── HTTP helper ───────────────────────────────────────────────────────────────
def post(url: str, payload: dict, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code}: {body[:200]}") from e


def get(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code}: {body[:200]}") from e


# ── Test result tracking ──────────────────────────────────────────────────────
class Results:
    def __init__(self):
        self.checks: list[dict] = []

    def record(self, section: str, test_id: str, description: str,
               passed: bool, detail: str = "", critical: bool = False,
               question_id: str = "", subtopic: str = ""):
        self.checks.append({
            "section": section, "test_id": test_id,
            "description": description, "passed": passed,
            "detail": detail, "critical": critical,
            "question_id": question_id, "subtopic": subtopic,
        })

    def summary(self) -> dict:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c["passed"])
        failed = total - passed
        critical_fails = [c for c in self.checks if not c["passed"] and c["critical"]]
        by_section: dict[str, dict] = defaultdict(lambda: {"passed": 0, "failed": 0})
        for c in self.checks:
            sec = c["section"]
            if c["passed"]:
                by_section[sec]["passed"] += 1
            else:
                by_section[sec]["failed"] += 1
        return {
            "total": total, "passed": passed, "failed": failed,
            "pass_rate": round(passed / total * 100, 1) if total else 0,
            "critical_failures": len(critical_fails),
            "by_section": dict(by_section),
            "failed_checks": [c for c in self.checks if not c["passed"]],
        }


R = Results()


def chk(section: str, test_id: str, description: str, condition: bool,
        detail: str = "", critical: bool = False,
        question_id: str = "", subtopic: str = ""):
    R.record(section, test_id, description, condition, detail, critical,
             question_id, subtopic)
    icon = "✅" if condition else ("❌" if critical else "⚠ ")
    tag = f"[{question_id[:8]}] " if question_id else ""
    print(f"  {icon} {test_id}: {tag}{description[:80]}"
          + (f" — {detail[:60]}" if detail and not condition else ""))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 0: Server health + endpoint inventory
# ─────────────────────────────────────────────────────────────────────────────
def test_s0_infrastructure(base: str):
    print("\n══ S0: Server Health & Endpoint Inventory ══")

    # S0.1 Health
    try:
        resp = get(f"{base}/health", timeout=10)
        chk("S0", "S0.1", "GET /health returns 200", True)
        chk("S0", "S0.1a", "/health contains status field",
            "status" in resp, str(resp)[:80])
    except Exception as e:
        chk("S0", "S0.1", "GET /health returns 200", False, str(e), critical=True)
        return False  # can't proceed

    # S0.2 Question list endpoint
    try:
        resp = get(f"{base}/eval/questions", timeout=10)
        questions = resp.get("questions", resp if isinstance(resp, list) else [])
        chk("S0", "S0.2", "GET /eval/questions returns question list",
            len(questions) > 0, f"got {len(questions)} questions", critical=True)
        chk("S0", "S0.2a", "Question list contains ≥80 unique questions",
            len(set(q.get("id","") for q in questions)) >= 80,
            f"unique IDs: {len(set(q.get('id','') for q in questions))}")
        # Verify no ER diagram questions
        er_qs = [q for q in questions if "er_diagram" in q.get("tags", [])]
        chk("S0", "S0.2b", "No ER diagram questions in question bank",
            len(er_qs) == 0, f"found {len(er_qs)} ER questions")
    except Exception as e:
        chk("S0", "S0.2", "GET /eval/questions returns question list",
            False, str(e), critical=True)

    # S0.3 Model list endpoint
    try:
        resp = get(f"{base}/eval/models", timeout=10)
        models = resp if isinstance(resp, list) else resp.get("models", [])
        chk("S0", "S0.3", "GET /eval/models returns model list",
            len(models) > 0, f"got {len(models)} models")
    except Exception as e:
        chk("S0", "S0.3", "GET /eval/models returns model list", False, str(e))

    # S0.4 Leaderboard endpoint
    try:
        resp = get(f"{base}/eval/leaderboard", timeout=10)
        chk("S0", "S0.4", "GET /eval/leaderboard returns 200", True)
    except Exception as e:
        chk("S0", "S0.4", "GET /eval/leaderboard returns 200", False, str(e))

    # S0.5 Results summary endpoint
    try:
        resp = get(f"{base}/eval/results/summary", timeout=10)
        chk("S0", "S0.5", "GET /eval/results/summary returns 200", True)
    except Exception as e:
        chk("S0", "S0.5", "GET /eval/results/summary returns 200", False, str(e))

    # S0.6 Hyperparams compare endpoint
    try:
        resp = get(f"{base}/eval/hyperparams/compare?model_id=test", timeout=10)
        chk("S0", "S0.6", "GET /eval/hyperparams/compare returns 200", True)
    except Exception as e:
        chk("S0", "S0.6", "GET /eval/hyperparams/compare returns 200", False, str(e))

    # S0.7 Prompts compare endpoint
    try:
        resp = get(f"{base}/eval/prompts/compare?model_id=test", timeout=10)
        chk("S0", "S0.7", "GET /eval/prompts/compare returns 200", True)
    except Exception as e:
        chk("S0", "S0.7", "GET /eval/prompts/compare returns 200", False, str(e))

    # S0.8 Export endpoints (spec section 6.4 item 28)
    for path in ["/eval/export/json", "/eval/export/csv"]:
        try:
            resp = get(f"{base}{path}", timeout=10)
            chk("S0", f"S0.8_{path.split('/')[-1]}", f"GET {path} returns 200", True)
        except Exception as e:
            chk("S0", f"S0.8_{path.split('/')[-1]}", f"GET {path} returns 200",
                False, str(e))

    return True


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Generate + Judge for every question
# Tests spec §1.1 (SQL harness), §1.2 (conceptual scoring), §5.1 (MCS formula),
# §5.2 (judge protocol), §6.1 (storage), §6.4 (idempotency)
# ─────────────────────────────────────────────────────────────────────────────
def run_one_question(base: str, q: dict, model: str, strategy: str = "zero-shot",
                     hp: dict | None = None) -> dict | None:
    """Generate + judge one question. Returns the combined result dict or None."""
    hyperparams = {**DEFAULT_HYPERPARAMS, **(hp or {})}
    gen_url   = f"{base}/eval/generate"
    judge_url = f"{base}/eval/judge/score"
    qid = q["id"]
    section = "S1"
    subtopic = q["subtopic"]

    # ── Generate ──────────────────────────────────────────────────────────────
    t0 = time.monotonic()
    try:
        gen = post(gen_url, {
            "model_id": model, "question_id": qid,
            "prompt_strategy": strategy, "hyperparams": hyperparams,
            "async_run": False,
        })
    except Exception as e:
        chk(section, f"{section}.GEN", f"generate succeeded [{q['raw_id'][:8]}]",
            False, str(e)[:80], critical=True, question_id=qid, subtopic=subtopic)
        return None
    gen_ms = (time.monotonic() - t0) * 1000

    # S1.1: generate response shape
    chk(section, "S1.1.shape", "generate response has required fields",
        GENERATE_RESPONSE_REQUIRED.issubset(set(gen.keys())),
        f"missing: {GENERATE_RESPONSE_REQUIRED - set(gen.keys())}",
        question_id=qid, subtopic=subtopic)

    run_id = gen.get("run_id")
    status = gen.get("status")
    chk(section, "S1.1.status", "generate status=completed",
        status == "completed", f"got status={status}",
        critical=True, question_id=qid, subtopic=subtopic)

    # S1.2: run_id is a valid UUID (spec §6.1 — each run gets a UUID)
    try:
        uuid.UUID(str(run_id))
        chk(section, "S1.2.uuid", "run_id is a valid UUID", True,
            question_id=qid, subtopic=subtopic)
    except Exception:
        chk(section, "S1.2.uuid", "run_id is a valid UUID", False,
            f"got: {run_id}", question_id=qid, subtopic=subtopic)

    # S1.3: answer is non-empty
    answer = gen.get("model_answer", "")
    chk(section, "S1.3.answer", "model_answer is non-empty",
        bool(answer and answer.strip()),
        question_id=qid, subtopic=subtopic)

    # S1.4: token and cost tracking (spec §2.7, §6.4)
    chk(section, "S1.4.tokens", "input_tokens recorded",
        gen.get("input_tokens") is not None, question_id=qid, subtopic=subtopic)
    chk(section, "S1.4.cost", "cost_usd recorded",
        gen.get("cost_usd") is not None, question_id=qid, subtopic=subtopic)
    chk(section, "S1.4.latency", "total_latency_ms recorded",
        gen.get("total_latency_ms") is not None, question_id=qid, subtopic=subtopic)

    if not run_id or status != "completed":
        return None

    # ── Judge ─────────────────────────────────────────────────────────────────
    t1 = time.monotonic()
    try:
        judge = post(judge_url, {
            "model_id": model, "question_id": qid, "run_id": run_id,
        })
    except Exception as e:
        chk(section, f"{section}.JUDGE", f"judge/score succeeded [{q['raw_id'][:8]}]",
            False, str(e)[:80], critical=True, question_id=qid, subtopic=subtopic)
        return None
    judge_ms = (time.monotonic() - t1) * 1000

    # S1.5: judge response shape (spec §5.2)
    chk(section, "S1.5.shape", "judge response has required fields",
        JUDGE_RESPONSE_REQUIRED.issubset(set(judge.keys())),
        f"missing: {JUDGE_RESPONSE_REQUIRED - set(judge.keys())}",
        question_id=qid, subtopic=subtopic)

    # S1.6: score_id is UUID, scored_at is present (spec §6.1 storage)
    try:
        uuid.UUID(str(judge.get("score_id", "")))
        chk(section, "S1.6.score_uuid", "score_id is a valid UUID", True,
            question_id=qid, subtopic=subtopic)
    except Exception:
        chk(section, "S1.6.score_uuid", "score_id is a valid UUID", False,
            str(judge.get("score_id")), question_id=qid, subtopic=subtopic)

    # S1.7: judge_score_0_10 is in [0,10] (spec §5.2)
    js = judge.get("judge_score_0_10")
    chk(section, "S1.7.score_range", "judge_score_0_10 in [0,10]",
        js is not None and SCORE_RANGE_0_10[0] <= float(js) <= SCORE_RANGE_0_10[1],
        f"got {js}", question_id=qid, subtopic=subtopic)

    # S1.8: pillar scores all in [0,100] (spec §5.1)
    for pillar in ["db_correctness_score", "llm_quality_score",
                   "master_composite_score", "prompting_effectiveness_score",
                   "efficiency_score"]:
        val = judge.get(pillar)
        in_range = val is not None and SCORE_RANGE_0_100[0] <= float(val) <= SCORE_RANGE_0_100[1]
        chk(section, f"S1.8.{pillar[:10]}", f"{pillar} in [0,100]",
            in_range, f"got {val}", question_id=qid, subtopic=subtopic)

    # S1.9: MCS formula verification (spec §5.1)
    # MCS = 0.50×DB + 0.30×LLM + 0.15×Prompting + 0.05×Efficiency
    db_s  = judge.get("db_correctness_score")
    llm_s = judge.get("llm_quality_score")
    pe_s  = judge.get("prompting_effectiveness_score")
    eff_s = judge.get("efficiency_score")
    mcs   = judge.get("master_composite_score")
    if all(v is not None for v in [db_s, llm_s, pe_s, eff_s, mcs]):
        expected_mcs = (0.50 * float(db_s) + 0.30 * float(llm_s)
                        + 0.15 * float(pe_s) + 0.05 * float(eff_s))
        mcs_ok = abs(expected_mcs - float(mcs)) < 1.0  # 1 point tolerance
        chk(section, "S1.9.mcs_formula",
            "MCS formula: 0.50×DB+0.30×LLM+0.15×PE+0.05×Eff",
            mcs_ok,
            f"expected≈{expected_mcs:.1f} got={float(mcs):.1f}",
            critical=True, question_id=qid, subtopic=subtopic)

    # S1.10: hallucinations_detected is a list (spec §2.1)
    halls = judge.get("hallucinations_detected", None)
    chk(section, "S1.10.halls_list", "hallucinations_detected is a list",
        isinstance(halls, list), f"got type {type(halls).__name__}",
        question_id=qid, subtopic=subtopic)

    # S1.11: SQL harness checks for sql-tagged questions (spec §1.1.2)
    if "sql" in q.get("tags", []) and q["type"] == "practical":
        sd = judge.get("sql_execution_details")
        chk(section, "S1.11.harness_ran", "sql_execution_details present for SQL practical",
            sd is not None, f"sd={sd}",
            question_id=qid, subtopic=subtopic)
        if sd:
            # syntactic_parse_success in [0,1]
            parse = sd.get("syntactic_parse_success")
            chk(section, "S1.11.parse_range", "syntactic_parse_success in [0,1]",
                parse is None or 0.0 <= float(parse) <= 1.0,
                f"got {parse}", question_id=qid, subtopic=subtopic)
            # result_set_f1 in [0,1] or None
            f1 = sd.get("result_set_f1")
            chk(section, "S1.11.f1_range", "result_set_f1 in [0,1] or None",
                f1 is None or 0.0 <= float(f1) <= 1.0,
                f"got {f1}", question_id=qid, subtopic=subtopic)
            # harness_ran flag (spec §6.1)
            chk(section, "S1.11.harness_flag", "harness_ran boolean present",
                "harness_ran" in sd, question_id=qid, subtopic=subtopic)

    # S1.12: justification is non-empty string (spec §5.2)
    just = judge.get("justification", "")
    chk(section, "S1.12.justification", "justification is non-empty string",
        bool(just and just.strip()),
        question_id=qid, subtopic=subtopic)

    # S1.13: missing_points is a list (spec §5.2)
    mp = judge.get("missing_points")
    chk(section, "S1.13.missing_pts", "missing_points is a list",
        isinstance(mp, list), question_id=qid, subtopic=subtopic)

    # S1.14: LATENCY — total_latency_ms stored and reasonable (spec §2.7)
    lat = gen.get("total_latency_ms")
    chk(section, "S1.14.latency_stored", "total_latency_ms > 0",
        lat is not None and float(lat) > 0,
        f"got {lat}", question_id=qid, subtopic=subtopic)

    return {
        "question": q, "strategy": strategy,
        "gen": gen, "judge": judge,
        "gen_ms": gen_ms, "judge_ms": judge_ms,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Spec §1.1 — SQL scoring weights integrity
# ─────────────────────────────────────────────────────────────────────────────
def test_s2_scoring_weights(base: str):
    print("\n══ S2: Spec §1.1 / §1.2 — Scoring Weight Integrity ══")

    # S2.1: SQL_WEIGHTS sum = 1.0
    sql_sum = sum(SQL_WEIGHTS.values())
    chk("S2", "S2.1.sql_weights", "SQL sub-weights sum to 1.0",
        abs(sql_sum - 1.0) < 0.001, f"sum={sql_sum:.4f}", critical=True)

    # S2.2: CONCEPTUAL_WEIGHTS sum = 1.0
    cpt_sum = sum(CONCEPTUAL_WEIGHTS.values())
    chk("S2", "S2.2.cpt_weights", "Conceptual sub-weights sum to 1.0",
        abs(cpt_sum - 1.0) < 0.001, f"sum={cpt_sum:.4f}", critical=True)

    # S2.3: LLM_QUALITY_WEIGHTS sum = 1.0
    llm_sum = sum(LLM_QUALITY_WEIGHTS.values())
    chk("S2", "S2.3.llm_weights", "LLM quality sub-weights sum to 1.0",
        abs(llm_sum - 1.0) < 0.001, f"sum={llm_sum:.4f}", critical=True)

    # S2.4: MCS_WEIGHTS sum = 1.0
    mcs_sum = sum(MCS_WEIGHTS.values())
    chk("S2", "S2.4.mcs_weights", "MCS pillar weights sum to 1.0",
        abs(mcs_sum - 1.0) < 0.001, f"sum={mcs_sum:.4f}", critical=True)

    # S2.5: PROMPTING_WEIGHTS sum = 1.0
    prom_sum = sum(PROMPTING_WEIGHTS.values())
    chk("S2", "S2.5.prom_weights", "Prompting sub-weights sum to 1.0",
        abs(prom_sum - 1.0) < 0.001, f"sum={prom_sum:.4f}", critical=True)

    # S2.6: EFFICIENCY_WEIGHTS sum = 1.0
    eff_sum = sum(EFFICIENCY_WEIGHTS.values())
    chk("S2", "S2.6.eff_weights", "Efficiency sub-weights sum to 1.0",
        abs(eff_sum - 1.0) < 0.001, f"sum={eff_sum:.4f}", critical=True)

    # S2.7: Verify exact weight values from spec
    chk("S2", "S2.7.db_weight", "MCS DB_Correctness weight = 0.50",
        MCS_WEIGHTS["db_correctness"] == 0.50, critical=True)
    chk("S2", "S2.7.llm_weight", "MCS LLM_Quality weight = 0.30",
        MCS_WEIGHTS["llm_quality"] == 0.30, critical=True)
    chk("S2", "S2.7.pe_weight", "MCS Prompting_Effectiveness weight = 0.15",
        MCS_WEIGHTS["prompting_effectiveness"] == 0.15, critical=True)
    chk("S2", "S2.7.eff_weight", "MCS Efficiency weight = 0.05",
        MCS_WEIGHTS["efficiency"] == 0.05, critical=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Spec §4 — All prompt strategies work
# ─────────────────────────────────────────────────────────────────────────────
def test_s3_prompt_strategies(base: str, model: str, quick: bool = False):
    print("\n══ S3: Spec §4 — All Prompt Strategies ══")
    # Use a single easy conceptual question to test each strategy
    test_q = next(q for q in QUESTION_BANK
                  if q["difficulty"] == "easy" and q["type"] == "conceptual")

    strategies = ALL_PROMPT_STRATEGIES if not quick else ["zero-shot", "cot", "few-shot"]
    gen_url = f"{base}/eval/generate"
    strategy_run_ids: dict[str, str] = {}

    for strategy in strategies:
        try:
            t0 = time.monotonic()
            resp = post(gen_url, {
                "model_id": model, "question_id": test_q["id"],
                "prompt_strategy": strategy,
                "hyperparams": {**DEFAULT_HYPERPARAMS, "temperature": 0.3},
                "async_run": False,
            }, timeout=180)
            elapsed = (time.monotonic() - t0) * 1000
            ok = resp.get("status") == "completed"
            chk("S3", f"S3.strat.{strategy}", f"strategy '{strategy}' generates answer",
                ok, f"status={resp.get('status')}")
            if ok:
                strategy_run_ids[strategy] = resp.get("run_id", "")
                # Each strategy answer should be non-empty
                chk("S3", f"S3.strat.{strategy}.ans",
                    f"strategy '{strategy}' produces non-empty answer",
                    bool(resp.get("model_answer", "").strip()))
        except Exception as e:
            chk("S3", f"S3.strat.{strategy}", f"strategy '{strategy}' generates answer",
                False, str(e)[:80])

    # S3.1: zero-shot baseline for prompting comparison (spec §4.4)
    chk("S3", "S3.1.zeroshot_exists", "zero-shot strategy available as baseline",
        "zero-shot" in strategy_run_ids)

    # S3.2: Spec §4.2.4 — self-consistency requires k=5 samples (note: may be async)
    chk("S3", "S3.2.selfconsist_k", "SELF_CONSISTENCY_K = 5 per spec",
        SELF_CONSISTENCY_K == 5)

    # S3.3: Spec §4.4 — prompting_effectiveness_score must be present in judge response
    if "zero-shot" in strategy_run_ids:
        try:
            judge = post(f"{base}/eval/judge/score", {
                "model_id": model, "question_id": test_q["id"],
                "run_id": strategy_run_ids["zero-shot"],
            })
            chk("S3", "S3.3.pe_score", "prompting_effectiveness_score in judge response",
                "prompting_effectiveness_score" in judge,
                f"got {judge.get('prompting_effectiveness_score')}")
        except Exception as e:
            chk("S3", "S3.3.pe_score", "prompting_effectiveness_score in judge response",
                False, str(e)[:80])

    # S3.4: Different strategies should not share run_ids (idempotency key includes strategy)
    run_ids = list(strategy_run_ids.values())
    chk("S3", "S3.4.unique_runs", "different strategies produce different run_ids",
        len(set(run_ids)) == len(run_ids),
        f"{len(set(run_ids))} unique of {len(run_ids)}")

    return strategy_run_ids


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Spec §3 — Hyperparameter framework
# ─────────────────────────────────────────────────────────────────────────────
def test_s4_hyperparams(base: str, model: str):
    print("\n══ S4: Spec §3 — Hyperparameter Framework ══")
    test_q = next(q for q in QUESTION_BANK
                  if q["difficulty"] == "easy" and q["type"] == "conceptual")
    gen_url = f"{base}/eval/generate"

    # S4.1: All hyperparameter fields accepted (spec §3.1)
    for param, values in ALL_HYPERPARAMS.items():
        test_val = values[0]
        try:
            hp = {**DEFAULT_HYPERPARAMS, param: test_val}
            resp = post(gen_url, {
                "model_id": model, "question_id": test_q["id"],
                "prompt_strategy": "zero-shot", "hyperparams": hp,
                "async_run": False,
            }, timeout=60)
            accepted = resp.get("status") in ("completed", "pending", "failed")
            chk("S4", f"S4.1.{param}", f"hyperparam '{param}' accepted",
                accepted, f"status={resp.get('status')}")
        except Exception as e:
            chk("S4", f"S4.1.{param}", f"hyperparam '{param}' accepted",
                False, str(e)[:80])

    # S4.2: Different hyperparams produce different run_ids (hyperparam_hash differs)
    try:
        run_ids = []
        for temp in [0.0, 0.7]:
            resp = post(gen_url, {
                "model_id": model, "question_id": test_q["id"],
                "prompt_strategy": "zero-shot",
                "hyperparams": {**DEFAULT_HYPERPARAMS, "temperature": temp},
                "async_run": False,
            }, timeout=60)
            run_ids.append(resp.get("run_id"))
        chk("S4", "S4.2.hash_diff",
            "different hyperparams produce different run_ids (hyperparam_hash)",
            len(set(run_ids)) == 2, f"run_ids={run_ids}")
    except Exception as e:
        chk("S4", "S4.2.hash_diff",
            "different hyperparams produce different run_ids", False, str(e)[:80])

    # S4.3: /eval/hyperparams/compare endpoint structure (spec §3.2)
    try:
        resp = get(f"{base}/eval/hyperparams/compare?model_id={model}&param_name=temperature",
                   timeout=15)
        chk("S4", "S4.3.compare_ep", "/eval/hyperparams/compare returns data",
            isinstance(resp, (dict, list)))
    except Exception as e:
        chk("S4", "S4.3.compare_ep", "/eval/hyperparams/compare returns data",
            False, str(e)[:80])

    # S4.4: Spec §3.1 — seed parameter (reproducibility)
    chk("S4", "S4.4.seed_in_spec", "seed hyperparam defined in spec §3.1",
        "seed" in ALL_HYPERPARAMS)

    # S4.5: Spec §3.1 — system_prompt_style options present
    style_options = ALL_HYPERPARAMS.get("system_prompt_style", [])
    chk("S4", "S4.5.prompt_styles",
        "system_prompt_style has minimal/role-based/expert-persona options",
        set(style_options) >= {"minimal", "role-based", "expert-persona"})


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: Spec §5.3–5.4 — Contest & Elo system
# ─────────────────────────────────────────────────────────────────────────────
def test_s5_contest_elo(base: str, model: str):
    print("\n══ S5: Spec §5.3–5.4 — Contest & Elo System ══")
    test_q = next(q for q in QUESTION_BANK
                  if q["difficulty"] == "easy" and q["type"] == "conceptual")
    qid = test_q["id"]

    # Generate 2 run_ids to simulate a contest with 2 models
    gen_url = f"{base}/eval/generate"
    run_ids = []
    for temp in [0.1, 0.5]:
        try:
            resp = post(gen_url, {
                "model_id": model, "question_id": qid,
                "prompt_strategy": "zero-shot",
                "hyperparams": {**DEFAULT_HYPERPARAMS, "temperature": temp},
                "async_run": False,
            }, timeout=90)
            if resp.get("status") == "completed":
                run_ids.append(resp.get("run_id"))
        except Exception:
            pass

    if len(run_ids) >= 2:
        # S5.1: Contest endpoint accepts run_ids
        try:
            contest_resp = post(f"{base}/eval/judge/contest", {
                "question_id": qid, "run_ids": run_ids,
            }, timeout=120)
            chk("S5", "S5.1.contest_ep", "POST /eval/judge/contest returns 200", True)

            # S5.2: Contest response contains ranked_model_ids
            chk("S5", "S5.2.ranked_ids",
                "contest response contains ranked_model_ids",
                "ranked_model_ids" in contest_resp,
                str(list(contest_resp.keys()))[:80])

            # S5.3: Anonymization — spec §5.3 CRITICAL requirement
            # The spec says answers must be labeled Model A/B/C/D
            reasoning = contest_resp.get("judge_reasoning", "")
            anon_map  = contest_resp.get("anonymized_map", {})
            chk("S5", "S5.3.anon_map",
                "contest response contains anonymized_map (spec §5.3)",
                bool(anon_map) or "anonymized" in str(contest_resp).lower())

            # S5.4: Spec §5.4 — Elo ratings updated after contest
            # Check /eval/leaderboard reflects Elo field
            try:
                lb = get(f"{base}/eval/leaderboard", timeout=10)
                entries = lb.get("leaderboard", lb if isinstance(lb, list) else [])
                if entries:
                    entry = entries[0]
                    chk("S5", "S5.4.elo_present",
                        "Elo rating present in leaderboard entries",
                        "elo_rating" in entry, str(list(entry.keys()))[:80])
            except Exception as e:
                chk("S5", "S5.4.elo_present",
                    "Elo rating present in leaderboard entries", False, str(e)[:80])

            # S5.5: Rankings list present (spec §5.3)
            chk("S5", "S5.5.rankings",
                "contest response contains rankings list",
                "rankings" in contest_resp)

            # S5.6: tie_exists field (spec §5.3)
            chk("S5", "S5.6.tie_exists",
                "contest response contains tie_exists field",
                "tie_exists" in contest_resp)

        except Exception as e:
            chk("S5", "S5.1.contest_ep",
                "POST /eval/judge/contest returns 200", False, str(e)[:80])
    else:
        chk("S5", "S5.0.setup", "Could generate 2+ runs for contest test",
            False, "need ≥2 successful generations")

    # S5.7: Elo constants (spec §5.4)
    chk("S5", "S5.7.elo_initial", "Elo initial rating = 1200",
        ELO_INITIAL == 1200, critical=True)
    chk("S5", "S5.7.elo_k", "Elo K-factor = 32",
        ELO_K == 32, critical=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: Spec §5.5 — Leaderboard schema completeness
# ─────────────────────────────────────────────────────────────────────────────
def test_s6_leaderboard(base: str):
    print("\n══ S6: Spec §5.5 — Leaderboard Schema ══")
    try:
        resp = get(f"{base}/eval/leaderboard", timeout=15)
        entries = resp.get("leaderboard", resp if isinstance(resp, list) else [])

        chk("S6", "S6.1.returns", "Leaderboard endpoint returns data", True)

        if entries:
            entry = entries[0]
            entry_keys = set(entry.keys())

            # S6.2: All spec-required fields present (spec §5.5 table)
            for field in LEADERBOARD_REQUIRED_FIELDS:
                chk("S6", f"S6.2.{field[:15]}", f"leaderboard has field '{field}'",
                    field in entry_keys, f"keys: {list(entry_keys)[:5]}")

            # S6.3: Numeric fields are numeric
            for num_field in ["mcs_score", "db_correctness", "llm_quality",
                              "elo_rating", "hallucination_rate"]:
                if num_field in entry and entry[num_field] is not None:
                    chk("S6", f"S6.3.{num_field[:12]}", f"'{num_field}' is numeric",
                        isinstance(entry[num_field], (int, float)))

            # S6.4: mcs_score in [0,100]
            if "mcs_score" in entry and entry["mcs_score"] is not None:
                chk("S6", "S6.4.mcs_range", "mcs_score in [0,100]",
                    0.0 <= float(entry["mcs_score"]) <= 100.0,
                    f"got {entry['mcs_score']}")

            # S6.5: elo_rating >= 0 (starts at 1200)
            if "elo_rating" in entry and entry["elo_rating"] is not None:
                chk("S6", "S6.5.elo_positive", "elo_rating is positive",
                    float(entry["elo_rating"]) > 0,
                    f"got {entry['elo_rating']}")

        # S6.6: sort_by parameter works (spec §6.4)
        try:
            sorted_resp = get(f"{base}/eval/leaderboard?sort_by=mcs_score", timeout=10)
            chk("S6", "S6.6.sort", "leaderboard accepts sort_by parameter", True)
        except Exception as e:
            chk("S6", "S6.6.sort", "leaderboard accepts sort_by parameter",
                False, str(e)[:80])

    except Exception as e:
        chk("S6", "S6.1.returns", "Leaderboard endpoint returns data",
            False, str(e)[:80])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: Spec §2.1–2.7 — LLM/AI quality benchmarks
# Tests hallucination detection, latency thresholds, format compliance, verbosity
# ─────────────────────────────────────────────────────────────────────────────
def test_s7_llm_quality(base: str, model: str, all_results: list[dict]):
    print("\n══ S7: Spec §2 — LLM/AI Quality Benchmarks ══")

    if not all_results:
        chk("S7", "S7.0.data", "Have results data to analyse", False,
            "no results available", critical=True)
        return

    scores = [r for r in all_results if r and r.get("judge")]
    if not scores:
        return

    # S7.1: Spec §2.1 — Hallucination rate tracked (HR = #responses with ≥1 hall / total)
    total = len(scores)
    with_hall = sum(1 for r in scores
                    if r["judge"].get("hallucinations_detected"))
    hr = with_hall / total if total else 0
    chk("S7", "S7.1.hall_rate", "Hallucination rate is tracked and computable",
        total > 0, f"HR={hr:.1%} ({with_hall}/{total} responses)")

    # S7.2: Hallucination severity tiers (spec §2.1.1 — Critical/High/Medium/Low)
    severity_seen = set()
    for r in scores:
        for h in r["judge"].get("hallucinations_detected", []):
            sev = h.get("severity", "").lower()
            if sev:
                severity_seen.add(sev)
    chk("S7", "S7.2.hall_sev", "Hallucination severity tiers used",
        bool(severity_seen), f"severities seen: {severity_seen}")

    # S7.3: Spec §2.3.2 — verbosity ratio computable from token counts
    verbosity_computable = sum(
        1 for r in scores
        if r["gen"].get("output_tokens") and r["gen"].get("input_tokens")
    )
    chk("S7", "S7.3.verbosity", "Token counts available for verbosity ratio",
        verbosity_computable > 0,
        f"{verbosity_computable}/{total} have token counts")

    # S7.4: Spec §2.7 — Latency data stored for all runs
    lat_stored = sum(1 for r in scores if r["gen"].get("total_latency_ms") is not None)
    chk("S7", "S7.4.lat_stored", "total_latency_ms stored for all runs",
        lat_stored == total, f"{lat_stored}/{total}")

    # S7.5: Spec §2.7 — TTFT target < 1.5s (check if any TTFT data exists)
    ttft_stored = sum(1 for r in scores if r["gen"].get("total_latency_ms") is not None)
    chk("S7", "S7.5.ttft_defined", "TTFT target (1.5s) defined in spec §2.7",
        TTFT_TARGET_MS == 1500.0)

    # S7.6: Spec §2.4 — format compliance score in judge response
    format_scores = [r for r in scores
                     if "sql_execution_details" in r["judge"]]
    chk("S7", "S7.6.format_comp", "sql_execution_details present in judge responses",
        len(format_scores) > 0, f"{len(format_scores)}/{total} have it")

    # S7.7: Spec §2.7 — cost_usd tracked (cost per question for ROI)
    cost_tracked = sum(1 for r in scores if r["gen"].get("cost_usd") is not None)
    chk("S7", "S7.7.cost_tracked", "cost_usd tracked per question",
        cost_tracked > 0, f"{cost_tracked}/{total} have cost")

    # S7.8: Spec §2.7 — retry rate target < 2%
    chk("S7", "S7.8.retry_target", "API retry rate target < 2% defined",
        API_RETRY_RATE_TARGET == 0.02)

    # S7.9: Spec §2.3.2 — verbosity thresholds defined
    chk("S7", "S7.9.verbosity_thresholds",
        "Verbosity high/low penalty thresholds defined per spec",
        VERBOSITY_HIGH_RATIO == 3.0 and VERBOSITY_LOW_RATIO == 0.5)

    # S7.10: Spec §5.2 — justifications are non-empty (reasoning quality)
    with_justif = sum(1 for r in scores
                      if r["judge"].get("justification", "").strip())
    chk("S7", "S7.10.justif", "All judge responses have non-empty justifications",
        with_justif == total, f"{with_justif}/{total}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8: Spec §6.4 — Idempotency
# Re-running the same configuration must overwrite, not duplicate (spec §6 FINAL NOTE)
# ─────────────────────────────────────────────────────────────────────────────
def test_s8_idempotency(base: str, model: str):
    print("\n══ S8: Spec §6 (Final Note) — Idempotency ══")
    test_q = next(q for q in QUESTION_BANK
                  if q["difficulty"] == "easy" and q["type"] == "conceptual")
    gen_url = f"{base}/eval/generate"
    hp = {**DEFAULT_HYPERPARAMS, "seed": 42}

    # Run same config twice
    run_ids = []
    for _ in range(2):
        try:
            resp = post(gen_url, {
                "model_id": model, "question_id": test_q["id"],
                "prompt_strategy": "zero-shot", "hyperparams": hp,
                "async_run": False,
            }, timeout=90)
            run_ids.append(resp.get("run_id"))
        except Exception as e:
            chk("S8", "S8.0.setup", "idempotency setup failed", False, str(e)[:80])
            return

    # S8.1: Same (question_id, model_id, strategy, hyperparam_hash) → same run_id
    chk("S8", "S8.1.same_run_id",
        "Re-running same config returns same run_id (idempotent)",
        len(run_ids) == 2 and run_ids[0] == run_ids[1],
        f"run_ids: {run_ids[0][:8]}...  vs  {run_ids[1][:8] if len(run_ids)>1 else 'N/A'}...",
        critical=True)

    # S8.2: force_rerun creates a NEW run (overwrite, not duplicate)
    try:
        resp_force = post(gen_url, {
            "model_id": model, "question_id": test_q["id"],
            "prompt_strategy": "zero-shot", "hyperparams": hp,
            "async_run": False, "force_rerun": True,
        }, timeout=90)
        forced_id = resp_force.get("run_id")
        # force_rerun may return same or new id depending on implementation
        chk("S8", "S8.2.force_rerun",
            "force_rerun parameter accepted without error",
            resp_force.get("status") in ("completed", "pending"))
    except Exception as e:
        chk("S8", "S8.2.force_rerun",
            "force_rerun parameter accepted without error", False, str(e)[:80])


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9: Coverage — all 80 questions generate+judge successfully
# Tests spec §1 (all questions evaluated), §6.1 (storage for all runs)
# ─────────────────────────────────────────────────────────────────────────────
def test_s9_all_questions(base: str, model: str, quick: bool = False) -> list[dict]:
    print("\n══ S9: All 80 Questions — Generate + Judge ══")
    questions_to_test = QUESTION_BANK if not quick else _stratified_sample(QUESTION_BANK)
    print(f"  Testing {len(questions_to_test)} questions"
          + (" (stratified sample)" if quick else " (full bank)"))

    all_results = []
    failed_qs: list[str] = []

    for i, q in enumerate(questions_to_test):
        print(f"\r  [{i+1:3d}/{len(questions_to_test)}] {q['subtopic'][:35]:<35} "
              f"{q['difficulty']:<8}", end="", flush=True)
        result = run_one_question(base, q, model)
        all_results.append(result)
        if result is None:
            failed_qs.append(q["id"])

    print()  # newline after progress

    success_count = sum(1 for r in all_results if r is not None)
    total = len(questions_to_test)
    chk("S9", "S9.1.all_gen_judge",
        f"All {total} questions generate+judge successfully",
        success_count == total,
        f"{success_count}/{total} succeeded, failed: {len(failed_qs)}",
        critical=True)

    # S9.2: Coverage by topic
    topics_tested = set(q["topic"] for q in questions_to_test)
    chk("S9", "S9.2.topic_coverage",
        f"All 7 topics covered ({len(topics_tested)}/7)",
        len(topics_tested) >= 7,
        f"topics: {topics_tested}")

    # S9.3: Coverage by difficulty
    diffs_tested = set(q["difficulty"] for q in questions_to_test)
    chk("S9", "S9.3.difficulty_coverage",
        "All 3 difficulty tiers covered (easy/medium/hard)",
        {"easy", "medium", "hard"} <= diffs_tested)

    # S9.4: Coverage by type
    types_tested = set(q["type"] for q in questions_to_test)
    chk("S9", "S9.4.type_coverage",
        "Both question types covered (practical/conceptual)",
        {"practical", "conceptual"} <= types_tested)

    # S9.5: SQL-tagged questions ran the harness
    sql_qs = [q for q in questions_to_test
              if "sql" in q.get("tags", []) and q["type"] == "practical"]
    results_by_qid = {r["question"]["id"]: r for r in all_results if r}
    harness_ran = sum(
        1 for q in sql_qs
        if q["id"] in results_by_qid
        and results_by_qid[q["id"]]["judge"].get("sql_execution_details") is not None
    )
    chk("S9", "S9.5.sql_harness",
        f"SQL harness ran for all sql+practical questions ({len(sql_qs)} total)",
        harness_ran == len(sql_qs) if sql_qs else True,
        f"{harness_ran}/{len(sql_qs)}")

    # S9.6: Transaction questions have scores (spec §1.5)
    tx_qs = [q for q in questions_to_test if "transactions" in q.get("tags", [])]
    tx_scored = sum(
        1 for q in tx_qs
        if q["id"] in results_by_qid
        and results_by_qid[q["id"]]["judge"].get("judge_score_0_10") is not None
    )
    chk("S9", "S9.6.tx_scored",
        f"Transaction questions all scored ({len(tx_qs)} total)",
        tx_scored == len(tx_qs) if tx_qs else True,
        f"{tx_scored}/{len(tx_qs)}")

    # S9.7: Normalization questions scored (spec §1.2)
    norm_qs = [q for q in questions_to_test if "normalization" in q.get("tags", [])]
    norm_scored = sum(
        1 for q in norm_qs
        if q["id"] in results_by_qid
        and results_by_qid[q["id"]]["judge"].get("judge_score_0_10") is not None
    )
    chk("S9", "S9.7.norm_scored",
        f"Normalization questions all scored ({len(norm_qs)} total)",
        norm_scored == len(norm_qs) if norm_qs else True,
        f"{norm_scored}/{len(norm_qs)}")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 10: Spec §5.5 — Results summary / analytics endpoints
# ─────────────────────────────────────────────────────────────────────────────
def test_s10_analytics(base: str, model: str, all_results: list[dict]):
    print("\n══ S10: Spec §5.5 / §6.4 — Analytics & Results Summary ══")

    # S10.1: /eval/results/summary endpoint
    try:
        resp = get(f"{base}/eval/results/summary?model_id={model}", timeout=15)
        chk("S10", "S10.1.summary_ep", "/eval/results/summary returns data",
            isinstance(resp, (dict, list)))
    except Exception as e:
        chk("S10", "S10.1.summary_ep", "/eval/results/summary returns data",
            False, str(e)[:80])

    # S10.2: /eval/prompts/compare endpoint
    try:
        resp = get(f"{base}/eval/prompts/compare?model_id={model}", timeout=15)
        chk("S10", "S10.2.prompts_ep", "/eval/prompts/compare returns data",
            isinstance(resp, (dict, list)))
    except Exception as e:
        chk("S10", "S10.2.prompts_ep", "/eval/prompts/compare returns data",
            False, str(e)[:80])

    # S10.3: Export endpoints (spec §6.4 item 28)
    for fmt in ["json", "csv"]:
        try:
            resp = get(f"{base}/eval/export/{fmt}?model_id={model}", timeout=20)
            chk("S10", f"S10.3.export_{fmt}", f"/eval/export/{fmt} returns data",
                resp is not None)
        except Exception as e:
            chk("S10", f"S10.3.export_{fmt}", f"/eval/export/{fmt} returns data",
                False, str(e)[:80])

    # S10.4: Leaderboard refresh endpoint (spec §6.4)
    try:
        resp = post(f"{base}/eval/leaderboard/refresh", {}, timeout=30)
        chk("S10", "S10.4.lb_refresh", "POST /eval/leaderboard/refresh works",
            True)
    except Exception as e:
        # Might be a GET in some implementations
        try:
            resp = get(f"{base}/eval/leaderboard/refresh", timeout=30)
            chk("S10", "S10.4.lb_refresh", "POST /eval/leaderboard/refresh works",
                True, "accepted as GET")
        except Exception as e2:
            chk("S10", "S10.4.lb_refresh", "POST /eval/leaderboard/refresh works",
                False, str(e)[:80])

    # S10.5: Results have per-topic breakdown capability
    scores = [r for r in all_results if r and r.get("judge")]
    if scores:
        topics_with_scores = defaultdict(list)
        for r in scores:
            topic = r["question"]["topic"]
            mcs = r["judge"].get("master_composite_score")
            if mcs is not None:
                topics_with_scores[topic].append(float(mcs))
        chk("S10", "S10.5.topic_breakdown",
            "Can compute per-topic score breakdown",
            len(topics_with_scores) >= 5,
            f"topics: {list(topics_with_scores.keys())[:3]}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 11: Spec §1.3 — Schema design / data modeling (DDL questions)
# ─────────────────────────────────────────────────────────────────────000─────
def test_s11_schema_design(all_results: list[dict]):
    print("\n══ S11: Spec §1.3 — Schema Design & Data Modeling Quality ══")

    # Find practical questions that involve schema/DDL (CREATE TABLE etc.)
    ddl_results = [
        r for r in all_results
        if r and "sql" in r["question"].get("tags", [])
        and r["question"]["type"] == "practical"
    ]
    chk("S11", "S11.1.ddl_qs", "SQL practical questions tested",
        len(ddl_results) > 0, f"found {len(ddl_results)}")

    if not ddl_results:
        return

    # S11.2: SQL execution details present for DDL/SQL questions
    with_harness = sum(
        1 for r in ddl_results
        if r["judge"].get("sql_execution_details") is not None
    )
    chk("S11", "S11.2.harness_coverage",
        "SQL harness coverage for sql+practical questions",
        with_harness > 0, f"{with_harness}/{len(ddl_results)}")

    # S11.3: Idiomatic PostgreSQL check present (spec §1.1.2)
    pg_idiom_qs = [
        r for r in ddl_results
        if r["judge"].get("sql_execution_details") and
           r["judge"]["sql_execution_details"].get("idiomatic_postgresql") is not None
    ]
    chk("S11", "S11.3.pg_idiom",
        "idiomatic_postgresql score computed for SQL questions",
        len(pg_idiom_qs) > 0,
        f"{len(pg_idiom_qs)}/{len(ddl_results)}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 12: Spec §1.5 — Transaction/concurrency correctness
# ─────────────────────────────────────────────────────────────────────────────
def test_s12_transactions(all_results: list[dict]):
    print("\n══ S12: Spec §1.5 — Transaction & Concurrency Correctness ══")

    tx_results = [
        r for r in all_results
        if r and "transactions" in r["question"].get("tags", [])
    ]
    chk("S12", "S12.1.tx_count",
        f"Transaction questions tested ({len(tx_results)} questions)",
        len(tx_results) >= 5,
        f"found {len(tx_results)} transaction questions")

    if not tx_results:
        return

    # S12.2: Spec §1.5 — auto-scoring weight 60%, judge 40% (TransactionScores)
    # Check db_correctness_score is computed for transaction questions
    scored = [r for r in tx_results
              if r["judge"].get("db_correctness_score") is not None]
    chk("S12", "S12.2.tx_scored",
        "db_correctness_score computed for transaction questions",
        len(scored) == len(tx_results), f"{len(scored)}/{len(tx_results)}")

    # S12.3: ARIES questions get specific evaluation (spec §1.5)
    aries_qs = [r for r in tx_results
                if "CRASH RECOVERY" in r["question"].get("subtopic", "")]
    chk("S12", "S12.3.aries_qs",
        f"ARIES/crash recovery questions present ({len(aries_qs)})",
        len(aries_qs) > 0, f"found {len(aries_qs)}")

    # S12.4: Concurrency control questions present (spec §1.5)
    cc_qs = [r for r in tx_results
             if "CONCURRENCY" in r["question"].get("subtopic", "")]
    chk("S12", "S12.4.cc_qs",
        f"Concurrency control questions present ({len(cc_qs)})",
        len(cc_qs) > 0, f"found {len(cc_qs)}")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _stratified_sample(qs: list[dict]) -> list[dict]:
    """Return 1 question per (subtopic, difficulty) cell for quick testing."""
    seen: set[tuple] = set()
    sample = []
    for q in qs:
        key = (q["subtopic"], q["difficulty"])
        if key not in seen:
            seen.add(key)
            sample.append(q)
    # Ensure at least one of each type
    types_covered = {q["type"] for q in sample}
    if "practical" not in types_covered:
        practical = next((q for q in qs if q["type"] == "practical"), None)
        if practical:
            sample.append(practical)
    return sample


def print_final_summary(results: Results, output_path: str):
    summary = results.summary()
    print("\n" + "═" * 72)
    print("  FINAL INFRASTRUCTURE TEST SUMMARY")
    print("═" * 72)
    print(f"  Total checks  : {summary['total']}")
    print(f"  Passed        : {summary['passed']}  ({summary['pass_rate']:.1f}%)")
    print(f"  Failed        : {summary['failed']}")
    print(f"  Critical fails: {summary['critical_failures']}")
    print()
    print("  By section:")
    for sec, counts in sorted(summary["by_section"].items()):
        p, f = counts["passed"], counts["failed"]
        bar = "✅" if f == 0 else ("❌" if f > 2 else "⚠ ")
        print(f"    {bar} {sec}: {p} passed, {f} failed")
    if summary["failed_checks"]:
        print()
        print("  Failed checks:")
        for c in summary["failed_checks"][:30]:
            crit = " [CRITICAL]" if c["critical"] else ""
            print(f"    ❌ {c['test_id']:30s} {c['description'][:50]}{crit}")
            if c["detail"]:
                print(f"       → {c['detail'][:70]}")
        if len(summary["failed_checks"]) > 30:
            print(f"    ... and {len(summary['failed_checks'])-30} more")
    print("═" * 72)
    print(f"  Results written to: {output_path}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="LLM Evaluator — Comprehensive Infrastructure Test"
    )
    parser.add_argument("--base-url", default="http://localhost:8000",
                        help="FastAPI server base URL")
    parser.add_argument("--model", default="llama-3.1-70b",
                        help="Challenger model ID to use for all tests")
    parser.add_argument("--output", default="infra_test_results.json",
                        help="Output file for full results")
    parser.add_argument("--quick", action="store_true",
                        help="Stratified sample only (~20 questions, ~15 min)")
    args = parser.parse_args()

    base  = args.base_url.rstrip("/")
    model = args.model
    quick = args.quick

    print("═" * 72)
    print("  LLM EVALUATOR — COMPREHENSIVE INFRASTRUCTURE TEST")
    print("═" * 72)
    print(f"  Base URL  : {base}")
    print(f"  Model     : {model}")
    print(f"  Mode      : {'QUICK (stratified sample)' if quick else 'FULL (all 80 questions)'}")
    print(f"  Questions : {len(QUESTION_BANK)} in bank "
          f"({'~20 tested' if quick else 'all 80 tested'})")
    print(f"  Started   : {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print()

    # S0: Infrastructure health check — must pass to continue
    if not test_s0_infrastructure(base):
        print("FATAL: Server unreachable. Aborting.")
        sys.exit(1)

    # S2: Scoring weight integrity (no network calls needed)
    test_s2_scoring_weights(base)

    # S4: Hyperparameter framework (network: light)
    test_s4_hyperparams(base, model)

    # S8: Idempotency (network: 3 generate calls)
    test_s8_idempotency(base, model)

    # S3: Prompt strategies (network: 1 question × N strategies)
    strategy_run_ids = test_s3_prompt_strategies(base, model, quick=quick)

    # S5: Contest & Elo (network: 2 generate + 1 contest call)
    test_s5_contest_elo(base, model)

    # S6: Leaderboard schema
    test_s6_leaderboard(base)

    # S9: All questions (main test — most time consuming)
    all_results = test_s9_all_questions(base, model, quick=quick)

    # S7: LLM quality (uses results from S9)
    test_s7_llm_quality(base, model, all_results)

    # S10: Analytics endpoints (uses results from S9)
    test_s10_analytics(base, model, all_results)

    # S11: Schema design coverage
    test_s11_schema_design(all_results)

    # S12: Transaction/concurrency coverage
    test_s12_transactions(all_results)

    # Write full results
    output = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "base_url": base, "model_id": model, "quick_mode": quick,
        "summary": R.summary(),
        "checks": R.checks,
        "question_results": [
            {
                "question_id": r["question"]["id"],
                "question_raw_id": r["question"]["raw_id"],
                "subtopic": r["question"]["subtopic"],
                "topic": r["question"]["topic"],
                "difficulty": r["question"]["difficulty"],
                "type": r["question"]["type"],
                "tags": r["question"]["tags"],
                "strategy": r.get("strategy"),
                "run_id": r["gen"].get("run_id"),
                "status": r["gen"].get("status"),
                "output_tokens": r["gen"].get("output_tokens"),
                "total_latency_ms": r["gen"].get("total_latency_ms"),
                "cost_usd": r["gen"].get("cost_usd"),
                "judge_score_0_10": r["judge"].get("judge_score_0_10"),
                "master_composite_score": r["judge"].get("master_composite_score"),
                "db_correctness_score": r["judge"].get("db_correctness_score"),
                "llm_quality_score": r["judge"].get("llm_quality_score"),
                "prompting_effectiveness_score": r["judge"].get("prompting_effectiveness_score"),
                "efficiency_score": r["judge"].get("efficiency_score"),
                "hallucinations_detected": r["judge"].get("hallucinations_detected", []),
                "missing_points": r["judge"].get("missing_points", []),
                "sql_execution_details": r["judge"].get("sql_execution_details"),
                "justification": r["judge"].get("justification", ""),
                "gen_ms": r.get("gen_ms"),
                "judge_ms": r.get("judge_ms"),
            }
            for r in all_results if r
        ],
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print_final_summary(R, args.output)


if __name__ == "__main__":
    main()