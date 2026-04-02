"""
evaluators/hallucination.py — Hallucination detection pipeline.

Implements the multi-tier detection strategy from Section 2.1:

  Tier 1 — Fabricated SQL function    → Regex + PostgreSQL catalog check
  Tier 2 — Wrong theorem attribution  → Ground truth comparison (LLM judge)
  Tier 3 — Invented constraint type   → PG schema validation
  Tier 4 — Contradictory claim        → LLM judge (logic check)
  Tier 5 — Plausible but wrong cost   → Ground truth comparison
  Tier 6 — Over-generalisation        → LLM judge

Each detection method is independent and results are merged into a single
hallucination report with severity tagging.
"""

from __future__ import annotations

import re
from typing import Optional


# ── Known PostgreSQL built-in functions and constructs ────────────────────────
# A representative list of real PostgreSQL functions. Any SQL function
# mentioned in a model's answer that is NOT in this list is a candidate
# for "fabricated function" hallucination.
KNOWN_PG_FUNCTIONS = frozenset({
    # Aggregate
    "count", "sum", "avg", "min", "max", "array_agg", "string_agg",
    "json_agg", "jsonb_agg", "json_object_agg", "bool_and", "bool_or",
    "every", "bit_and", "bit_or", "percentile_cont", "percentile_disc",
    "mode", "corr", "covar_pop", "covar_samp", "regr_slope", "regr_intercept",
    "stddev", "stddev_pop", "stddev_samp", "variance", "var_pop", "var_samp",
    # Window
    "row_number", "rank", "dense_rank", "percent_rank", "cume_dist",
    "ntile", "lag", "lead", "first_value", "last_value", "nth_value",
    # String
    "lower", "upper", "length", "substr", "substring", "trim", "ltrim",
    "rtrim", "lpad", "rpad", "replace", "split_part", "regexp_replace",
    "regexp_match", "regexp_matches", "strpos", "position", "overlay",
    "left", "right", "concat", "concat_ws", "format", "md5", "initcap",
    "translate", "ascii", "chr", "repeat", "reverse",
    # Numeric
    "abs", "ceil", "ceiling", "floor", "round", "trunc", "sign",
    "power", "sqrt", "exp", "ln", "log", "mod", "div", "random",
    "setseed", "pi", "degrees", "radians", "sin", "cos", "tan",
    # Date/Time
    "now", "current_timestamp", "current_date", "current_time", "localtime",
    "localtimestamp", "age", "date_part", "date_trunc", "extract",
    "to_timestamp", "to_date", "to_char", "make_date", "make_time",
    "make_interval", "clock_timestamp", "statement_timestamp", "timeofday",
    "justify_days", "justify_hours", "justify_interval",
    # JSON
    "json_build_object", "jsonb_build_object", "json_build_array",
    "jsonb_build_array", "json_extract_path", "jsonb_extract_path",
    "json_extract_path_text", "jsonb_extract_path_text",
    "json_array_elements", "jsonb_array_elements", "json_typeof",
    "jsonb_typeof", "json_strip_nulls", "jsonb_strip_nulls",
    "jsonb_set", "jsonb_insert", "jsonb_delete", "to_json", "to_jsonb",
    "row_to_json",
    # Array
    "array_length", "array_dims", "array_upper", "array_lower",
    "array_cat", "array_append", "array_prepend", "array_to_string",
    "string_to_array", "unnest", "cardinality", "array_remove",
    "array_replace", "array_position", "array_positions",
    # Type conversion / casting
    "cast", "to_number", "to_char",
    # Control flow
    "coalesce", "nullif", "greatest", "least", "case",
    # Text search
    "to_tsvector", "to_tsquery", "plainto_tsquery", "phraseto_tsquery",
    "websearch_to_tsquery", "ts_rank", "ts_rank_cd", "ts_headline",
    # System
    "pg_size_pretty", "pg_relation_size", "pg_total_relation_size",
    "pg_table_size", "pg_indexes_size", "pg_column_size",
    "current_user", "session_user", "current_schema", "current_database",
    "version", "pg_backend_pid", "pg_postmaster_start_time",
    # Geometric / Range / Misc
    "generate_series", "generate_subscripts", "array_fill",
    "row", "nextval", "currval", "lastval", "setval",
})

KNOWN_PG_CONSTRAINTS = frozenset({
    "primary key", "foreign key", "unique", "not null", "check",
    "default", "references", "exclude", "deferrable", "initially deferred",
    "initially immediate",
})

# DBMS facts that are commonly hallucinated with incorrect attributions
KNOWN_FACTS = {
    "3nf was proposed by codd": "3NF was introduced by Edgar F. Codd in 1971",
    "bcnf was defined by codd and boyce": "BCNF was defined by Codd and Boyce in 1974",
    "2pl guarantees no deadlock": False,   # 2PL does NOT guarantee freedom from deadlock
    "serializable means no anomalies": True,
    "acid stands for atomicity consistency isolation durability": True,
    "b+ tree insertion is o(1)": False,    # It is O(log n)
    "hash index supports range queries efficiently": False,
}


# ── Detection Functions ───────────────────────────────────────────────────────
def detect_fabricated_sql_functions(answer_text: str) -> list[dict]:
    """
    Tier 1: Extract SQL function calls from the answer and flag any that are
    not in the known PostgreSQL function registry.
    """
    hallucinations = []
    # Match word immediately before (
    function_pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.IGNORECASE)
    matches = function_pattern.findall(answer_text)

    # Exclude common SQL keywords that appear with parens
    sql_keywords = {
        "if", "case", "when", "then", "else", "end", "select", "from",
        "where", "as", "on", "in", "not", "and", "or", "like", "between",
        "exists", "all", "any", "some",
    }

    for func_name in set(matches):
        lower = func_name.lower()
        if lower in sql_keywords:
            continue
        if lower not in KNOWN_PG_FUNCTIONS:
            # Check if it looks like a genuine UDF or PL/pgSQL function name
            # (contains underscore and uses common naming patterns)
            if not _looks_like_user_defined(func_name):
                hallucinations.append({
                    "type": "fabricated_sql_function",
                    "text": f"Unknown PostgreSQL function: {func_name}(…)",
                    "severity": "critical",
                    "detected_by": "regex+pg_catalog",
                })
    return hallucinations


def _looks_like_user_defined(name: str) -> bool:
    """Heuristic: if a function name looks like a user-defined procedure, don't flag it."""
    # User-defined functions typically have descriptive snake_case names
    # We flag only names that look like they're claiming to be built-in PostgreSQL functions
    suspicious_patterns = [
        r"^pg_\w+",          # Claims to be a pg_ system function
        r"^[A-Z]{2,}_\w+",   # ALL_CAPS prefix typical of invented names
    ]
    return not any(re.match(p, name, re.IGNORECASE) for p in suspicious_patterns)


def detect_invented_constraints(answer_text: str) -> list[dict]:
    """
    Tier 3: Detect references to non-existent PostgreSQL constraint types.
    """
    hallucinations = []
    # Look for 'KEY', 'CONSTRAINT', or 'INDEX' preceded by unusual qualifiers
    invented_patterns = [
        r"\bTEMPORAL\s+KEY\b",
        r"\bTIME\s+KEY\b",
        r"\bIMMUTABLE\s+CONSTRAINT\b",
        r"\bDYNAMIC\s+CONSTRAINT\b",
        r"\bVIRTUAL\s+KEY\b",
        r"\bCOMPUTED\s+CONSTRAINT\b",
        r"\bSEQUENTIAL\s+INDEX\b",
    ]
    for pattern in invented_patterns:
        if re.search(pattern, answer_text, re.IGNORECASE):
            hallucinations.append({
                "type": "invented_constraint_type",
                "text": f"Non-existent PostgreSQL construct matching: {pattern}",
                "severity": "high",
                "detected_by": "regex",
            })
    return hallucinations


def detect_known_wrong_facts(answer_text: str) -> list[dict]:
    """
    Tier 2 & 5: Check for commonly hallucinated DBMS facts that can be
    detected with simple string matching against a known-wrong-fact registry.
    """
    hallucinations = []
    text_lower = answer_text.lower()

    wrong_fact_patterns = [
        (
            r"2pl\s+guarantees?\s+(?:no|freedom\s+from)\s+deadlock",
            "2PL does NOT guarantee freedom from deadlock; it only guarantees "
            "conflict-serializability. Deadlocks are still possible.",
            "high",
        ),
        (
            r"b[\+\-]?[\s-]tree\s+insertion\s+is\s+o\(1\)",
            "B+ tree insertion is O(log n), not O(1).",
            "medium",
        ),
        (
            r"hash\s+index\s+(?:supports?|handles?)\s+range\s+quer",
            "Hash indexes do NOT support range queries efficiently; "
            "they are designed for equality lookups only.",
            "high",
        ),
        (
            r"aries\s+(?:is\s+used\s+by\s+all|all\s+dbms\s+use)",
            "Not all DBMS systems use ARIES for crash recovery; it is one "
            "of several recovery algorithms.",
            "low",
        ),
        (
            r"repeatable\s+read\s+prevents?\s+phantom\s+read",
            "In PostgreSQL, REPEATABLE READ prevents phantom reads (unlike "
            "the SQL standard), but this is PostgreSQL-specific behaviour.",
            "medium",
        ),
        (
            r"3nf\s+was\s+(?:defined|proposed|introduced)\s+(?:by\s+)?(?:codd\s+in\s+1985|in\s+1985)",
            "3NF was defined by Codd in 1971, not 1985.",
            "high",
        ),
    ]

    for pattern, description, severity in wrong_fact_patterns:
        if re.search(pattern, text_lower):
            hallucinations.append({
                "type": "wrong_fact",
                "text": description,
                "severity": severity,
                "detected_by": "ground_truth_comparison",
            })

    return hallucinations


def run_automated_hallucination_checks(answer_text: str) -> list[dict]:
    """
    Run all automated (non-LLM) hallucination checks and return the combined
    list of detected hallucinations.
    """
    hallucinations: list[dict] = []
    if not answer_text or not answer_text.strip():
        return hallucinations

    hallucinations.extend(detect_fabricated_sql_functions(answer_text))
    hallucinations.extend(detect_invented_constraints(answer_text))
    hallucinations.extend(detect_known_wrong_facts(answer_text))

    return hallucinations


def compute_hallucination_metrics(hallucinations: list[dict]) -> dict:
    """
    Compute hallucination rate flag and severity score from a list of
    detected hallucination dicts.

    Returns:
        {
          "has_hallucination": bool,
          "count": int,
          "severity_score": float 0–1,
          "critical_count": int,
          "high_count": int,
        }
    """
    from config import HALLUCINATION_SEVERITY

    if not hallucinations:
        return {
            "has_hallucination": False,
            "count": 0,
            "severity_score": 0.0,
            "critical_count": 0,
            "high_count": 0,
        }

    severity_totals = {s: 0 for s in HALLUCINATION_SEVERITY}
    total_weight = 0.0
    max_possible = len(hallucinations) * HALLUCINATION_SEVERITY["critical"]

    for h in hallucinations:
        s = h.get("severity", "low")
        weight = HALLUCINATION_SEVERITY.get(s, 0.5)
        total_weight += weight
        if s in severity_totals:
            severity_totals[s] += 1

    severity_score = min(total_weight / max_possible, 1.0) if max_possible > 0 else 0.0

    return {
        "has_hallucination": True,
        "count": len(hallucinations),
        "severity_score": severity_score,
        "critical_count": severity_totals.get("critical", 0),
        "high_count": severity_totals.get("high", 0),
    }
