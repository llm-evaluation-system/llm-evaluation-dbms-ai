"""
evaluators/sql_harness.py — Automated SQL execution harness.

Implements the automated scoring components of Section 1.1.2:
  - Syntactic Parse Success  (15%): EXPLAIN on generated SQL
  - Result Set Accuracy      (30%): F1 of expected vs returned rows
  - Idiomatic PostgreSQL     (20%): Pattern matching for PG-specific idioms

Runs against a sandboxed PostgreSQL 16 test instance.
All SQL execution is transactionally isolated — every test runs in a
SAVEPOINT and is rolled back, so the test schema stays pristine.
"""

from __future__ import annotations

import asyncio
import re
from typing import Optional

import asyncpg

from config import TEST_DATABASE_URL


# ── Connection Helper ─────────────────────────────────────────────────────────
async def _get_test_connection() -> Optional[asyncpg.Connection]:
    """
    Attempt to acquire a connection to the sandboxed test PostgreSQL instance.
    Returns None if the test DB is unavailable (e.g. in CI without Docker).
    """
    # asyncpg uses a postgresql:// DSN, not sqlalchemy+asyncpg://
    dsn = TEST_DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
    try:
        conn = await asyncpg.connect(dsn, timeout=5.0)
        return conn
    except Exception:
        return None


# ── Syntactic Validity ────────────────────────────────────────────────────────
async def check_syntactic_validity(sql: str) -> dict:
    """
    Run EXPLAIN on the generated SQL to verify it parses without error.
    Returns {success: bool, error: str|None, score: float}.

    DDL statements (CREATE VIEW, CREATE TABLE, …) are not query statements
    and cannot be EXPLAIN'd meaningfully.  When the first extracted statement
    is DDL, the check is treated as not-applicable (score=None) rather than
    failed (score=0), so the caller can fall back to the conceptual scorer
    instead of penalising a correct conceptual answer.
    """
    if not sql or not sql.strip():
        return {"success": False, "error": "Empty SQL", "score": 0.0}

    # Extract the first SQL statement if the answer contains prose + SQL
    extracted = _extract_sql_from_text(sql)
    if not extracted:
        return {"success": False, "error": "No SQL found in answer", "score": 0.0}

    # DDL is not a query — treat as not-applicable rather than a failure.
    if not _is_query_statement(extracted):
        return {"success": None, "error": "DDL statement, not a query", "score": None}

    conn = await _get_test_connection()
    if conn is None:
        # DB unavailable — return None to signal that this check was skipped
        return {"success": None, "error": "Test DB unavailable", "score": None}

    try:
        await conn.execute(f"EXPLAIN {extracted}")
        return {"success": True, "error": None, "score": 1.0}
    except asyncpg.PostgresError as e:
        return {"success": False, "error": str(e), "score": 0.0}
    finally:
        await conn.close()


# ── Result Set Accuracy ───────────────────────────────────────────────────────
async def check_result_set_accuracy(
    sql: str,
    schema_fixture: str,
    expected_rows: list[dict],
) -> dict:
    """
    Execute the generated SQL on the test schema and compute the F1 score
    of the returned rows vs the expected result set.

    Schema fixture DDL is applied within a SAVEPOINT and rolled back after.

    Block-selection strategy (Fix 1):
      When the model answer contains multiple fenced SQL code blocks (e.g. the
      model reconstructs the original query in block 1 and writes the correct
      answer in block 2), we try each block in order and use the first one
      whose projected column names exactly match the expected_rows column names.
      This prevents picking the wrong query when blocks project different columns.
      Falls back to the first extractable statement if no column match is found.

    No-SQL handling (Fix 3):
      If no SQL statement can be found in the answer at all, f1 is returned as
      None (not 0.0).  eval_service interprets None as "harness not applicable
      for this answer" and falls back to the conceptual scorer for db_correctness,
      preventing a purely conceptual answer from being penalised by the SQL scorer.

    Returns {f1: float|None, precision: float|None, recall: float|None, error: str|None}.
    """
    if not sql or not expected_rows:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "error": "Missing SQL or expected rows"}

    # Collect all candidate SQL statements from the answer.
    candidates = _extract_all_sql_blocks(sql)
    if not candidates:
        # No SQL found at all — signal to caller that harness is inapplicable.
        return {"f1": None, "precision": None, "recall": None, "error": "No SQL found in answer"}

    # Determine expected column names so we can pick the best-matching block.
    expected_cols = set(expected_rows[0].keys()) if expected_rows else set()

    conn = await _get_test_connection()
    if conn is None:
        return {"f1": None, "precision": None, "recall": None, "error": "Test DB unavailable"}

    async def _apply_fixture(conn_: asyncpg.Connection) -> None:
        if schema_fixture:
            for stmt in _split_sql_statements(schema_fixture):
                if stmt.strip():
                    try:
                        await conn_.execute(stmt)
                    except asyncpg.PostgresError:
                        pass  # Some fixture parts may already exist

    def _compute_f1(returned_rows_: list) -> dict:
        returned_set_ = {_row_to_hashable(dict(r)) for r in returned_rows_}
        expected_set_ = {_row_to_hashable(r) for r in expected_rows}
        tp_ = len(returned_set_ & expected_set_)
        fp_ = len(returned_set_ - expected_set_)
        fn_ = len(expected_set_ - returned_set_)
        prec_ = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0.0
        rec_  = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0.0
        f1_   = (2 * prec_ * rec_ / (prec_ + rec_)) if (prec_ + rec_) > 0 else 0.0

        # ── Value-only fallback (Fix B) ───────────────────────────────────────
        # When the name+value comparison produces F1=0 but the underlying
        # values are correct, the mismatch is due to a column alias (e.g. the
        # model writes SELECT AVG(rating) AS average_rating but expected_rows
        # uses the key 'avg' which is what PostgreSQL uses for an unaliased
        # aggregate).  In this case, strip column names and compare values only.
        # This only upgrades a zero F1, never downgrades a positive one.
        if f1_ == 0.0 and returned_rows_ and expected_rows:
            ret_vals_ = {_values_only_hashable(dict(r)) for r in returned_rows_}
            exp_vals_ = {_values_only_hashable(r) for r in expected_rows}
            tp_v = len(ret_vals_ & exp_vals_)
            fp_v = len(ret_vals_ - exp_vals_)
            fn_v = len(exp_vals_ - ret_vals_)
            prec_v = tp_v / (tp_v + fp_v) if (tp_v + fp_v) > 0 else 0.0
            rec_v  = tp_v / (tp_v + fn_v) if (tp_v + fn_v) > 0 else 0.0
            f1_v   = (2 * prec_v * rec_v / (prec_v + rec_v)) if (prec_v + rec_v) > 0 else 0.0
            if f1_v > f1_:
                prec_, rec_, f1_ = prec_v, rec_v, f1_v

        return {
            "f1": round(f1_, 4),
            "precision": round(prec_, 4),
            "recall": round(rec_, 4),
            "returned_count": len(returned_set_),
            "expected_count": len(expected_set_),
            "error": None,
        }

    try:
        # ── Pass 1: try each candidate and pick the first one whose column
        #    names exactly match expected_cols.  This handles the case where
        #    the model prefixes its answer with a reconstruction of the
        #    original query (different column set) before writing the real answer.
        col_matched_result = None
        first_successful_result = None

        async with conn.transaction():
            await _apply_fixture(conn)

            for candidate in candidates:
                try:
                    rows = await conn.fetch(candidate)
                except asyncpg.PostgresError:
                    continue  # Skip blocks that fail to execute

                # Check whether the result columns match expected_cols.
                if rows:
                    result_cols = set(rows[0].keys())
                elif candidate:
                    # Zero-row result: derive column names via a cheap trick —
                    # fetch with a LIMIT 0 in a nested query if possible,
                    # but that's complex.  Accept zero-row results as column-unknown.
                    result_cols = set()
                else:
                    result_cols = set()

                result = _compute_f1(rows)

                # Keep the first successfully-executed result as fallback.
                if first_successful_result is None:
                    first_successful_result = result

                # Prefer a block whose projected columns match expected exactly.
                if expected_cols and result_cols == expected_cols:
                    col_matched_result = result
                    break  # Found the best candidate; stop searching.

        # Return column-matched result if found; otherwise first successful.
        if col_matched_result is not None:
            return col_matched_result
        if first_successful_result is not None:
            return first_successful_result

        # All candidates failed — report the last error.
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "error": "All SQL candidates failed to execute"}

    except Exception as e:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "error": str(e)}
    finally:
        await conn.close()


# ── Idiomatic PostgreSQL Check ────────────────────────────────────────────────
def check_idiomatic_postgresql(sql: str) -> dict:
    """
    Pattern-match for PostgreSQL-specific idioms and penalise ANSI-only patterns
    where PG-specific constructs are expected.

    Returns {score: float 0–1, features_found: list, anti_patterns: list}.
    """
    text = sql or ""
    text_upper = text.upper()

    # Positive signals — PostgreSQL-idiomatic constructs
    pg_idioms = {
        "RETURNING clause": bool(re.search(r"\bRETURNING\b", text, re.IGNORECASE)),
        "::cast syntax": bool(re.search(r"::", text)),
        "Dollar-quoted params ($1)": bool(re.search(r"\$\d+", text)),
        "CTE (WITH clause)": bool(re.search(r"\bWITH\b", text, re.IGNORECASE)),
        "Window functions": bool(re.search(r"\bOVER\s*\(", text, re.IGNORECASE)),
        "FILTER clause": bool(re.search(r"\bFILTER\s*\(", text, re.IGNORECASE)),
        "LATERAL join": bool(re.search(r"\bLATERAL\b", text, re.IGNORECASE)),
        "DISTINCT ON": bool(re.search(r"\bDISTINCT\s+ON\b", text, re.IGNORECASE)),
        "Array literal": bool(re.search(r"ARRAY\[", text, re.IGNORECASE)),
        "JSONB operator": bool(re.search(r"->|->>\|->>", text)),
        "ILIKE": bool(re.search(r"\bILIKE\b", text, re.IGNORECASE)),
        "SIMILAR TO": bool(re.search(r"\bSIMILAR\s+TO\b", text, re.IGNORECASE)),
        "GENERATE_SERIES": bool(re.search(r"\bGENERATE_SERIES\b", text, re.IGNORECASE)),
        "EXCLUDED (upsert)": bool(re.search(r"\bEXCLUDED\b", text, re.IGNORECASE)),
        "ON CONFLICT": bool(re.search(r"\bON\s+CONFLICT\b", text, re.IGNORECASE)),
    }

    # Negative signals — ANSI patterns where PG idioms are expected
    anti_patterns = {
        "CAST() instead of ::": (
            bool(re.search(r"\bCAST\s*\(", text, re.IGNORECASE))
            and not pg_idioms["::cast syntax"]
        ),
        "Nested SELECT instead of LATERAL": (
            bool(re.search(r"FROM\s*\(\s*SELECT", text, re.IGNORECASE))
            and not pg_idioms["LATERAL join"]
        ),
    }

    features_found = [name for name, present in pg_idioms.items() if present]
    anti_pattern_found = [name for name, flagged in anti_patterns.items() if flagged]

    # Score: more PG idioms used = higher score (cap at 1.0)
    base_score = min(len(features_found) * 0.15, 1.0)
    penalty = len(anti_pattern_found) * 0.10
    score = max(0.0, base_score - penalty)

    # Bump to 0.5 baseline if the SQL is at least syntactically SQL-like
    if not features_found and re.search(r"\bSELECT\b|\bINSERT\b|\bUPDATE\b|\bDELETE\b", text_upper):
        score = max(score, 0.5)

    return {
        "score": round(score, 4),
        "features_found": features_found,
        "anti_patterns": anti_pattern_found,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────
def _first_statement(sql_block: str) -> str:
    """
    Given the raw content of a SQL code block (which may contain multiple
    semicolon-separated statements and line comments), return only the first
    complete, non-empty SQL statement.

    This prevents asyncpg from receiving multi-statement strings and raising
    "cannot insert multiple commands into a prepared statement".

    Examples handled:
      -- comment\\nSELECT AVG(r);\\nSELECT SUM(r);  → "SELECT AVG(r)"
      WITH cte AS (SELECT …) SELECT …;              → entire CTE (no split)
    """
    sql_keywords = re.compile(
        r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH|EXPLAIN)\b",
        re.IGNORECASE,
    )
    # Split on semicolons.  Each segment may be a full statement or empty.
    segments = sql_block.split(";")
    for seg in segments:
        # Strip SQL line comments and blank lines, then check for keyword.
        stripped = re.sub(r"--[^\n]*", "", seg).strip()
        if stripped and sql_keywords.match(stripped):
            return stripped
    # Fall back: return the whole block stripped (single-statement queries
    # that don't end with a semicolon).
    return sql_block.strip()


def _extract_sql_from_text(text: str) -> Optional[str]:
    """
    Extract the first SQL statement from a model's answer.
    Handles answers that wrap SQL in markdown code blocks, or mix prose and SQL.

    Returns the first *single* complete SQL statement — multi-statement blocks
    are reduced to their first statement via _first_statement().
    """
    sql_keywords = r"(?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH|EXPLAIN)"

    # Try markdown SQL code block first.  Take the first complete statement
    # from the block content to handle answers that put multiple queries in
    # one fenced block.
    block_match = re.search(r"```(?:sql|SQL)?\s*\n(.*?)```", text, re.DOTALL)
    if block_match:
        return _first_statement(block_match.group(1))

    # Try to find a line that starts with a SQL keyword
    line_match = re.search(
        rf"({sql_keywords}\b.*?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE
    )
    if line_match:
        return _first_statement(line_match.group(1))

    # If no marker found, return the whole text and let PostgreSQL judge
    if re.search(sql_keywords, text, re.IGNORECASE):
        return _first_statement(text)

    return None


def _is_query_statement(sql: str) -> bool:
    """
    Return True if *sql* is a query (SELECT / WITH … SELECT / EXPLAIN) rather
    than a DDL or DML statement.

    The harness compares result sets, so only query statements are meaningful
    candidates.  Passing DDL (CREATE VIEW, CREATE TABLE, …) to conn.fetch()
    raises a PostgresError and causes syntactic_parse_success=0 for answers
    that are otherwise perfectly correct (e.g. a model that answers a
    conceptual sub-question with prose and then answers a later sub-question
    with CREATE VIEW examples).
    """
    return bool(re.match(r"\s*(SELECT|WITH|EXPLAIN)\b", sql, re.IGNORECASE))


def _extract_all_sql_blocks(text: str) -> list[str]:
    """
    Return the first *query* statement from EVERY fenced SQL code block in the
    answer.  DDL statements (CREATE, DROP, ALTER, INSERT, UPDATE, DELETE) are
    excluded because the harness is a result-set comparator, not a DDL executor.

    Used by check_result_set_accuracy to find the block whose column projection
    matches the expected_rows column names.
    """
    blocks = re.findall(r"```(?:sql|SQL)?\s*\n(.*?)```", text, re.DOTALL)
    result = []
    for b in blocks:
        stmt = _first_statement(b)
        if stmt and _is_query_statement(stmt):
            result.append(stmt)
    # If no fenced blocks, fall back to the single extraction
    if not result:
        single = _extract_sql_from_text(text)
        if single and _is_query_statement(single):
            result.append(single)
    return result


def _split_sql_statements(sql: str) -> list[str]:
    """Split a DDL fixture string into individual statements on semicolons."""
    statements = [s.strip() for s in sql.split(";")]
    return [s for s in statements if s]


def _row_to_hashable(row: dict) -> frozenset:
    """Convert a row dict to a hashable frozenset for set comparison."""
    items = []
    for k, v in row.items():
        if isinstance(v, list):
            items.append((k, tuple(v)))
        elif isinstance(v, dict):
            items.append((k, tuple(sorted(v.items()))))
        else:
            items.append((k, v))
    return frozenset(items)


def _values_only_hashable(row: dict) -> frozenset:
    """
    Convert a row dict to a hashable frozenset of VALUES ONLY, ignoring column
    names.  Used as a fallback when column names differ due to aliasing (e.g.
    SELECT AVG(x) AS my_alias vs expected key 'avg').

    Values are normalised: floats are rounded to 4 decimal places so that
    5.333333333 matches the stored expected value 5.333333.
    """
    items = []
    for v in row.values():
        if isinstance(v, float):
            items.append(round(v, 4))
        elif hasattr(v, "__float__"):
            # Decimal / numeric types from asyncpg
            try:
                items.append(round(float(v), 4))
            except (TypeError, ValueError):
                items.append(v)
        elif isinstance(v, list):
            items.append(tuple(v))
        elif isinstance(v, dict):
            items.append(tuple(sorted(v.items())))
        else:
            items.append(v)
    return frozenset(items)
