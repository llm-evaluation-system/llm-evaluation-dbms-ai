"""
evaluators/format_compliance.py — Automated format compliance checker.

Implements the five format compliance metrics from Section 2.4:
  1. JSON Output Validity       (25%)
  2. SQL Code Block Formatting  (20%)
  3. Numbered Steps Compliance  (20%)
  4. Schema/Table Formatting    (20%)
  5. Length Constraints         (15%)

Each checker is independent and returns a 0/1 or 0-1 score.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from config import FORMAT_WEIGHTS


def check_json_validity(answer: str) -> dict:
    """
    Metric 1: Verify that the answer parses as valid JSON when JSON output
    was requested.
    """
    if not answer or not answer.strip():
        return {"score": 0.0, "valid": False, "error": "Empty response"}

    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", answer.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE).strip()

    try:
        json.loads(cleaned)
        return {"score": 1.0, "valid": True, "error": None}
    except json.JSONDecodeError as e:
        return {"score": 0.0, "valid": False, "error": str(e)}


def check_sql_code_block(answer: str) -> dict:
    """
    Metric 2: SQL answers must be enclosed in proper markdown code blocks.
    No intermixed prose within the SQL block.
    """
    if not answer:
        return {"score": 0.0, "has_code_block": False}

    sql_block = re.search(r"```(?:sql|SQL)\s*\n.*?```", answer, re.DOTALL)
    if sql_block:
        block_content = sql_block.group(0)
        # Penalise if prose is intermixed inside the block
        lines = block_content.split("\n")
        sql_lines = [l for l in lines[1:-1] if l.strip()]
        prose_lines = [
            l for l in sql_lines
            if not re.match(
                r"^\s*(--|SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH|FROM|WHERE"
                r"|JOIN|ON|GROUP|ORDER|HAVING|LIMIT|OFFSET|SET|VALUES|RETURNING|\(|\)|,|;)",
                l, re.IGNORECASE
            )
            and len(l.strip()) > 20
        ]
        penalty = min(len(prose_lines) * 0.15, 0.5)
        return {"score": max(0.5, 1.0 - penalty), "has_code_block": True}

    # Partial credit if SQL is present but not in a code block
    has_sql = bool(re.search(
        r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE TABLE|CREATE INDEX)\b",
        answer, re.IGNORECASE
    ))
    return {"score": 0.3 if has_sql else 0.0, "has_code_block": False}


def check_numbered_steps(answer: str) -> dict:
    """
    Metric 3: Step-by-step questions must produce numbered, sequential answers.
    """
    if not answer:
        return {"score": 0.0, "has_numbered_steps": False}

    # Look for patterns like "1.", "Step 1:", "1)", "(1)"
    numbered_patterns = [
        r"^\s*\d+\.",           # 1.
        r"^\s*Step\s+\d+:",     # Step 1:
        r"^\s*\d+\)",           # 1)
        r"^\s*\(\d+\)",         # (1)
    ]
    step_count = 0
    for line in answer.split("\n"):
        if any(re.match(p, line, re.IGNORECASE) for p in numbered_patterns):
            step_count += 1

    if step_count >= 3:
        return {"score": 1.0, "has_numbered_steps": True, "step_count": step_count}
    elif step_count >= 1:
        return {"score": 0.5, "has_numbered_steps": True, "step_count": step_count}
    else:
        return {"score": 0.0, "has_numbered_steps": False, "step_count": 0}


def check_schema_formatting(answer: str) -> dict:
    """
    Metric 4: Schema design answers must use DDL (CREATE TABLE), not prose descriptions.
    """
    if not answer:
        return {"score": 0.0, "has_ddl": False}

    has_create_table = bool(re.search(r"\bCREATE\s+TABLE\b", answer, re.IGNORECASE))
    has_alter_table = bool(re.search(r"\bALTER\s+TABLE\b", answer, re.IGNORECASE))
    has_foreign_key = bool(re.search(r"\bFOREIGN\s+KEY\b", answer, re.IGNORECASE))

    if has_create_table:
        score = 0.7
        if has_foreign_key:
            score += 0.2
        if has_alter_table or re.search(r"\bCREATE\s+INDEX\b", answer, re.IGNORECASE):
            score += 0.1
        return {"score": min(score, 1.0), "has_ddl": True}

    # Prose description of schema — partial credit
    schema_prose = bool(re.search(
        r"\b(primary key|foreign key|table|column|attribute|entity|relation)\b",
        answer, re.IGNORECASE
    ))
    return {"score": 0.3 if schema_prose else 0.0, "has_ddl": False}


def check_length_compliance(
    answer: str,
    length_instruction: Optional[str] = None,
) -> dict:
    """
    Metric 5: Check that the answer respects any explicit length constraints
    in the question (e.g. 'in one paragraph', 'list 3 key points').
    This is a heuristic check; full assessment requires LLM judge.
    """
    if not answer or not length_instruction:
        return {"score": 1.0, "compliant": True, "note": "No length constraint specified"}

    word_count = len(answer.split())
    instruction_lower = length_instruction.lower()

    # "one paragraph" — expect 50-200 words
    if "one paragraph" in instruction_lower or "single paragraph" in instruction_lower:
        paragraph_count = len([p for p in answer.split("\n\n") if p.strip()])
        if paragraph_count == 1 and 30 <= word_count <= 250:
            return {"score": 1.0, "compliant": True}
        return {"score": 0.4, "compliant": False, "note": f"{paragraph_count} paragraphs, {word_count} words"}

    # "list N key points"
    n_match = re.search(r"list\s+(\d+)\s+key\s+points?", instruction_lower)
    if n_match:
        expected_n = int(n_match.group(1))
        step_count = sum(
            1 for line in answer.split("\n")
            if re.match(r"^\s*[-•*\d]", line)
        )
        if abs(step_count - expected_n) <= 1:
            return {"score": 1.0, "compliant": True}
        return {"score": 0.5, "compliant": False, "note": f"Expected {expected_n} points, found {step_count}"}

    return {"score": 1.0, "compliant": True, "note": "Length constraint check passed"}


def compute_format_compliance_score(
    answer: str,
    question_type: str = "conceptual",
    length_instruction: Optional[str] = None,
    expected_formats: Optional[list[str]] = None,
    tags: Optional[list[str]] = None,
) -> dict:
    """
    Compute the overall format compliance score (0–1) for an answer.

    Args:
        answer:            The model's answer text.
        question_type:     'conceptual' | 'practical' (corrected bank values).
                           Legacy values ('sql', 'schema', etc.) still accepted
                           for backward compatibility.
        length_instruction: Any explicit length constraint from the question.
        expected_formats:   List of formats requested: ['json', 'sql_block', 'numbered', 'ddl']
        tags:              Question tags list from the question bank (e.g. ['sql', 'transactions']).
                           Used for content-aware routing since question_type no longer carries
                           content semantics after the question-bank correction.

    Returns dict with individual metric scores and weighted composite.
    """
    from question_bank.sql_fixtures import needs_sql_harness, needs_numbered_steps_check

    _tags = tags or []
    formats = expected_formats or []
    w = FORMAT_WEIGHTS

    # Determine whether SQL block formatting should be checked.
    # Trigger when:
    #   - caller explicitly requests it via expected_formats
    #   - question is a practical sql question (new routing)
    #   - legacy: question_type was 'sql' (old routing, kept for compat)
    _is_sql = (
        "sql_block" in formats
        or needs_sql_harness(question_type, _tags)
        or question_type == "sql"          # legacy compat
    )

    # Numbered steps check for transaction traces, ARIES logs, lock sequences.
    # Trigger when:
    #   - caller explicitly requests it
    #   - question is a practical transactions question (new routing)
    #   - legacy: question_type was 'transaction' / 'relational_algebra' / 'query_optimization'
    _needs_numbered = (
        "numbered" in formats
        or needs_numbered_steps_check(question_type, _tags)
        or question_type in ("transaction", "relational_algebra", "query_optimization")  # legacy
    )

    # Schema/DDL formatting check.
    # Trigger when:
    #   - caller explicitly requests it
    #   - same conditions as SQL harness (practical sql questions produce DDL)
    #   - legacy: question_type was 'schema'
    _needs_schema = (
        "ddl" in formats
        or needs_sql_harness(question_type, _tags)
        or question_type == "schema"       # legacy compat
    )

    # Run checks conditionally based on question type and requested formats
    json_score = check_json_validity(answer)["score"] if "json" in formats else 1.0
    sql_block_score = check_sql_code_block(answer)["score"] if _is_sql else 1.0
    numbered_score = check_numbered_steps(answer)["score"] if _needs_numbered else 1.0
    schema_score = check_schema_formatting(answer)["score"] if _needs_schema else 1.0
    length_result = check_length_compliance(answer, length_instruction)
    length_score = length_result["score"]

    composite = (
        w["json_output_validity"] * json_score
        + w["sql_code_block"] * sql_block_score
        + w["numbered_steps"] * numbered_score
        + w["schema_table_formatting"] * schema_score
        + w["length_constraints"] * length_score
    )

    return {
        "composite_score": round(composite, 4),
        "json_score": json_score,
        "sql_block_score": sql_block_score,
        "numbered_steps_score": numbered_score,
        "schema_formatting_score": schema_score,
        "length_compliance_score": length_score,
    }
