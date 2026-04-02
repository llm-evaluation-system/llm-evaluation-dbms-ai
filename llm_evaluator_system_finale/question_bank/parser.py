"""
question_bank/parser.py — One-time parser that converts the Excel question bank
into a structured, validated JSON representation persisted to disk.

The Excel file has the following layout:
  - Column 0: Topic         (populated only on first row of a new topic group)
  - Column 1: Subtopic      (one per row)
  - Column 2: Questions     (Python-style list stored as a string, or raw text)
  - Column 3: Expected Answer

The parser:
  1. Reads all rows and forward-fills the Topic column.
  2. Extracts individual question/answer pairs from each cell, which may
     contain Python list literals or plain text.
  3. Assigns question types based on subtopic heuristics.
  4. Assigns difficulty tiers based on exercise number patterns.
  5. Writes the result to data/question_bank.json and returns the parsed data.

Run this script once after cloning the repo:
    python question_bank/parser.py
"""

import ast
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

# Allow imports from parent package when running as __main__
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import QUESTION_BANK_EXCEL_PATH, QUESTION_BANK_SHEET_NAME


# ── Topic → question type heuristics ─────────────────────────────────────────
# IMPORTANT: question_type has exactly two valid values after the question-bank
# correction: 'conceptual' (explanation/analysis) or 'practical' (produces an
# artifact — SQL, RA expression, algorithm trace, numeric computation).
# Content-specific routing (SQL harness, rubric selection) is done via TAGS,
# not question_type.  See question_bank/sql_fixtures.py for routing helpers.
#
# These subtopic defaults are HEURISTICS only. The corrected data/question_bank.json
# is the authoritative source. If the parser is re-run, its output should be
# reviewed and the per-question corrections from the analysis session re-applied.
SUBTOPIC_TO_TYPE: dict[str, str] = {
    "INTRODUCTION TO DATABASE DESIGN": "practical",    # draw ER diagrams
    "THE RELATIONAL MODEL": "practical",               # write SQL DDL/DML
    "RELATIONAL ALGEBRA AND CALCULUS": "practical",    # write RA/TRC/DRC
    "SQL: QUERIES, CONSTRAINTS, TRIGGERS": "practical",# write SQL queries
    "DATABASE APPLICATION DEVELOPMENT": "conceptual",  # explain APIs
    "INTERNET APPLICATIONS": "conceptual",             # explain web concepts
    "OVERVIEW OF STORAGE AND INDEXING": "conceptual",  # explain index choices
    "STORING DATA: DISKS AND FILES": "conceptual",     # explain storage
    "TREE-STRUCTURED INDEXING": "conceptual",          # explain B+ tree
    "HASH-BASED INDEXING": "conceptual",               # explain hashing
    "OVERVIEW OF QUERY EVALUATION": "conceptual",      # explain query eval
    "EXTERNAL SORTING": "conceptual",                  # compute sort costs
    "EVALUATING RELATIONAL OPERATORS": "practical",    # trace join algorithms
    "A TYPICAL RELATIONAL QUERY OPTIMIZER": "conceptual", # explain optimizer
    "OVERVIEW OF TRANSACTION MANAGEMENT": "conceptual",# explain ACID/locks
    "CONCURRENCY CONTROL": "conceptual",               # explain CC protocols
    "CRASH RECOVERY": "conceptual",                    # explain ARIES
    "SCHEMA REFINEMENT AND NORMAL FORMS": "conceptual",# explain normalization
    "PHYSICAL DATABASE DESIGN AND TUNING": "practical",# design indexes/schema
    "SECURITY AND AUTHORIZATION": "conceptual",        # explain security
    "DATA WAREHOUSING AND DECISION SUPPORT": "conceptual", # explain OLAP
}

# ── Difficulty heuristics based on exercise number suffix ─────────────────────
def _infer_difficulty(exercise_number: str, question_text: str) -> str:
    """
    Heuristic difficulty assignment.
    Odd exercise numbers (1, 3, 5, …) are typically review questions → easy.
    Even exercise numbers (2, 4, 6, …) or those with long multi-part text → medium/hard.
    Questions containing 'every', 'optimal', 'trace', 'compare', 'design' → hard.
    """
    hard_keywords = {
        "every", "optimal", "trace", "compare", "prove", "implement",
        "aries", "serializability", "normalization", "decompose", "bcnf",
        "3nf", "elo", "pairwise", "tournament", "query plan"
    }
    medium_keywords = {
        "explain", "describe", "identify", "define", "list", "give an example",
        "what is", "how does", "contrast", "difference"
    }

    text_lower = (question_text or "").lower()
    if any(kw in text_lower for kw in hard_keywords):
        return "hard"
    if any(kw in text_lower for kw in medium_keywords):
        return "medium"

    # Use the exercise number parity as a tiebreaker
    if exercise_number:
        nums = re.findall(r"\d+", exercise_number)
        if nums:
            last_num = int(nums[-1])
            if last_num % 2 == 1:
                return "easy"
            return "medium"
    return "medium"


# ── String-to-list extraction ─────────────────────────────────────────────────
def _extract_list_from_string(raw: str) -> list[str]:
    """
    Try to extract a Python list literal from the cell value.
    Handles both `questions = [...]` / `answers = [...]` patterns and
    triple-quoted string lists.  Falls back to the raw string if parsing fails.
    """
    if not isinstance(raw, str) or not raw.strip():
        return []

    raw = raw.strip()

    # Pattern: questions = [...] or answers = [...]
    match = re.search(r"(?:questions|answers)\s*=\s*(\[.*\])", raw, re.DOTALL)
    if match:
        list_str = match.group(1)
        try:
            items = ast.literal_eval(list_str)
            if isinstance(items, list):
                return [str(i).strip() for i in items if str(i).strip()]
        except (ValueError, SyntaxError):
            pass

    # Pattern: already a bare Python list
    if raw.startswith("["):
        try:
            items = ast.literal_eval(raw)
            if isinstance(items, list):
                return [str(i).strip() for i in items if str(i).strip()]
        except (ValueError, SyntaxError):
            pass

    # Fallback: treat the entire cell as one item
    return [raw]


# ── Exercise number extraction ────────────────────────────────────────────────
def _extract_exercise_number(text: str) -> tuple[str, str]:
    """
    Return (exercise_number, cleaned_text) where exercise_number is e.g.
    'Exercise 5.1' and cleaned_text has the prefix stripped.
    """
    pattern = r"^(Exercise\s+[\d\.]+):?\s*"
    m = re.match(pattern, text.strip(), re.IGNORECASE)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    # Also handle '# Exercise 5.1' (markdown heading)
    pattern2 = r"^#+\s*(Exercise\s+[\d\.]+)\s*\n+"
    m2 = re.match(pattern2, text.strip(), re.IGNORECASE)
    if m2:
        return m2.group(1).strip(), text[m2.end():].strip()
    return "", text.strip()


# ── Generate a stable short ID ────────────────────────────────────────────────
def _stable_id(topic: str, subtopic: str, exercise: str, index: int) -> str:
    seed = f"{topic}|{subtopic}|{exercise}|{index}"
    return hashlib.md5(seed.encode()).hexdigest()[:12]


# ── Main parser ────────────────────────────────────────────────────────────────
def parse_question_bank(excel_path: str = QUESTION_BANK_EXCEL_PATH) -> list[dict]:
    """
    Parse the Excel question bank and return a flat list of question dicts.
    Each dict has keys: id, topic, subtopic, exercise_number, question_text,
    expected_answer, question_type, difficulty, tags.
    """
    df = pd.read_excel(excel_path, sheet_name=QUESTION_BANK_SHEET_NAME, dtype=str)

    # Forward-fill the Topic column (merged cells appear as NaN after the first)
    df["Topic"] = df["Topic"].ffill()
    df.columns = ["topic", "subtopic", "questions_raw", "answers_raw"]
    df = df.fillna("")

    questions: list[dict] = []
    global_index = 0

    for _, row in df.iterrows():
        topic = row["topic"].strip().upper()
        subtopic = row["subtopic"].strip().upper()

        if not subtopic:
            continue  # skip completely empty rows

        q_list = _extract_list_from_string(row["questions_raw"])
        a_list = _extract_list_from_string(row["answers_raw"])

        # Pad answers to match questions if lengths differ
        while len(a_list) < len(q_list):
            a_list.append("")

        q_type = SUBTOPIC_TO_TYPE.get(subtopic, "conceptual")

        for i, q_text in enumerate(q_list):
            if not q_text.strip():
                continue

            exercise_number, cleaned_q = _extract_exercise_number(q_text)
            expected_answer = a_list[i].strip() if i < len(a_list) else ""

            difficulty = _infer_difficulty(exercise_number, cleaned_q)

            # Generate simple keyword tags
            tags = _extract_tags(cleaned_q, subtopic)

            questions.append({
                "id": _stable_id(topic, subtopic, exercise_number, global_index),
                "topic": topic,
                "subtopic": subtopic,
                "exercise_number": exercise_number or None,
                "question_text": cleaned_q,
                "expected_answer": expected_answer or None,
                "question_type": q_type,
                "difficulty": difficulty,
                "tags": tags,
            })
            global_index += 1

    return questions


def _extract_tags(question_text: str, subtopic: str) -> list[str]:
    """Generate keyword tags for filtering and retrieval."""
    tags: set[str] = set()
    text_lower = question_text.lower()

    # Valid tags (must match exactly the tags used in the corrected question bank):
    #   sql, normalization, transactions, database_concepts, relational_model, er_diagram
    tag_keywords = {
        "sql": ["select", "from", "where", "join", "group by", "having",
                "insert", "update", "delete", "create table", "trigger",
                "view", "constraint", "jdbc", "sqlj", "embedded sql",
                "window", "materialized view"],
        "normalization": ["1nf", "2nf", "3nf", "bcnf", "normal form",
                          "functional dependency", "decompose", "lossless",
                          "dependency preservation"],
        "transactions": ["acid", "serializability", "2pl", "lock", "deadlock",
                         "commit", "rollback", "isolation", "schedule", "aries",
                         "redo", "undo", "checkpoint", "recovery", "crash",
                         "timestamp", "optimistic", "multiversion"],
        "er_diagram": ["entity", "relationship", "er diagram", "cardinality",
                       "participation constraint", "weak entity", "ternary",
                       "aggregation", "role indicator"],
        "relational_model": ["relational algebra", "tuple calculus", "domain calculus",
                             "relation schema", "candidate key", "foreign key",
                             "relational model", "unsafe query", "division"],
        "database_concepts": ["b+ tree", "b-tree", "hash index", "clustered",
                              "buffer", "disk", "page", "record", "i/o",
                              "external sort", "query plan", "cost estimation",
                              "join algorithm", "nested loop", "selectivity",
                              "olap", "star schema", "fact table", "view",
                              "materialization", "security", "encryption", "ssl"],
    }

    for tag, keywords in tag_keywords.items():
        if any(kw in text_lower for kw in keywords):
            tags.add(tag)

    # NOTE: do NOT add the raw subtopic as a tag — it produces invalid tag values.
    # Tags are constrained to: sql, normalization, transactions, database_concepts,
    # relational_model, er_diagram. The keyword matching above handles tagging.

    return sorted(tags)


def save_question_bank_json(
    questions: list[dict],
    output_path: str = "data/question_bank.json"
) -> None:
    """Persist the parsed question bank to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(questions)} questions to {output_path}")


def load_question_bank_json(path: str = "data/question_bank.json") -> list[dict]:
    """Load the pre-parsed question bank from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    excel_path = sys.argv[1] if len(sys.argv) > 1 else QUESTION_BANK_EXCEL_PATH
    print(f"Parsing question bank from: {excel_path}")
    questions = parse_question_bank(excel_path)
    print(f"Parsed {len(questions)} questions across subtopics:")

    # Print summary
    from collections import Counter
    subtopic_counts = Counter(q["subtopic"] for q in questions)
    for subtopic, count in sorted(subtopic_counts.items()):
        print(f"  {subtopic}: {count} questions")

    save_question_bank_json(questions)
    print("\nDone.")
