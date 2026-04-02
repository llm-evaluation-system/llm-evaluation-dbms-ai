"""
prompting/templates.py — All nine prompting strategy templates as specified in
Section 4 of the evaluation framework specification.

Each strategy is a callable that receives the question text, optional few-shot
examples, and optional model context, and returns a (system_prompt, user_prompt)
tuple ready to be sent to the LLM client.

Strategy registry:
  zero-shot      | one-shot    | few-shot      | cot
  few-shot-cot   | self-consistency | role-prompting
  least-to-most  | react
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptPair:
    """Structured output from a prompt builder."""
    system_prompt: str
    user_prompt: str
    strategy: str


# ── System Prompt Styles ──────────────────────────────────────────────────────
SYSTEM_PROMPTS = {
    "minimal": (
        "You are a helpful assistant."
    ),
    "role-based": (
        "You are a database systems expert with deep knowledge of relational "
        "databases, SQL, and database management systems theory."
    ),
    "expert-persona": (
        "You are an expert DBMS professor and principal database architect with "
        "20 years of experience teaching from the Ramakrishnan & Gehrke "
        "'Database Management Systems' textbook, specialising in PostgreSQL. "
        "You provide precise, complete, textbook-accurate answers to database "
        "questions. When writing SQL, you use idiomatic PostgreSQL syntax. "
        "When answering theory questions, you cite exact definitions and avoid "
        "hallucinated facts."
    ),
}


def _system(style: str) -> str:
    return SYSTEM_PROMPTS.get(style, SYSTEM_PROMPTS["expert-persona"])


def _format_example(example: dict, include_reasoning: bool = False) -> str:
    """Format a single few-shot example as a Q&A block."""
    q = example.get("question", "")
    a = example.get("answer", "")
    reasoning = example.get("reasoning", "")
    lines = [f"Question: {q}"]
    if include_reasoning and reasoning:
        lines.append(f"Reasoning:\n{reasoning}")
    lines.append(f"Answer: {a}")
    return "\n".join(lines)


# ── 1. Zero-Shot ─────────────────────────────────────────────────────────────
def zero_shot(
    question: str,
    system_prompt_style: str = "expert-persona",
    **_,
) -> PromptPair:
    """
    Section 4.2.1 — Zero-Shot Baseline.
    No examples, no chain-of-thought scaffolding.  Tests raw model knowledge.
    """
    return PromptPair(
        system_prompt=_system(system_prompt_style),
        user_prompt=question,
        strategy="zero-shot",
    )


# ── 2. One-Shot ──────────────────────────────────────────────────────────────
def one_shot(
    question: str,
    examples: Optional[list[dict]] = None,
    system_prompt_style: str = "expert-persona",
    **_,
) -> PromptPair:
    """
    Section 4.1 — One-Shot.
    One worked example from the same topic area anchors format and style.
    """
    if not examples:
        # Fall back to zero-shot if no example available
        return zero_shot(question, system_prompt_style)

    example_block = _format_example(examples[0])
    prompt = (
        f"Here is an example of a well-answered DBMS question:\n\n"
        f"{example_block}\n\n"
        f"Now answer the following question in the same style:\n\n"
        f"Question: {question}"
    )
    return PromptPair(
        system_prompt=_system(system_prompt_style),
        user_prompt=prompt,
        strategy="one-shot",
    )


# ── 3. Few-Shot (3-shot) ──────────────────────────────────────────────────────
def few_shot(
    question: str,
    examples: Optional[list[dict]] = None,
    system_prompt_style: str = "expert-persona",
    n_examples: int = 3,
    **_,
) -> PromptPair:
    """
    Section 4.1 — Few-Shot (3-shot by default).
    Three worked examples from the same topic; strongest in-context learning.
    """
    if not examples:
        return zero_shot(question, system_prompt_style)

    selected = examples[:n_examples]
    example_blocks = "\n\n".join(
        f"Example {i+1}:\n{_format_example(ex)}"
        for i, ex in enumerate(selected)
    )
    prompt = (
        f"Here are {len(selected)} example DBMS questions with correct answers:\n\n"
        f"{example_blocks}\n\n"
        f"Now answer the following question following the same approach and format:\n\n"
        f"Question: {question}"
    )
    return PromptPair(
        system_prompt=_system(system_prompt_style),
        user_prompt=prompt,
        strategy="few-shot",
    )


# ── 4. Chain-of-Thought (CoT) ─────────────────────────────────────────────────
def chain_of_thought(
    question: str,
    system_prompt_style: str = "expert-persona",
    cot_variant: str = "explicit",
    **_,
) -> PromptPair:
    """
    Section 4.2.2 — Chain-of-Thought (Explicit variant).
    Appends explicit step-by-step reasoning scaffold.  Particularly effective
    for normalization decomposition, ARIES trace, and serializability checking.
    """
    if cot_variant == "implicit":
        suffix = "\n\nLet us think step by step."
    else:
        suffix = (
            "\n\nApproach this step by step:\n"
            "Step 1: Identify the relevant DBMS concept(s).\n"
            "Step 2: Apply definitions and rules to the given scenario.\n"
            "Step 3: Derive and state your final answer clearly."
        )

    return PromptPair(
        system_prompt=_system(system_prompt_style),
        user_prompt=question + suffix,
        strategy="cot",
    )


# ── 5. Few-Shot + CoT ─────────────────────────────────────────────────────────
def few_shot_cot(
    question: str,
    examples: Optional[list[dict]] = None,
    system_prompt_style: str = "expert-persona",
    n_examples: int = 3,
    **_,
) -> PromptPair:
    """
    Section 4.1 — Few-Shot + Chain-of-Thought.
    Gold standard for complex multi-step questions: each example shows the
    full step-by-step reasoning, not just the final answer.
    """
    if not examples:
        return chain_of_thought(question, system_prompt_style)

    selected = examples[:n_examples]
    example_blocks = "\n\n".join(
        f"Example {i+1}:\n{_format_example(ex, include_reasoning=True)}"
        for i, ex in enumerate(selected)
    )
    prompt = (
        f"Here are {len(selected)} DBMS examples with complete step-by-step reasoning:\n\n"
        f"{example_blocks}\n\n"
        f"Now answer the following question using the same step-by-step reasoning format:\n\n"
        f"Question: {question}\n\n"
        f"Step 1: Identify the relevant DBMS concept(s).\n"
        f"Step 2: Apply definitions and rules to the given scenario.\n"
        f"Step 3: Derive and state your final answer clearly."
    )
    return PromptPair(
        system_prompt=_system(system_prompt_style),
        user_prompt=prompt,
        strategy="few-shot-cot",
    )


# ── 6. Self-Consistency ───────────────────────────────────────────────────────
def self_consistency(
    question: str,
    system_prompt_style: str = "expert-persona",
    **_,
) -> PromptPair:
    """
    Section 4.2.4 — Self-Consistency.
    The prompt itself is equivalent to CoT; the self-consistency logic (k=5
    sampling + majority vote) is implemented in the runner, not the prompt.
    This function returns the single-sample prompt.
    """
    return chain_of_thought(question, system_prompt_style, cot_variant="explicit")


# ── 7. Role Prompting ─────────────────────────────────────────────────────────
def role_prompting(
    question: str,
    system_prompt_style: str = "expert-persona",
    **_,
) -> PromptPair:
    """
    Section 4.1 — Role Prompting.
    Enriched system prompt primes the model as an expert DBA / DBMS professor.
    For simple factual / definition questions this alone is often sufficient.
    """
    enhanced_system = (
        "You are a world-class DBMS professor, the author of multiple textbooks "
        "on relational database systems, and a principal engineer at a major "
        "database vendor. You have memorised the entire Ramakrishnan & Gehrke "
        "textbook and can answer any question from it accurately. You are also "
        "a PostgreSQL core contributor who writes idiomatic, production-quality "
        "SQL. Answer every question with textbook precision and completeness."
    )
    return PromptPair(
        system_prompt=enhanced_system,
        user_prompt=question,
        strategy="role-prompting",
    )


# ── 8. Least-to-Most ─────────────────────────────────────────────────────────
def least_to_most(
    question: str,
    system_prompt_style: str = "expert-persona",
    **_,
) -> PromptPair:
    """
    Section 4.2.5 — Least-to-Most Prompting.
    Ideal for multi-step DBMS problems: normalization decomposition, ARIES
    recovery trace, query optimizer plan construction.

    Phase 1 prompt asks the model to list sub-problems first, then solve each.
    """
    prompt = (
        f"To answer the following question, follow this structured approach:\n\n"
        f"1. DECOMPOSE: First, list all sub-problems you need to solve to answer "
        f"the question completely. Number each sub-problem clearly.\n\n"
        f"2. SOLVE: For each sub-problem in order, provide a complete solution "
        f"before moving to the next sub-problem.\n\n"
        f"3. SYNTHESISE: Combine your sub-problem solutions into a clear, "
        f"complete final answer.\n\n"
        f"Question: {question}"
    )
    return PromptPair(
        system_prompt=_system(system_prompt_style),
        user_prompt=prompt,
        strategy="least-to-most",
    )


# ── 9. ReAct (Reason + Act) ───────────────────────────────────────────────────
def react(
    question: str,
    system_prompt_style: str = "expert-persona",
    **_,
) -> PromptPair:
    """
    Section 4.1 — ReAct (Reason + Act).
    Interleaves reasoning, action (e.g. 'check constraint'), and observation.
    Best for iterative diagnostic or debugging questions.
    """
    prompt = (
        f"Solve the following DBMS question by interleaving Thought, Action, "
        f"and Observation steps:\n\n"
        f"- Thought: What do I need to analyse or determine?\n"
        f"- Action: What specific DBMS rule, definition, or check do I apply?\n"
        f"- Observation: What is the result of applying that rule?\n\n"
        f"Repeat Thought/Action/Observation until you have enough information, "
        f"then provide a clear Final Answer.\n\n"
        f"Question: {question}"
    )
    return PromptPair(
        system_prompt=_system(system_prompt_style),
        user_prompt=prompt,
        strategy="react",
    )


# ── Strategy Dispatcher ───────────────────────────────────────────────────────
STRATEGY_MAP: dict[str, callable] = {
    "zero-shot": zero_shot,
    "one-shot": one_shot,
    "few-shot": few_shot,
    "cot": chain_of_thought,
    "few-shot-cot": few_shot_cot,
    "self-consistency": self_consistency,
    "role-prompting": role_prompting,
    "least-to-most": least_to_most,
    "react": react,
}


def build_prompt(
    strategy: str,
    question: str,
    examples: Optional[list[dict]] = None,
    system_prompt_style: str = "expert-persona",
    schema_fixture: Optional[str] = None,
    judge_hint: Optional[str] = None,
    **kwargs,
) -> PromptPair:
    """
    Dispatch to the correct prompt builder based on the strategy name.

    Args:
        strategy:           One of the nine strategy identifiers.
        question:           The question text from the question bank.
        examples:           Pre-fetched few-shot examples (may be None).
        system_prompt_style: minimal | role-based | expert-persona.
        schema_fixture:     DDL for the test DB schema (sql+practical questions).
                            Prepended to the prompt so the model uses the exact
                            table/column names that the execution harness expects.
        judge_hint:         Additional evaluation context from sql_fixtures.py.
                            Included when it contains table name remaps (e.g.
                            femployees, ffrom/fto) that the model must respect.
        **kwargs:           Strategy-specific overrides (e.g. cot_variant).

    Returns:
        PromptPair with system_prompt and user_prompt ready to send.
    """
    builder = STRATEGY_MAP.get(strategy)
    if builder is None:
        raise ValueError(
            f"Unknown prompt strategy '{strategy}'. "
            f"Available: {list(STRATEGY_MAP.keys())}"
        )

    # For sql+practical questions: prepend the exact schema DDL so the model
    # uses the correct table/column names that the execution harness expects.
    # Without this, models write natural names (e.g. Employees, from, to) that
    # differ from fixture names (femployees, ffrom, fto) and all queries fail.
    augmented_question = question
    if schema_fixture:
        # Extract only CREATE TABLE lines (not INSERT data) to keep prompt short
        ddl_lines = [
            line for line in schema_fixture.split("\n")
            if any(kw in line.upper() for kw in
                   ("CREATE TABLE", "PRIMARY KEY", "REFERENCES", "VARCHAR",
                    "INTEGER", "REAL", "FLOAT", "TEXT", "TIME", "BOOLEAN",
                    "FOREIGN KEY", ");"))
        ]
        schema_summary = "\n".join(ddl_lines).strip()
        if schema_summary:
            augmented_question = (
                f"IMPORTANT — Use EXACTLY these table and column names "
                f"(the test database uses these exact names):\n"
                f"```sql\n{schema_summary}\n```\n\n"
                f"{question}"
            )

    result = builder(
        question=augmented_question,
        examples=examples,
        system_prompt_style=system_prompt_style,
        **kwargs,
    )
    return result


# ── Topic-Specific Defaults (Section 4.5) ────────────────────────────────────
TOPIC_DEFAULT_STRATEGY: dict[str, str] = {
    "SQL: QUERIES, CONSTRAINTS, TRIGGERS": "few-shot",
    "SCHEMA REFINEMENT AND NORMAL FORMS": "few-shot-cot",
    "OVERVIEW OF TRANSACTION MANAGEMENT": "least-to-most",
    "CONCURRENCY CONTROL": "least-to-most",
    "CRASH RECOVERY": "least-to-most",
    "INTRODUCTION TO DATABASE DESIGN": "role-prompting",
    "THE RELATIONAL MODEL": "role-prompting",
    "RELATIONAL ALGEBRA AND CALCULUS": "few-shot-cot",
    "OVERVIEW OF QUERY EVALUATION": "cot",
    "EXTERNAL SORTING": "cot",
    "EVALUATING RELATIONAL OPERATORS": "cot",
    "A TYPICAL RELATIONAL QUERY OPTIMIZER": "few-shot-cot",
    "OVERVIEW OF STORAGE AND INDEXING": "role-prompting",
    "STORING DATA: DISKS AND FILES": "role-prompting",
    "TREE-STRUCTURED INDEXING": "cot",
    "HASH-BASED INDEXING": "cot",
    "SCHEMA REFINEMENT AND NORMAL FORMS": "few-shot-cot",
    "PHYSICAL DATABASE DESIGN AND TUNING": "few-shot",
    "SECURITY AND AUTHORIZATION": "few-shot",
    "DATA WAREHOUSING AND DECISION SUPPORT": "few-shot-cot",
    "DATABASE APPLICATION DEVELOPMENT": "role-prompting",
    "INTERNET APPLICATIONS": "role-prompting",
}


def get_default_strategy(subtopic: str) -> str:
    """Return the recommended default prompting strategy for a given subtopic."""
    return TOPIC_DEFAULT_STRATEGY.get(subtopic.upper(), "few-shot-cot")
