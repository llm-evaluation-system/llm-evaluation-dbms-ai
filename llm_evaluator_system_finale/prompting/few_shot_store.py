"""
prompting/few_shot_store.py — Pre-computed few-shot example bank with leakage guard.

Examples are stratified by subtopic and difficulty tier.  The leakage guard
ensures that a question cannot appear as its own few-shot example (Section 4.2.3).

This module exposes `get_examples(question_id, subtopic, difficulty, strategy, n)`
which is called by the prompt builder at generation time.
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select, and_, not_
from sqlalchemy.ext.asyncio import AsyncSession

from models.db_models import FewShotExample, Question, Subtopic


async def get_examples(
    session: AsyncSession,
    question_id: str,
    subtopic_id: str,
    difficulty: str,
    strategy: str = "few-shot",
    n: int = 3,
) -> list[dict]:
    """
    Retrieve up to `n` few-shot examples for the given subtopic and difficulty,
    excluding the question being evaluated (leakage guard).

    Returns a list of dicts with keys: question, answer, reasoning.
    """
    stmt = (
        select(FewShotExample)
        .where(
            and_(
                FewShotExample.subtopic_id == subtopic_id,
                FewShotExample.difficulty == difficulty,
                FewShotExample.prompt_strategy.in_(
                    [strategy, "few-shot-cot"] if "cot" not in strategy else [strategy]
                ),
                FewShotExample.question_id != question_id,  # leakage guard
            )
        )
        .limit(n)
    )
    result = await session.execute(stmt)
    rows = result.scalars().all()

    # If strict difficulty match yields fewer than n, relax difficulty constraint
    if len(rows) < n:
        stmt_relaxed = (
            select(FewShotExample)
            .where(
                and_(
                    FewShotExample.subtopic_id == subtopic_id,
                    FewShotExample.question_id != question_id,
                )
            )
            .limit(n)
        )
        result_r = await session.execute(stmt_relaxed)
        rows = result_r.scalars().all()

    return [
        {
            "question": row.example_question,
            "answer": row.example_answer,
            "reasoning": "",  # populated if CoT examples are stored
        }
        for row in rows[:n]
    ]


async def seed_few_shot_examples(session: AsyncSession) -> int:
    """
    Build the few-shot example store from the existing question bank.

    For each question that has a non-empty expected_answer, insert a
    FewShotExample record (if it does not already exist) using the expected
    answer as the canonical answer.  For CoT strategies, the example_answer
    includes a generic reasoning scaffold.

    Returns the number of new examples inserted.
    """
    stmt = (
        select(Question)
        .join(Subtopic, Question.subtopic_id == Subtopic.id)
        .where(Question.expected_answer.isnot(None))
    )
    result = await session.execute(stmt)
    questions = result.scalars().all()

    inserted = 0
    for q in questions:
        for strategy in ["few-shot", "few-shot-cot"]:
            # Check for existing record
            existing = await session.execute(
                select(FewShotExample).where(
                    and_(
                        FewShotExample.question_id == q.id,
                        FewShotExample.prompt_strategy == strategy,
                    )
                )
            )
            if existing.scalar_one_or_none() is not None:
                continue

            if strategy == "few-shot-cot":
                answer = (
                    "Step 1: Identify the relevant DBMS concept(s).\n"
                    f"[Relevant concept: based on the question context]\n\n"
                    f"Step 2: Apply the relevant definitions and rules.\n"
                    f"[Apply the rules to the given scenario]\n\n"
                    f"Step 3: Final answer:\n{q.expected_answer}"
                )
            else:
                answer = q.expected_answer

            example = FewShotExample(
                question_id=q.id,
                subtopic_id=q.subtopic_id,
                difficulty=q.difficulty,
                prompt_strategy=strategy,
                example_question=q.question_text,
                example_answer=answer,
            )
            session.add(example)
            inserted += 1

    await session.flush()
    return inserted
