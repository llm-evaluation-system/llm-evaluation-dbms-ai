"""
question_bank/loader.py — Load the parsed question bank JSON into PostgreSQL.

This module handles the idempotent insertion of topics, subtopics, and
questions.  Running it multiple times will not create duplicates — existing
records are updated in place via upsert semantics.

Usage (one-time setup):
    python question_bank/loader.py
"""

import asyncio
import json
import sys
import uuid as uuid_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from config import QUESTION_BANK_JSON_PATH
from database import AsyncSessionLocal, create_all_tables
from models.db_models import LLMModel, Question, Subtopic, Topic
from config import CHALLENGER_MODELS, JUDGE_MODEL
from question_bank.sql_fixtures import get_schema_fixture, get_expected_rows, get_judge_hint


def _normalize_id(raw_id: str) -> str:
    """
    Normalize a question ID to PostgreSQL UUID format.

    The question bank JSON uses 12-character hex hashes as IDs.  PostgreSQL
    UUID columns require a full 36-character UUID string (8-4-4-4-12).
    We zero-pad the hash to 32 hex chars and format it as a UUID so the
    mapping is deterministic and reversible.
    """
    # Already a proper UUID — return as-is
    try:
        return str(uuid_module.UUID(raw_id))
    except ValueError:
        pass
    # 12-char (or other-length) hex hash — zero-pad to 32 chars
    padded = raw_id.ljust(32, "0")[:32]
    try:
        return str(uuid_module.UUID(padded))
    except ValueError:
        # Fallback: generate a deterministic UUID v5 from the raw string
        return str(uuid_module.uuid5(uuid_module.NAMESPACE_OID, raw_id))


async def load_topics_and_questions(session: AsyncSession, questions: list[dict]) -> None:
    """
    Upsert all topics, subtopics, and questions into the database.
    Uses Python-level upsert logic (check → insert or update).
    """
    topic_cache: dict[str, str] = {}     # topic_name → topic.id
    subtopic_cache: dict[str, str] = {}  # (topic_id, subtopic_name) → subtopic.id

    for q_data in questions:
        topic_name = q_data["topic"]
        subtopic_name = q_data["subtopic"]

        # ── Topic ──
        if topic_name not in topic_cache:
            result = await session.execute(
                select(Topic).where(Topic.name == topic_name)
            )
            existing_topic = result.scalar_one_or_none()
            if existing_topic is None:
                new_topic = Topic(name=topic_name)
                session.add(new_topic)
                await session.flush()
                topic_cache[topic_name] = new_topic.id
            else:
                topic_cache[topic_name] = existing_topic.id

        topic_id = topic_cache[topic_name]
        cache_key = (topic_id, subtopic_name)

        # ── Subtopic ──
        if cache_key not in subtopic_cache:
            result = await session.execute(
                select(Subtopic).where(
                    Subtopic.topic_id == topic_id,
                    Subtopic.name == subtopic_name,
                )
            )
            existing_sub = result.scalar_one_or_none()
            if existing_sub is None:
                new_sub = Subtopic(topic_id=topic_id, name=subtopic_name)
                session.add(new_sub)
                await session.flush()
                subtopic_cache[cache_key] = new_sub.id
            else:
                subtopic_cache[cache_key] = existing_sub.id

        subtopic_id = subtopic_cache[cache_key]

        # ── Question (upsert by stable normalized UUID) ──
        q_id = _normalize_id(q_data["id"])
        result = await session.execute(
            select(Question).where(Question.id == q_id)
        )
        existing_q = result.scalar_one_or_none()

        # Resolve pre-computed SQL fixtures (schema DDL + expected rows)
        # The raw 12-char hex id is the key used in sql_fixtures.py
        raw_id = q_data["id"]
        fixture_schema = get_schema_fixture(raw_id)
        fixture_rows   = get_expected_rows(raw_id)

        if existing_q is None:
            new_q = Question(
                id=q_id,
                subtopic_id=subtopic_id,
                exercise_number=q_data.get("exercise_number"),
                question_text=q_data["question_text"],
                expected_answer=q_data.get("expected_answer"),
                question_type=q_data["question_type"],
                difficulty=q_data["difficulty"],
                tags=q_data.get("tags", []),
                schema_fixture=fixture_schema,
                expected_rows=fixture_rows,
            )
            session.add(new_q)
        else:
            # Update mutable fields only
            existing_q.question_text = q_data["question_text"]
            existing_q.expected_answer = q_data.get("expected_answer")
            existing_q.question_type = q_data["question_type"]
            existing_q.difficulty = q_data["difficulty"]
            existing_q.tags = q_data.get("tags", [])
            existing_q.subtopic_id = subtopic_id
            # Always refresh fixtures so re-runs pick up any fixture improvements
            if fixture_schema is not None:
                existing_q.schema_fixture = fixture_schema
            if fixture_rows is not None:
                existing_q.expected_rows = fixture_rows


async def load_model_registry(session: AsyncSession) -> None:
    """
    Upsert challenger and judge models into the llm_models table.
    """
    all_models = dict(CHALLENGER_MODELS)
    judge_entry = {
        JUDGE_MODEL["model_id"]: {
            "provider": JUDGE_MODEL["provider"],
            "display_name": JUDGE_MODEL["display_name"],
            "api_model": JUDGE_MODEL["api_model"],
            "cost_per_1k_input_tokens": 0.0025,
            "cost_per_1k_output_tokens": 0.010,
            "max_context_tokens": 128000,
            "supports_seed": True,
            "is_judge": True,
        }
    }

    for model_id, cfg in {**all_models, **judge_entry}.items():
        result = await session.execute(
            select(LLMModel).where(LLMModel.model_id == model_id)
        )
        existing = result.scalar_one_or_none()
        is_judge = cfg.get("is_judge", False)

        if existing is None:
            new_model = LLMModel(
                model_id=model_id,
                display_name=cfg["display_name"],
                provider=cfg["provider"],
                api_model=cfg["api_model"],
                is_judge=is_judge,
                cost_per_1k_input_tokens=cfg.get("cost_per_1k_input_tokens"),
                cost_per_1k_output_tokens=cfg.get("cost_per_1k_output_tokens"),
                max_context_tokens=cfg.get("max_context_tokens"),
                supports_seed=cfg.get("supports_seed", False),
            )
            session.add(new_model)
        else:
            existing.display_name = cfg["display_name"]
            existing.api_model = cfg["api_model"]
            existing.is_judge = is_judge
            existing.cost_per_1k_input_tokens = cfg.get("cost_per_1k_input_tokens")
            existing.cost_per_1k_output_tokens = cfg.get("cost_per_1k_output_tokens")
            existing.max_context_tokens = cfg.get("max_context_tokens")
            existing.supports_seed = cfg.get("supports_seed", False)


async def run_loader(question_bank_path: str = QUESTION_BANK_JSON_PATH) -> None:
    await create_all_tables()

    # Parse if JSON doesn't exist yet
    if not Path(question_bank_path).exists():
        print(f"JSON not found at {question_bank_path}. Running parser first…")
        from question_bank.parser import parse_question_bank, save_question_bank_json
        questions = parse_question_bank()
        save_question_bank_json(questions, question_bank_path)
    else:
        with open(question_bank_path, encoding="utf-8") as f:
            questions = json.load(f)

    print(f"Loading {len(questions)} questions into PostgreSQL…")

    async with AsyncSessionLocal() as session:
        async with session.begin():
            await load_model_registry(session)
            await load_topics_and_questions(session, questions)

    print("Question bank and model registry loaded successfully.")


if __name__ == "__main__":
    asyncio.run(run_loader())
