"""
judge/elo.py — Elo rating system for model contest rankings.

Implements the standard Elo formula with K=32 as specified in Section 5.4.
Each contest is treated as a multi-player match.  For each (winner, loser)
pair derived from the placement ranking, expected scores are computed and
ratings updated accordingly.

All models start at Elo = 1200 (ELO_INITIAL_RATING in config.py).
"""

from __future__ import annotations

import math
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import ELO_INITIAL_RATING, ELO_K_FACTOR
from models.db_models import EloHistory, Leaderboard, LLMModel


def expected_score(rating_a: int, rating_b: int) -> float:
    """
    Standard Elo expected score for player A when facing player B.
    E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    """
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def updated_rating(
    current_rating: int,
    actual_score: float,
    expected: float,
    k: int = ELO_K_FACTOR,
) -> int:
    """
    Compute the new Elo rating after one match.
    R_new = R_old + K * (S_actual - E_expected)
    Rounds to nearest integer.
    """
    return round(current_rating + k * (actual_score - expected))


def placement_to_score(placement: int, total: int) -> float:
    """
    Convert a contest placement (1 = best) into a normalised actual score
    suitable for Elo calculation.
    1st place  → 1.0
    Last place → 0.0
    Others     → linear interpolation
    """
    if total <= 1:
        return 1.0
    return (total - placement) / (total - 1)


async def update_elo_ratings(
    session: AsyncSession,
    contest_result_id: str,
    ranked_placements: list[dict],
) -> list[dict]:
    """
    Update Elo ratings for all models that participated in a contest.

    Args:
        session:           Async DB session.
        contest_result_id: UUID of the ContestResult row.
        ranked_placements: List of dicts [{model_id, placement}] sorted by placement.

    Returns:
        List of dicts [{model_id, old_rating, new_rating, delta, placement}]
    """
    total = len(ranked_placements)
    updates: list[dict] = []

    # Collect current ratings from the leaderboard
    current_ratings: dict[str, int] = {}
    for entry in ranked_placements:
        model_id = entry["model_id"]
        lb_result = await session.execute(
            select(Leaderboard).where(Leaderboard.model_id == model_id)
        )
        lb = lb_result.scalar_one_or_none()
        current_ratings[model_id] = lb.elo_rating if lb else ELO_INITIAL_RATING

    # Compute new ratings using all pairwise combinations
    new_ratings: dict[str, int] = dict(current_ratings)
    for i, entry_i in enumerate(ranked_placements):
        mid_i = entry_i["model_id"]
        score_i = placement_to_score(entry_i["placement"], total)

        for j, entry_j in enumerate(ranked_placements):
            if i == j:
                continue
            mid_j = entry_j["model_id"]
            score_j = placement_to_score(entry_j["placement"], total)

            e_i = expected_score(current_ratings[mid_i], current_ratings[mid_j])

            # Determine actual score for this pairwise comparison
            if entry_i["placement"] < entry_j["placement"]:
                actual = 1.0   # i beat j
            elif entry_i["placement"] > entry_j["placement"]:
                actual = 0.0   # i lost to j
            else:
                actual = 0.5   # tie

            delta = ELO_K_FACTOR * (actual - e_i) / (total - 1)
            new_ratings[mid_i] = round(new_ratings[mid_i] + delta)

    # Persist updates
    for entry in ranked_placements:
        model_id = entry["model_id"]
        old_rating = current_ratings[model_id]
        new_rating = new_ratings[model_id]
        delta = new_rating - old_rating

        # Upsert leaderboard entry
        lb_result = await session.execute(
            select(Leaderboard).where(Leaderboard.model_id == model_id)
        )
        lb = lb_result.scalar_one_or_none()

        if lb is None:
            lb = Leaderboard(model_id=model_id, elo_rating=new_rating)
            session.add(lb)
        else:
            lb.elo_rating = new_rating
            lb.contest_total = (lb.contest_total or 0) + 1
            if entry["placement"] == 1:
                lb.contest_wins = (lb.contest_wins or 0) + 1
            if lb.contest_total > 0:
                lb.win_rate = (lb.contest_wins or 0) / lb.contest_total

        # Insert Elo history record
        elo_record = EloHistory(
            model_id=model_id,
            contest_result_id=contest_result_id,
            old_rating=old_rating,
            new_rating=new_rating,
            delta=delta,
            placement=entry["placement"],
        )
        session.add(elo_record)

        updates.append({
            "model_id": model_id,
            "old_rating": old_rating,
            "new_rating": new_rating,
            "delta": delta,
            "placement": entry["placement"],
        })

    await session.flush()
    return updates
