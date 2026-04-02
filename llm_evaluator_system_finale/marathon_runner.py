#!/usr/bin/env python3
"""
marathon_runner.py — Full Spec-Compliant Benchmark Runner
==========================================================

WHAT THIS REPLACES
------------------
The original marathon_runner dispatched everything to Celery asynchronously
and polled a completion counter in a sleep loop. With 1,000+ jobs queued to
a single-concurrency worker, this took 3 days.

HOW THIS IS FAST
----------------
Every call uses /eval/generate with async_run=False (the default). The
FastAPI handler calls the LLM synchronously and returns only when the answer
is written to the database. No Celery, no Redis, no polling loops.

Client-side parallelism (ThreadPoolExecutor, max_workers=3) keeps 3 LLM
calls in flight at once without overwhelming Groq's rate limits (30 req/min).
Groq's 429 handler in providers.py already sleeps 65s and retries — we don't
need extra backoff on this side.

FULL SPEC COVERAGE
------------------
Phase 0  Health-check + seed few-shot examples
Phase 1  ALL 80 questions × 2 models × zero-shot (default HP)
         → judge/score every run
         → pairwise contest for every question
Phase 2  ALL 9 strategies × 20 stratified questions × 2 models
         (one question per subtopic, covers every DBMS domain)
         → judge/score every run
         → prompts/compare report per model
Phase 3  Hyperparam sweep on ALL 80 questions × 2 models × zero-shot
         temperature: 0.0, 0.7, 1.0  (0.3 is Phase 1's default)
         top_p:       0.7, 0.85, 1.0 (0.9 is Phase 1's default)
         → hyperparams/compare report per model (temperature + top_p)
Phase 4  Leaderboard refresh + full export

OUTPUT (saved to final_results/)
---------------------------------
  leaderboard.json
  results_summary.json
  full_export.json
  hyperparams_llama.json
  hyperparams_gpt4o.json
  prompts_compare_llama.json
  prompts_compare_gpt4o.json

USAGE
-----
  python marathon_runner.py [--base-url http://localhost:8000]
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Optional

# ── Configuration ─────────────────────────────────────────────────────────────

MODELS = ["llama-3.1-70b", "gpt-4o"]

ALL_STRATEGIES = [
    "zero-shot",
    "one-shot",
    "few-shot",
    "cot",
    "few-shot-cot",
    "self-consistency",
    "role-prompting",
    "least-to-most",
    "react",
]

DEFAULT_HP = {"temperature": 0.3, "top_p": 0.9, "max_tokens": 1024}

# Hyperparam sweep values excluding the default (already covered in Phase 1)
TEMPERATURE_SWEEP = [0.0, 0.7, 1.0]   # 0.3 = DEFAULT_HP["temperature"]
TOP_P_SWEEP       = [0.7, 0.85, 1.0]  # 0.9 = DEFAULT_HP["top_p"]

# Parallelism: 3 concurrent LLM calls — safe under Groq's 30 req/min limit
# (3 workers × ~10s avg latency = ~18 req/min peak)
MAX_WORKERS = 3

OUT_DIR = "final_results"


# ── HTTP layer ────────────────────────────────────────────────────────────────

def _http(base: str, method: str, path: str,
          payload: Optional[dict] = None, timeout: int = 300) -> dict:
    """
    Single HTTP call. Returns {"ok": bool, "body": any, "error": str|None}.
    Never raises — all exceptions are caught and returned in "error".
    """
    url = base.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return {"ok": True, "body": json.loads(resp.read().decode()), "error": None}
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode()[:300]
        except Exception:
            detail = str(exc)
        return {"ok": False, "body": None, "error": f"HTTP {exc.code}: {detail}"}
    except Exception as exc:
        return {"ok": False, "body": None, "error": str(exc)[:300]}


def GET(base: str, path: str, timeout: int = 60) -> dict:
    return _http(base, "GET", path, timeout=timeout)


def POST(base: str, path: str, payload: dict, timeout: int = 300) -> dict:
    return _http(base, "POST", path, payload=payload, timeout=timeout)


# ── Logging ───────────────────────────────────────────────────────────────────

_log_lines: list[str] = []


def log(msg: str, indent: int = 0) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {'  ' * indent}{msg}"
    print(line, flush=True)
    _log_lines.append(line)


def section(title: str) -> None:
    bar = "=" * 68
    log("")
    log(bar)
    log(f"  {title}")
    log(bar)


def save(filename: str, data: Any) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, filename), "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── Core operations ───────────────────────────────────────────────────────────

def generate_sync(base: str, model_id: str, question_id: str,
                  strategy: str, hyperparams: dict) -> dict:
    """
    POST /eval/generate with async_run=False.
    Blocks until the LLM responds. Returns enriched result dict.
    """
    payload = {
        "model_id": model_id,
        "question_id": question_id,
        "prompt_strategy": strategy,
        "hyperparams": hyperparams,
        "async_run": False,
    }
    t0 = time.time()
    r = POST(base, "/eval/generate", payload, timeout=360)
    elapsed = round((time.time() - t0) * 1000)

    result = {
        "model_id": model_id,
        "question_id": question_id,
        "strategy": strategy,
        "hyperparams": hyperparams,
        "ok": r["ok"],
        "error": r["error"],
        "latency_ms": elapsed,
        "run_id": None,
        "status": None,
    }
    if r["ok"] and r["body"]:
        b = r["body"]
        result["run_id"] = b.get("run_id")
        result["status"] = b.get("status")
        result["input_tokens"] = b.get("input_tokens")
        result["output_tokens"] = b.get("output_tokens")
        result["cost_usd"] = b.get("cost_usd")
    return result


def judge_score(base: str, run_id: str, model_id: str, question_id: str) -> dict:
    """POST /eval/judge/score for a completed run."""
    r = POST(base, "/eval/judge/score", {
        "run_id": run_id,
        "model_id": model_id,
        "question_id": question_id,
    }, timeout=360)
    result = {"run_id": run_id, "model_id": model_id,
              "question_id": question_id, "ok": r["ok"], "error": r["error"]}
    if r["ok"] and r["body"]:
        b = r["body"]
        result.update({
            "score_id": b.get("score_id"),
            "judge_score_0_10": b.get("judge_score_0_10"),
            "master_composite_score": b.get("master_composite_score"),
            "db_correctness_score": b.get("db_correctness_score"),
            "llm_quality_score": b.get("llm_quality_score"),
            "prompting_effectiveness_score": b.get("prompting_effectiveness_score"),
            "efficiency_score": b.get("efficiency_score"),
            "hallucinations_detected": b.get("hallucinations_detected", []),
            "missing_points": b.get("missing_points", []),
            "justification_preview": (b.get("justification") or "")[:200],
            "sql_execution_details": b.get("sql_execution_details"),
        })
    return result


def run_parallel(tasks: list[dict], base: str, desc: str) -> list[dict]:
    """
    Execute a list of task dicts in parallel (MAX_WORKERS at a time).
    Each task dict must have "fn" (callable) and "kwargs" (dict).
    Returns list of result dicts with progress logged.
    """
    total = len(tasks)
    results = []
    done = 0
    ok_count = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(t["fn"], **t["kwargs"]): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if result.get("ok"):
                ok_count += 1
            # Progress line
            model = result.get("model_id", "")[:15]
            strat = result.get("strategy", "")[:14]
            rid = (result.get("run_id") or "")[:8]
            mcs_str = ""
            if "master_composite_score" in result and result["master_composite_score"] is not None:
                mcs_str = f" MCS={result['master_composite_score']:.1f}"
            err_str = f" ERR={result['error'][:50]}" if result.get("error") else ""
            mark = "✓" if result.get("ok") else "✗"
            log(f"{mark} [{done:4d}/{total}] {desc} | {model} | {strat} "
                f"| run={rid}…{mcs_str}{err_str}", indent=1)
            results.append(result)

    log(f"  {ok_count}/{total} succeeded for: {desc}")
    return results


# ── Stratified sample: 1 question per subtopic ────────────────────────────────

def pick_stratified_sample(all_questions: list[dict]) -> list[dict]:
    """
    Select exactly one question per subtopic (20 subtopics → 20 questions).
    Preference order: medium > easy > hard (most representative difficulty).
    """
    by_subtopic: dict[str, list[dict]] = {}
    for q in all_questions:
        by_subtopic.setdefault(q["subtopic"], []).append(q)

    selected = []
    for subtopic, qs in sorted(by_subtopic.items()):
        # Prefer medium, then easy, then hard
        for diff in ["medium", "easy", "hard"]:
            candidates = [q for q in qs if q["difficulty"] == diff]
            if candidates:
                selected.append(candidates[0])
                break
    return selected


# ── Main ──────────────────────────────────────────────────────────────────────

def main(base_url: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    run_start = time.time()

    # ── Phase 0: Health + Seed ────────────────────────────────────────────────
    section("PHASE 0 — Health Check + Seed Few-Shot Examples")

    h = GET(base_url, "/health")
    if not h["ok"]:
        print(f"\nFATAL: Cannot reach {base_url} — {h['error']}")
        print("Start the stack first: docker compose up -d")
        sys.exit(1)
    log(f"API healthy: {h['body']}")

    seed = POST(base_url, "/eval/seed-examples", {})
    log(f"Seed examples: {'ok' if seed['ok'] else seed['error']}")

    # Fetch all 80 questions
    r = GET(base_url, "/eval/questions?limit=200")
    if not r["ok"]:
        print(f"FATAL: Cannot fetch questions: {r['error']}")
        sys.exit(1)
    all_questions = r["body"].get("questions", [])
    all_q_ids = [q["id"] for q in all_questions]
    log(f"Loaded {len(all_questions)} questions from DB")

    if len(all_questions) < 80:
        log(f"WARNING: Expected 80 questions, got {len(all_questions)}. "
            "Run question_bank/loader.py first.")

    # Build ID → metadata map
    q_meta = {q["id"]: q for q in all_questions}

    # Stratified sample: one question per subtopic (for strategy comparison)
    strat_sample = pick_stratified_sample(all_questions)
    strat_q_ids = [q["id"] for q in strat_sample]
    log(f"Stratified sample: {len(strat_sample)} questions "
        f"(1 per subtopic, for all-9-strategy comparison)")
    for q in strat_sample:
        log(f"  {q['id'][:8]}… {q['difficulty']:6} {q['question_type']:11} "
            f"{q['subtopic']}", indent=1)

    # ── Phase 1: All 80 questions × 2 models × zero-shot ──────────────────────
    section("PHASE 1 — All 80 Questions × 2 Models × Zero-Shot (Default HP)")
    log(f"  {len(all_q_ids)} questions × {len(MODELS)} models = "
        f"{len(all_q_ids) * len(MODELS)} generate calls, then judge + contest")

    p1_tasks = [
        {
            "fn": generate_sync,
            "kwargs": dict(base=base_url, model_id=m, question_id=qid,
                           strategy="zero-shot", hyperparams=DEFAULT_HP),
        }
        for qid in all_q_ids
        for m in MODELS
    ]
    p1_results = run_parallel(p1_tasks, base_url, "gen/zero-shot")
    save("phase1_generate.json", p1_results)

    # Judge score all successful Phase 1 runs
    log("\n  Scoring Phase 1 runs...")
    p1_score_tasks = [
        {
            "fn": judge_score,
            "kwargs": dict(base=base_url, run_id=r["run_id"],
                           model_id=r["model_id"], question_id=r["question_id"]),
        }
        for r in p1_results
        if r["ok"] and r.get("run_id") and r["run_id"] not in ("pending", None)
    ]
    p1_scores = run_parallel(p1_score_tasks, base_url, "judge/score")
    save("phase1_scores.json", p1_scores)

    # Pairwise contests for ALL 80 questions (using Phase 1 zero-shot run_ids)
    log("\n  Running pairwise contests for all 80 questions...")
    # Build: question_id → {model_id → run_id}
    q_model_runs: dict[str, dict[str, str]] = {}
    for r in p1_results:
        if r["ok"] and r.get("run_id") and r["run_id"] not in ("pending", None):
            q_model_runs.setdefault(r["question_id"], {})[r["model_id"]] = r["run_id"]

    contest_results = []
    contested = 0
    skipped = 0
    for qid in all_q_ids:
        model_runs = q_model_runs.get(qid, {})
        if len(model_runs) < 2:
            skipped += 1
            contest_results.append({
                "question_id": qid, "ok": False,
                "error": f"Only {len(model_runs)} model answer(s) available"
            })
            continue

        r = POST(base_url, "/eval/judge/contest", {
            "question_id": qid,
            "run_ids": list(model_runs.values()),
        }, timeout=360)

        entry = {
            "question_id": qid,
            "subtopic": q_meta.get(qid, {}).get("subtopic", "?"),
            "difficulty": q_meta.get(qid, {}).get("difficulty", "?"),
            "question_type": q_meta.get(qid, {}).get("question_type", "?"),
            "model_run_map": model_runs,
            "ok": r["ok"],
            "error": r["error"],
        }
        if r["ok"] and r["body"]:
            b = r["body"]
            entry.update({
                "contest_id": b.get("contest_id"),
                "ranked_model_ids": b.get("ranked_model_ids"),
                "tie_exists": b.get("tie_exists"),
                "elo_updates": b.get("elo_updates"),
                "judge_reasoning_preview": (b.get("judge_reasoning") or "")[:300],
                "ranking_with_scores": b.get("ranking_with_scores"),
            })
            winner = (b.get("ranked_model_ids") or [None])[0]
            log(f"  ✓ contest q={qid[:8]}… winner={winner} "
                f"tie={b.get('tie_exists')} elo={b.get('elo_updates')}", indent=1)
            contested += 1
        else:
            log(f"  ✗ contest q={qid[:8]}… {r['error']}", indent=1)
        contest_results.append(entry)

    log(f"  Contests: {contested} run, {skipped} skipped")
    save("phase1_contests.json", contest_results)

    # ── Phase 2: All 9 Strategies × Stratified 20 Questions × 2 Models ────────
    section("PHASE 2 — All 9 Prompting Strategies × 20 Stratified Questions × 2 Models")
    other_strategies = [s for s in ALL_STRATEGIES if s != "zero-shot"]
    log(f"  {len(other_strategies)} additional strategies × {len(strat_q_ids)} questions "
        f"× {len(MODELS)} models = "
        f"{len(other_strategies) * len(strat_q_ids) * len(MODELS)} generate calls")
    log("  (zero-shot already done in Phase 1 for these questions)")

    p2_tasks = [
        {
            "fn": generate_sync,
            "kwargs": dict(base=base_url, model_id=m, question_id=qid,
                           strategy=s, hyperparams=DEFAULT_HP),
        }
        for s in other_strategies
        for qid in strat_q_ids
        for m in MODELS
    ]
    p2_results = run_parallel(p2_tasks, base_url, "gen/strategies")
    save("phase2_generate.json", p2_results)

    # Judge score all Phase 2 runs
    log("\n  Scoring Phase 2 runs...")
    p2_score_tasks = [
        {
            "fn": judge_score,
            "kwargs": dict(base=base_url, run_id=r["run_id"],
                           model_id=r["model_id"], question_id=r["question_id"]),
        }
        for r in p2_results
        if r["ok"] and r.get("run_id") and r["run_id"] not in ("pending", None)
    ]
    p2_scores = run_parallel(p2_score_tasks, base_url, "judge/score/strat")
    save("phase2_scores.json", p2_scores)

    # ── Phase 3: Hyperparam Sweep — All 80 Questions × 2 Models × Zero-Shot ───
    section("PHASE 3 — Hyperparam Sweep: Temperature + Top-P (All 80 Questions)")
    log(f"  Temperature sweep values (excl. default 0.3): {TEMPERATURE_SWEEP}")
    log(f"  Top-P sweep values (excl. default 0.9): {TOP_P_SWEEP}")
    sweep_hp_configs = (
        [{**DEFAULT_HP, "temperature": t} for t in TEMPERATURE_SWEEP] +
        [{**DEFAULT_HP, "top_p": tp} for tp in TOP_P_SWEEP]
    )
    log(f"  {len(sweep_hp_configs)} HP configs × {len(all_q_ids)} questions "
        f"× {len(MODELS)} models = "
        f"{len(sweep_hp_configs) * len(all_q_ids) * len(MODELS)} generate calls")
    log("  Note: no judge scoring for hyperparam sweep runs — "
        "hyperparams/compare uses MCS from the scored zero-shot runs")

    p3_tasks = [
        {
            "fn": generate_sync,
            "kwargs": dict(base=base_url, model_id=m, question_id=qid,
                           strategy="zero-shot", hyperparams=hp),
        }
        for hp in sweep_hp_configs
        for qid in all_q_ids
        for m in MODELS
    ]
    p3_results = run_parallel(p3_tasks, base_url, "gen/hyperparam-sweep")
    save("phase3_hyperparam_generate.json", p3_results)

    # Judge score hyperparam sweep runs too — required for hyperparams/compare
    # which reads master_composite_score grouped by hyperparam value
    log("\n  Scoring Phase 3 (hyperparam sweep) runs for sensitivity analysis...")
    p3_score_tasks = [
        {
            "fn": judge_score,
            "kwargs": dict(base=base_url, run_id=r["run_id"],
                           model_id=r["model_id"], question_id=r["question_id"]),
        }
        for r in p3_results
        if r["ok"] and r.get("run_id") and r["run_id"] not in ("pending", None)
    ]
    p3_scores = run_parallel(p3_score_tasks, base_url, "judge/score/hp")
    save("phase3_scores.json", p3_scores)

    # ── Phase 4: Reporting Endpoints ──────────────────────────────────────────
    section("PHASE 4 — Reporting: Leaderboard + All Comparison Endpoints")

    # Refresh leaderboard (recomputes Elo, best/worst topic, win rates)
    log("Refreshing leaderboard...")
    r_refresh = POST(base_url, "/eval/leaderboard/refresh", {}, timeout=60)
    log(f"  {'✓' if r_refresh['ok'] else '✗'} refresh: {r_refresh['body']}")

    # Leaderboard
    log("GET /eval/leaderboard")
    r_lb = GET(base_url, "/eval/leaderboard", timeout=60)
    if r_lb["ok"] and r_lb["body"]:
        lb = r_lb["body"]
        log(f"  total_models={lb.get('total_models')} "
            f"total_contests={lb.get('total_contests')} "
            f"total_runs={lb.get('total_runs')}")
        for e in lb.get("entries", []):
            log(f"  Rank {e.get('rank')}: {e.get('model_id'):18} "
                f"MCS={e.get('mcs_score')} DB={e.get('db_correctness')} "
                f"LLM={e.get('llm_quality')} Elo={e.get('elo_rating')} "
                f"Wins={e.get('contest_wins')}/{e.get('contest_total')} "
                f"Hall={e.get('hallucination_rate')}", indent=1)
    save("leaderboard.json", r_lb["body"] or {"error": r_lb["error"]})

    # Results summary
    log("GET /eval/results/summary")
    r_summary = GET(base_url, "/eval/results/summary?limit=1000", timeout=60)
    if r_summary["ok"] and r_summary["body"]:
        rows = r_summary["body"].get("results", [])
        log(f"  {len(rows)} rows returned")
        for row in rows:
            log(f"  {row.get('model_id'):18} | {row.get('subtopic',''):35} "
                f"| {row.get('prompt_strategy',''):16} "
                f"| MCS={row.get('avg_mcs','?'):5} "
                f"| DB={row.get('avg_db_correctness','?'):5} "
                f"| n={row.get('question_count','?')}", indent=1)
    save("results_summary.json", r_summary["body"] or {"error": r_summary["error"]})

    # Hyperparams compare — temperature and top_p per model
    log("GET /eval/hyperparams/compare (temperature + top_p per model)")
    for model_id, filename in [("llama-3.1-70b", "hyperparams_llama.json"),
                                ("gpt-4o", "hyperparams_gpt4o.json")]:
        combined = {"model_id": model_id, "parameters": {}}
        for param in ["temperature", "top_p"]:
            path = (f"/eval/hyperparams/compare"
                    f"?model_id={urllib.parse.quote(model_id)}"
                    f"&param_name={param}")
            r = GET(base_url, path, timeout=60)
            if r["ok"] and r["body"]:
                b = r["body"]
                combined["parameters"][param] = {
                    "results": b.get("results", []),
                    "recommended_value": b.get("recommended_value"),
                    "sensitivity": b.get("sensitivity", "UNKNOWN"),
                }
                log(f"  {model_id} | {param}: sensitivity={b.get('sensitivity')} "
                    f"recommended={b.get('recommended_value')} "
                    f"n_values={len(b.get('results', []))}", indent=1)
            else:
                combined["parameters"][param] = {
                    "results": [], "recommended_value": None,
                    "sensitivity": "UNKNOWN", "error": r["error"],
                }
                log(f"  {model_id} | {param}: FAILED — {r['error']}", indent=1)
        save(filename, combined)

    # Prompts compare per model
    log("GET /eval/prompts/compare per model")
    for model_id, filename in [("llama-3.1-70b", "prompts_compare_llama.json"),
                                ("gpt-4o", "prompts_compare_gpt4o.json")]:
        path = f"/eval/prompts/compare?model_id={urllib.parse.quote(model_id)}"
        r = GET(base_url, path, timeout=60)
        if r["ok"] and r["body"]:
            b = r["body"]
            strats = b.get("strategies", [])
            log(f"  {model_id}: {len(strats)} strategies compared | "
                f"recommended={b.get('recommended_strategy')} | "
                f"zero-shot baseline MCS={b.get('zero_shot_baseline_mcs')}", indent=1)
            for s in strats:
                log(f"    {s.get('strategy'):16} MCS={s.get('avg_mcs'):5} "
                    f"lift={s.get('accuracy_lift_vs_zeroshot'):+.3f} "
                    f"sigma={s.get('consistency_sigma'):.3f}", indent=2)
        else:
            log(f"  {model_id}: FAILED — {r['error']}", indent=1)
        save(filename, r["body"] or {"error": r["error"]})

    # Full export
    log("GET /eval/export/json")
    r_export = GET(base_url, "/eval/export/json", timeout=120)
    save("full_export.json", r_export["body"] or {"error": r_export["error"]})
    if r_export["ok"] and isinstance(r_export["body"], list):
        export_data = r_export["body"]
        n_scored = sum(1 for e in export_data if e.get("mcs") is not None)
        n_null = sum(1 for e in export_data if e.get("mcs") is None)
        log(f"  {len(export_data)} total runs | {n_scored} scored | {n_null} unscored")

    # ── Summary ───────────────────────────────────────────────────────────────
    section("RUN COMPLETE")
    elapsed = round(time.time() - run_start)

    p1_ok = sum(1 for r in p1_results if r["ok"])
    p1_scored = sum(1 for r in p1_scores if r["ok"])
    p2_ok = sum(1 for r in p2_results if r["ok"])
    p2_scored = sum(1 for r in p2_scores if r["ok"])
    p3_ok = sum(1 for r in p3_results if r["ok"])
    p3_scored = sum(1 for r in p3_scores if r["ok"])
    c_ok = sum(1 for r in contest_results if r["ok"])

    summary_lines = [
        "=" * 68,
        "  MARATHON RUNNER — FINAL SUMMARY",
        "=" * 68,
        f"  Elapsed time           : {elapsed}s ({elapsed // 60}m {elapsed % 60}s)",
        f"  Base URL               : {base_url}",
        "",
        "  PHASE 1 — Zero-Shot, All 80 Questions",
        f"    Generate calls        : {p1_ok}/{len(p1_results)} succeeded",
        f"    Judge scores          : {p1_scored}/{len(p1_score_tasks)} succeeded",
        f"    Contests              : {contested}/{len(all_q_ids)} questions",
        "",
        "  PHASE 2 — All 9 Strategies, 20 Stratified Questions",
        f"    Generate calls        : {p2_ok}/{len(p2_tasks)} succeeded",
        f"    Judge scores          : {p2_scored}/{len(p2_score_tasks)} succeeded",
        "",
        "  PHASE 3 — Hyperparam Sweep (temp + top_p), All 80 Questions",
        f"    Generate calls        : {p3_ok}/{len(p3_tasks)} succeeded",
        f"    Judge scores          : {p3_scored}/{len(p3_score_tasks)} succeeded",
        "",
        "  OUTPUT FILES  (in final_results/)",
        "    leaderboard.json",
        "    results_summary.json",
        "    full_export.json",
        "    hyperparams_llama.json",
        "    hyperparams_gpt4o.json",
        "    prompts_compare_llama.json",
        "    prompts_compare_gpt4o.json",
        "    phase1_generate.json   (raw generate results)",
        "    phase1_scores.json     (judge scores)",
        "    phase1_contests.json   (contest results + Elo updates)",
        "    phase2_generate.json",
        "    phase2_scores.json",
        "    phase3_hyperparam_generate.json",
        "    phase3_scores.json",
        "=" * 68,
    ]

    summary_text = "\n".join(summary_lines)
    print("\n" + summary_text)
    save("SUMMARY.txt", summary_text)
    save("run_log.txt", "\n".join(_log_lines))

    # Save a machine-readable audit record
    save("audit.json", {
        "run_started_at": datetime.fromtimestamp(run_start).isoformat(),
        "run_ended_at": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "base_url": base_url,
        "models": MODELS,
        "all_strategies": ALL_STRATEGIES,
        "all_questions_count": len(all_questions),
        "stratified_sample_count": len(strat_sample),
        "temperature_sweep": TEMPERATURE_SWEEP,
        "top_p_sweep": TOP_P_SWEEP,
        "phase1": {
            "generates_ok": p1_ok, "generates_total": len(p1_results),
            "scores_ok": p1_scored, "scores_total": len(p1_score_tasks),
            "contests_ok": contested, "contests_total": len(all_q_ids),
        },
        "phase2": {
            "generates_ok": p2_ok, "generates_total": len(p2_results),
            "scores_ok": p2_scored, "scores_total": len(p2_score_tasks),
        },
        "phase3": {
            "generates_ok": p3_ok, "generates_total": len(p3_results),
            "scores_ok": p3_scored, "scores_total": len(p3_score_tasks),
        },
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full spec-compliant LLM Evaluator benchmark runner"
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000",
        help="FastAPI base URL (default: http://localhost:8000)"
    )
    args = parser.parse_args()
    main(args.base_url)
