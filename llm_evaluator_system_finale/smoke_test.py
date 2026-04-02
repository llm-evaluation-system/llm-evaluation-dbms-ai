#!/usr/bin/env python3
"""
smoke_test.py — Run all 13 evaluation questions end-to-end.

For each question:
  1. POST /eval/generate  → get run_id
  2. POST /eval/judge/score (using run_id from step 1)

Results are saved to smoke_test_results.json and printed as a summary table.

Usage:
    python3 smoke_test.py [--base-url http://localhost:8000]
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import urllib.request
import urllib.error

# ── Configuration ─────────────────────────────────────────────────────────────

# Exact hyperparams from the original smoke test run (Evaluation.md)
HYPERPARAMS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "max_tokens": 1024,
    "top_k": -1,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "system_prompt_style": "expert-persona",
    "seed": 0,
}

# All 13 questions in the order they appear in Evaluation.md
QUESTIONS = [
    {"num": 1,  "question_id": "0def7bdc-534e-0000-0000-000000000000"},
    {"num": 2,  "question_id": "c6c4ce74-44f2-0000-0000-000000000000"},
    {"num": 3,  "question_id": "0c053e7b-ed52-0000-0000-000000000000"},
    {"num": 4,  "question_id": "6c3f53eb-80bb-0000-0000-000000000000"},
    {"num": 5,  "question_id": "c36cf025-1982-0000-0000-000000000000"},
    {"num": 6,  "question_id": "4d30f627-adcf-0000-0000-000000000000"},
    {"num": 7,  "question_id": "d9c7e0be-6bc3-0000-0000-000000000000"},
    {"num": 8,  "question_id": "d1d97baf-9e4c-0000-0000-000000000000"},
    {"num": 9,  "question_id": "f4ddd3bd-b9b0-0000-0000-000000000000"},
    {"num": 10, "question_id": "f31cf7c1-6458-0000-0000-000000000000"},
    {"num": 11, "question_id": "e6f3c650-6091-0000-0000-000000000000"},
    {"num": 12, "question_id": "be6d6e30-18b2-0000-0000-000000000000"},
    {"num": 13, "question_id": "a16ad64a-b216-0000-0000-000000000000"},
]

MODEL_ID = "llama-3.1-70b"
PROMPT_STRATEGY = "zero-shot"


# ── HTTP helper ────────────────────────────────────────────────────────────────

def post(url: str, payload: dict, timeout: int = 120) -> dict:
    """POST JSON to url, return parsed response. Raises on HTTP errors."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e


# ── Main runner ────────────────────────────────────────────────────────────────

def run_all(base_url: str) -> list[dict]:
    generate_url = f"{base_url}/eval/generate"
    judge_url    = f"{base_url}/eval/judge/score"

    results = []
    total = len(QUESTIONS)

    print(f"\n{'═'*72}")
    print(f"  Smoke test — {total} questions  |  model: {MODEL_ID}")
    print(f"  Base URL : {base_url}")
    print(f"  Started  : {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"{'═'*72}\n")

    for q in QUESTIONS:
        num        = q["num"]
        qid        = q["question_id"]
        row: dict  = {"question_num": num, "question_id": qid}

        print(f"Q{num:02d}/{total}  {qid}")

        # ── Step 1: Generate ──────────────────────────────────────────────────
        gen_payload = {
            "model_id":        MODEL_ID,
            "question_id":     qid,
            "prompt_strategy": PROMPT_STRATEGY,
            "hyperparams":     HYPERPARAMS,
            "async_run":       False,
        }
        t0 = time.monotonic()
        try:
            gen_resp = post(generate_url, gen_payload)
        except Exception as exc:
            row["generate_error"] = str(exc)
            row["status"] = "generate_failed"
            print(f"       ❌  generate failed: {exc}\n")
            results.append(row)
            continue
        gen_ms = (time.monotonic() - t0) * 1000

        run_id = gen_resp.get("run_id")
        status = gen_resp.get("status", "unknown")
        tokens_out = gen_resp.get("output_tokens", "?")
        row.update({
            "run_id":         run_id,
            "generate_status": status,
            "input_tokens":   gen_resp.get("input_tokens"),
            "output_tokens":  gen_resp.get("output_tokens"),
            "cost_usd":       gen_resp.get("cost_usd"),
            "generate_latency_ms": round(gen_ms),
        })

        if status != "completed" or not run_id:
            row["status"] = "generate_incomplete"
            print(f"       ⚠   status={status}  run_id={run_id}\n")
            results.append(row)
            continue

        print(f"       ✓  generate  run_id={run_id}  tokens={tokens_out}  {round(gen_ms)}ms")

        # ── Step 2: Judge ─────────────────────────────────────────────────────
        judge_payload = {
            "model_id":    MODEL_ID,
            "question_id": qid,
            "run_id":      run_id,
        }
        t1 = time.monotonic()
        try:
            judge_resp = post(judge_url, judge_payload)
        except Exception as exc:
            row["judge_error"] = str(exc)
            row["status"] = "judge_failed"
            print(f"       ❌  judge failed: {exc}\n")
            results.append(row)
            continue
        judge_ms = (time.monotonic() - t1) * 1000

        score        = judge_resp.get("judge_score_0_10")
        mcs          = judge_resp.get("master_composite_score")
        db_score     = judge_resp.get("db_correctness_score")
        sql_details  = judge_resp.get("sql_execution_details")
        hallucinations = judge_resp.get("hallucinations_detected", [])
        missing        = judge_resp.get("missing_points", [])

        row.update({
            "status":                   "ok",
            "judge_score_0_10":         score,
            "master_composite_score":   mcs,
            "db_correctness_score":     db_score,
            "llm_quality_score":        judge_resp.get("llm_quality_score"),
            "prompting_effectiveness":  judge_resp.get("prompting_effectiveness_score"),
            "efficiency_score":         judge_resp.get("efficiency_score"),
            "justification":            judge_resp.get("justification"),
            "hallucinations_detected":  hallucinations,
            "missing_points":           missing,
            "sql_execution_details":    sql_details,
            "judge_latency_ms":         round(judge_ms),
            # full responses preserved for downstream analysis
            "_generate_response":       gen_resp,
            "_judge_response":          judge_resp,
        })

        # Console summary line
        harness_icon = ""
        if sql_details:
            f1   = sql_details.get("result_set_f1")
            syn  = sql_details.get("syntactic_parse_success")
            harness_icon = f"  harness: parse={syn} f1={f1}"

        hall_summary = ""
        if hallucinations:
            sevs = [h.get("severity","?") for h in hallucinations]
            hall_summary = f"  hallucinations={sevs}"

        print(f"       ✓  judge     score={score}/10  mcs={round(mcs,1) if mcs else '?'}"
              f"  db={round(db_score,1) if db_score else '?'}"
              f"{harness_icon}{hall_summary}  {round(judge_ms)}ms")
        print()

        results.append(row)

    return results


def print_summary(results: list[dict]) -> None:
    print(f"\n{'═'*72}")
    print("  RESULTS SUMMARY")
    print(f"{'═'*72}")
    header = f"{'Q':>3}  {'question_id':<40}  {'score':>5}  {'MCS':>6}  {'DB':>6}  {'F1':>5}  status"
    print(header)
    print("─" * 80)

    for r in results:
        num   = r["question_num"]
        qid   = r["question_id"]
        score = r.get("judge_score_0_10", "")
        mcs   = r.get("master_composite_score", "")
        db    = r.get("db_correctness_score", "")
        status = r.get("status", "unknown")

        f1 = ""
        sd = r.get("sql_execution_details")
        if sd:
            f1 = sd.get("result_set_f1", "")

        score_str = f"{score:>5.1f}" if isinstance(score, (int, float)) else f"{'?':>5}"
        mcs_str   = f"{mcs:>6.1f}"   if isinstance(mcs,   (int, float)) else f"{'?':>6}"
        db_str    = f"{db:>6.1f}"    if isinstance(db,    (int, float)) else f"{'?':>6}"
        f1_str    = f"{f1:>5.2f}"    if isinstance(f1,    (int, float)) else f"{'—':>5}"

        icon = "✅" if status == "ok" else "❌"
        print(f"{icon} {num:>2}  {qid:<40}  {score_str}  {mcs_str}  {db_str}  {f1_str}  {status}")

    ok_count   = sum(1 for r in results if r.get("status") == "ok")
    fail_count = len(results) - ok_count
    scores     = [r["judge_score_0_10"] for r in results if isinstance(r.get("judge_score_0_10"), (int, float))]
    avg_score  = sum(scores) / len(scores) if scores else 0

    print("─" * 80)
    print(f"  Passed: {ok_count}/{len(results)}   Failed: {fail_count}   Avg judge score: {avg_score:.2f}/10")
    print(f"{'═'*72}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 13-question smoke test against the eval API.")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the running FastAPI server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--output",
        default="smoke_test_results.json",
        help="File to write full results JSON to (default: smoke_test_results.json)",
    )
    args = parser.parse_args()

    # Quick health check
    try:
        health_req = urllib.request.Request(f"{args.base_url}/health")
        with urllib.request.urlopen(health_req, timeout=5):
            pass
    except Exception:
        # /health may not exist — try the docs endpoint as fallback
        try:
            docs_req = urllib.request.Request(f"{args.base_url}/docs")
            with urllib.request.urlopen(docs_req, timeout=5):
                pass
        except Exception:
            print(f"\n⚠  Could not reach {args.base_url} — is the server running?")
            print("   Start it with:  docker compose up -d\n")
            sys.exit(1)

    results = run_all(args.base_url)
    print_summary(results)

    # Save full results (strip internal _generate_response/_judge_response for
    # the top-level view but keep them nested under their keys)
    output_path = args.output
    with open(output_path, "w") as f:
        json.dump(
            {
                "run_at": datetime.now(timezone.utc).isoformat(),
                "base_url": args.base_url,
                "model_id": MODEL_ID,
                "prompt_strategy": PROMPT_STRATEGY,
                "hyperparams": HYPERPARAMS,
                "results": results,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"  Full results written to: {output_path}\n")


if __name__ == "__main__":
    main()