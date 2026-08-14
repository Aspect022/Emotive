"""
Trial Expander — Phase 2B
Flattens the nested JSON `answers` column in activity_results_rows.csv into
a per-trial flat table. Adds ground-truth columns from the fixed question bank
and computes timing estimates.

CRITICAL: `started_at` = activity END time (not start).
  true_activity_start = started_at_unix - time_taken_seconds
"""

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "Data"
OUT_DIR = ROOT / "results"

# ── Fixed question bank (from lib/questions.ts) ──────────────────────────────
# Each entry: (id, difficulty, category, timeLimit, stimulusTime_or_None)
# stimulusTime: seconds the stimulus card is shown before answer phase starts
QUESTION_BANK = {
    "arithmetic": [
        (1, "easy",   "Addition",             15, None),
        (2, "easy",   "Division",             15, None),
        (3, "easy",   "Addition",             15, None),
        (4, "easy",   "Subtraction",          15, None),
        (5, "medium", "Addition",             20, None),
        (6, "medium", "Multiplication",       20, None),
        (7, "medium", "Division",             20, None),
        (8, "hard",   "Percentage",           30, None),
        (9, "hard",   "Multiplication",       30, None),
        (10,"hard",   "Multi-step",           30, None),
    ],
    "pattern": [
        (1, "easy",   "Arithmetic Sequence",  15, None),
        (2, "easy",   "Arithmetic Sequence",  15, None),
        (3, "easy",   "Odd Numbers",          15, None),
        (4, "easy",   "Decreasing Sequence",  15, None),
        (5, "medium", "Perfect Squares",      20, None),
        (6, "medium", "Fibonacci",            20, None),
        (7, "medium", "Perfect Cubes",        20, None),
        (8, "hard",   "Triangular Differences",30, None),
        (9, "hard",   "Squares",              30, None),
        (10,"hard",   "n(n+1)",               30, None),
    ],
    "memory": [
        (1, "easy",   "Digit Recall",         10, 5),
        (2, "easy",   "Word Recall",          10, 5),
        (3, "easy",   "Digit Recall",         10, 5),
        (4, "easy",   "Word Recall",          10, 5),
        (5, "medium", "Digit Recall",         12, 6),
        (6, "medium", "Position Recall",      12, 6),
        (7, "medium", "Position Recall",      12, 6),
        (8, "hard",   "Sequence Recall",      15, 7),
        (9, "hard",   "Position Recall",      15, 7),
        (10,"hard",   "Digit Analysis",       15, 7),
    ],
    "comprehension": [
        (1, "easy",   "Fact Extraction",      25, None),
        (2, "medium", "Detail Recall",        25, None),
        (3, "easy",   "Fact Extraction",      25, None),
        (4, "medium", "Inference",            25, None),
        (5, "easy",   "Detail Recall",        25, None),
        (6, "easy",   "Fact Extraction",      25, None),
        (7, "easy",   "Detail Recall",        25, None),
        (8, "medium", "Detail Recall",        25, None),
        (9, "medium", "Inference",            25, None),
        (10,"easy",   "Fact Extraction",      25, None),
    ],
    "attention": [
        (1, "easy",   "Oddball Detection",    8,  5),
        (2, "easy",   "Count Target",         8,  5),
        (3, "easy",   "Pattern Break",        8,  5),
        (4, "easy",   "Position Detect",      8,  5),
        (5, "medium", "Rule Violation",       10, 6),
        (6, "medium", "Count Target",         10, 6),
        (7, "medium", "Count Target",         10, 6),
        (8, "hard",   "Count Target",         12, 7),
        (9, "hard",   "Rule Violation",       12, 7),
        (10,"hard",   "Pattern Break",        12, 7),
    ],
}

# Comprehension passages use a reading phase (unmetered time before first question)
# and between passage 1 and passage 2 groups. We cannot estimate reading time.
COMPREHENSION_PASSAGE_BREAK_BEFORE = {1, 6}  # question indices (1-based) that start a new passage


def parse_supabase_dt(s: str) -> float | None:
    """Parse Supabase ISO timestamp to UTC Unix float."""
    if not s:
        return None
    try:
        s = s.strip()
        # "2026-04-21 11:36:15.123456+00" → "2026-04-21T11:36:15+00:00"
        s = s.replace(" ", "T")
        # strip sub-second
        if "." in s:
            base, rest = s.split(".", 1)
            # keep timezone from rest
            if "+" in rest:
                tz = "+" + rest.split("+", 1)[1]
            elif rest.endswith("Z"):
                tz = "Z"
            else:
                tz = ""
            s = base + tz
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        if not ("+" in s[10:] or "-" in s[10:]):
            s += "+00:00"
        dt = datetime.fromisoformat(s)
        return dt.timestamp()
    except Exception:
        return None


def load_activity_results(data_dir: Path) -> list[dict]:
    """Find and load activity_results_rows.csv from anywhere under data_dir."""
    matches = list(data_dir.rglob("activity_results_rows.csv"))
    if not matches:
        print("  ERROR: activity_results_rows.csv not found.", file=sys.stderr)
        sys.exit(1)
    path = matches[0]
    print(f"  Loading behavioral data: {path}")
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def expand_session(session: dict) -> list[dict]:
    """
    Given one session row from activity_results, return a list of trial rows.
    """
    session_id = session.get("id", "")
    participant_id = session.get("participant_id", "")
    activity_type = session.get("activity_type", "").lower()
    total_q = int(session.get("total_questions", 10) or 10)
    score = int(session.get("score", 0) or 0)
    time_taken_s = int(session.get("time_taken_seconds", 0) or 0)

    # ── Parse timestamps ──────────────────────────────────────────────────────
    started_at_raw = session.get("started_at", "")
    completed_at_raw = session.get("completed_at", "")
    # started_at = activity END (logged at completion per source code)
    started_at_unix = parse_supabase_dt(started_at_raw)
    completed_at_unix = parse_supabase_dt(completed_at_raw)

    # True activity start
    activity_start_unix = (started_at_unix - time_taken_s) if started_at_unix and time_taken_s else None

    # ── Parse answers JSON ────────────────────────────────────────────────────
    answers_raw = session.get("answers", "") or "[]"
    try:
        answers = json.loads(answers_raw)
    except json.JSONDecodeError:
        answers = []

    if not isinstance(answers, list):
        answers = []

    # ── Ground-truth question bank ────────────────────────────────────────────
    bank = QUESTION_BANK.get(activity_type, [])

    trials = []
    cumulative_active_s = 0.0  # cumulative active time from start of session

    for i, ans in enumerate(answers):
        q_idx = i  # 0-based
        q_num = i + 1  # 1-based

        # From question bank (if available)
        if q_idx < len(bank):
            q_id, difficulty, category, time_limit_s, stimulus_time_s = bank[q_idx]
        else:
            q_id, difficulty, category, time_limit_s, stimulus_time_s = (
                q_num, "unknown", "unknown", None, None
            )

        # From behavioral record
        selected = ans.get("selected", None)
        correct_idx = ans.get("correct", None)
        time_taken_q = ans.get("timeTaken", None)
        timed_out = bool(ans.get("timedOut", False))

        # Is it correct?
        if selected is not None and correct_idx is not None:
            is_correct = int(selected) == int(correct_idx)
        else:
            is_correct = False  # timed out = incorrect

        # ── Per-question timing estimates ─────────────────────────────────────
        # For stimulus tasks: stimulus phase comes before answer phase
        # timeTaken only covers the ANSWER phase
        if stimulus_time_s is not None:
            question_total_min_s = (stimulus_time_s or 0) + (time_taken_q or 0)
        else:
            question_total_min_s = time_taken_q or 0

        # Comprehension: question 1 and 6 have a preceding reading phase (unmeasured)
        has_reading_phase = (activity_type == "comprehension" and q_num in COMPREHENSION_PASSAGE_BREAK_BEFORE)

        # Estimated EEG-relative onset (seconds from activity_start_unix)
        # This is a lower bound — does NOT include inter-question gaps or reading time
        q_onset_offset_s = cumulative_active_s
        q_answer_onset_offset_s = (
            q_onset_offset_s + (stimulus_time_s or 0)
        ) if stimulus_time_s else q_onset_offset_s

        # Estimated absolute timestamps
        if activity_start_unix:
            q_onset_est_unix = activity_start_unix + q_onset_offset_s
            q_answer_onset_est_unix = activity_start_unix + q_answer_onset_offset_s
        else:
            q_onset_est_unix = None
            q_answer_onset_est_unix = None

        # Advance cumulative for next question
        cumulative_active_s += question_total_min_s

        trials.append({
            # Session identifiers
            "session_id": session_id,
            "participant_id": participant_id,
            "activity_type": activity_type,
            "session_score": score,
            "session_total_q": total_q,
            "session_accuracy": round(score / total_q, 3) if total_q else None,
            "session_time_taken_s": time_taken_s,
            # Performance labels (for DL)
            "session_high_performer": int(score >= 7),   # binary: >=70%
            "session_score_bin3": (
                "high" if score >= 8 else "mid" if score >= 5 else "low"
            ),  # 3-class performance

            # Timing (ground truth)
            "activity_start_unix_est": round(activity_start_unix, 3) if activity_start_unix else "",
            "activity_end_unix_est": round(started_at_unix, 3) if started_at_unix else "",
            "started_at_raw": started_at_raw,
            "completed_at_raw": completed_at_raw,

            # Trial identifiers
            "trial_index": q_idx,   # 0-based within session
            "question_number": q_num,  # 1-based
            "question_id": q_id,

            # Ground-truth question properties (from fixed bank)
            "difficulty": difficulty,
            "category": category,
            "time_limit_s": time_limit_s,
            "stimulus_time_s": stimulus_time_s or "",
            "has_reading_phase": has_reading_phase,

            # Participant responses
            "selected_option": selected,
            "correct_option": correct_idx,
            "is_correct": int(is_correct),
            "timed_out": int(timed_out),
            "time_taken_q_s": time_taken_q,   # integer seconds, answer phase only

            # Timing estimates (for EEG alignment — approximate)
            "q_onset_offset_s_est": round(q_onset_offset_s, 1),
            "q_answer_onset_offset_s_est": round(q_answer_onset_offset_s, 1),
            "q_onset_unix_est": round(q_onset_est_unix, 3) if q_onset_est_unix else "",
            "q_answer_onset_unix_est": round(q_answer_onset_est_unix, 3) if q_answer_onset_est_unix else "",
            "timing_note": (
                "ESTIMATE_ONLY_inter_question_gaps_unknown"
                if not has_reading_phase else
                "ESTIMATE_ONLY_reading_phase_duration_unknown"
            ),
        })

    return trials


def compute_session_enriched(sessions: list[dict]) -> list[dict]:
    """
    Returns one row per session with corrected timestamps and performance labels.
    """
    enriched = []
    for s in sessions:
        time_taken_s = int(s.get("time_taken_seconds", 0) or 0)
        started_at_unix = parse_supabase_dt(s.get("started_at", ""))
        activity_start_unix = (
            (started_at_unix - time_taken_s) if started_at_unix and time_taken_s else None
        )
        total_q = int(s.get("total_questions", 10) or 10)
        score = int(s.get("score", 0) or 0)
        accuracy = round(score / total_q, 3) if total_q else None

        answers_raw = s.get("answers", "") or "[]"
        try:
            answers = json.loads(answers_raw)
        except Exception:
            answers = []

        enriched.append({
            "session_id": s.get("id", ""),
            "participant_id": s.get("participant_id", ""),
            "activity_type": s.get("activity_type", "").lower(),
            "score": score,
            "total_questions": total_q,
            "accuracy": accuracy,
            "high_performer": int(score >= 7),
            "score_bin3": "high" if score >= 8 else "mid" if score >= 5 else "low",
            "time_taken_seconds": time_taken_s,
            "n_trials_parsed": len(answers) if isinstance(answers, list) else 0,
            "n_timed_out": sum(1 for a in answers if isinstance(a, dict) and a.get("timedOut", False)),
            "started_at_raw": s.get("started_at", ""),
            "started_at_note": "THIS_IS_ACTIVITY_END_TIME",
            "activity_start_unix_est": round(activity_start_unix, 3) if activity_start_unix else "",
            "activity_end_unix_est": round(started_at_unix, 3) if started_at_unix else "",
            "completed_at_raw": s.get("completed_at", ""),
        })
    return enriched


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2B: BEHAVIORAL TRIAL EXPANSION")
    print("=" * 60)

    sessions = load_activity_results(DATA_DIR)
    print(f"  Sessions loaded: {len(sessions)}")

    # Expand to trials
    all_trials = []
    for sess in sessions:
        trials = expand_session(sess)
        all_trials.extend(trials)

    print(f"  Total trials expanded: {len(all_trials)}")

    # Sessions enriched
    enriched = compute_session_enriched(sessions)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Write trials flat
    trials_path = OUT_DIR / "trials_flat.csv"
    if all_trials:
        fields = list(all_trials[0].keys())
        with open(trials_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_trials)
        print(f"  [OK] trials_flat.csv written: {trials_path}")

    # Write sessions enriched
    sessions_path = OUT_DIR / "sessions_enriched.csv"
    if enriched:
        fields = list(enriched[0].keys())
        with open(sessions_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(enriched)
        print(f"  [OK] sessions_enriched.csv written: {sessions_path}")

    # Validation: sum(timeTaken) vs time_taken_seconds
    print("\n  Validation (sum of per-question timeTaken vs session time_taken_seconds):")
    mismatches = 0
    for sess in sessions:
        answers_raw = sess.get("answers", "") or "[]"
        try:
            answers = json.loads(answers_raw)
        except Exception:
            answers = []
        if not isinstance(answers, list):
            continue
        q_sum = sum(int(a.get("timeTaken", 0) or 0) for a in answers if isinstance(a, dict))
        session_s = int(sess.get("time_taken_seconds", 0) or 0)
        delta = abs(q_sum - session_s)
        if delta > 30:  # more than 30s discrepancy
            mismatches += 1

    total_valid = sum(
        1 for s in sessions
        if json.loads(s.get("answers", "[]") or "[]") and isinstance(json.loads(s.get("answers", "[]") or "[]"), list)
    )
    print(f"    Sessions with answers: {total_valid}")
    print(f"    Sessions with >30s timing mismatch: {mismatches}")
    print(f"    (Mismatches expected due to inter-question gaps and reading phases)")

    # Breakdown by activity type
    from collections import Counter
    activity_counts = Counter(t["activity_type"] for t in all_trials)
    print("\n  Trials by activity type:")
    for act, cnt in sorted(activity_counts.items()):
        print(f"    {act}: {cnt} trials")

    # Label balance
    perf_counts = Counter(t["session_high_performer"] for t in all_trials)
    print(f"\n  Performance label balance (high=1, low=0):")
    for label, cnt in sorted(perf_counts.items()):
        print(f"    {label}: {cnt} ({100*cnt/len(all_trials):.1f}%)")
