"""
Participant Linker — Phase 2A
Matches EEG filenames to Supabase participant UUIDs.

Strategy:
  - Numbered files (participant_N):  match by recording timestamp vs Supabase created_at
  - Named files (FirstName_*):       fuzzy first-name match against full_name
  - Imagination/stress files:        flagged as separate sub-study
"""

import csv
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
INVENTORY_CSV = ROOT / "results" / "file_inventory.csv"
PARTICIPANTS_CSV = next(
    (p for p in (ROOT / "Data").rglob("participants_rows*.csv")),
    None
)
OUT_DIR = ROOT / "results"
OUT_LINKAGE = OUT_DIR / "participant_linkage.csv"
OUT_LINKAGE_REPORT = OUT_DIR / "participant_linkage_report.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_emotiv_timestamp_from_filename(fname: str) -> float | None:
    """
    Extract the Unix timestamp embedded in an EMOTIV filename.
    Format: ..._EPOCX_<serial>_<YYYY>.<MM>.<DD>T<HH>.<MM>.<SS>+<TZ>.md.*
    Returns UTC Unix timestamp float, or None.
    """
    m = re.search(
        r"(\d{4})\.(\d{2})\.(\d{2})T(\d{2})\.(\d{2})\.(\d{2})\+(\d{2})\.(\d{2})",
        fname
    )
    if not m:
        return None
    year, mon, day, hr, mn, sec, tz_h, tz_m = (int(x) for x in m.groups())
    # Build naive local datetime then convert to UTC
    local_offset_sec = tz_h * 3600 + tz_m * 60
    ts_local = datetime(year, mon, day, hr, mn, sec)
    ts_utc = ts_local.timestamp() - local_offset_sec
    return ts_utc


def name_similarity(eeg_prefix: str, full_name: str) -> float:
    """
    Simple first-name similarity score.
    eeg_prefix: e.g. "Nishanth", "Brijesh 1", "Dhrshan G"
    full_name:  e.g. "Nishanth Sharma"
    Returns 0.0–1.0.
    """
    # Normalize both
    eeg_parts = re.sub(r"\s+", " ", eeg_prefix.lower().strip()).split()
    name_parts = re.sub(r"\s+", " ", full_name.lower().strip()).split()

    if not eeg_parts or not name_parts:
        return 0.0

    # Direct first-token match is strongest signal
    if eeg_parts[0] == name_parts[0]:
        return 1.0

    # Levenshtein-like character overlap for typo tolerance
    a, b = eeg_parts[0], name_parts[0]
    matches = sum(ca == cb for ca, cb in zip(a, b))
    score = matches / max(len(a), len(b))
    return score


def load_participants(csv_path: Path) -> list[dict]:
    """Load Supabase participants export."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_inventory(csv_path: Path) -> list[dict]:
    """Load file_inventory.csv produced by Phase 1."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Sub-study classifier
# ---------------------------------------------------------------------------

def classify_session(eeg_title: str, filepath: str) -> str:
    """
    Returns one of: 'main_numbered', 'main_named', 'imagination', 'stress', 'unknown'
    """
    fp_lower = filepath.lower()
    title_lower = eeg_title.lower()

    if "imagination" in fp_lower or "imagination" in title_lower:
        return "imagination"
    if "stress" in fp_lower or "stress" in title_lower:
        return "stress"
    if re.match(r"participant_\d+", title_lower):
        return "main_numbered"
    # Heuristic: if it has a recognizable proper name
    if re.match(r"[a-z][a-z ]+", title_lower) and len(title_lower) > 2:
        return "main_named"
    return "unknown"


# ---------------------------------------------------------------------------
# Main linkage logic
# ---------------------------------------------------------------------------

def link_participants(inventory: list[dict], participants: list[dict]) -> list[dict]:
    """
    Returns list of linkage records with fields:
      eeg_filepath, eeg_title, sub_study, eeg_recording_start_utc,
      matched_uuid, matched_full_name, match_method, match_confidence,
      match_score, notes
    """
    results = []

    # Only process EEG data files (not marker/JSON)
    eeg_files = [
        r for r in inventory
        if r.get("file_type", "") == "EMOTIV_EEG_CSV"
        or r.get("inferred_role", "") == "EEG_PRIMARY"
    ]

    # Parse participant timestamps for timestamp-matching
    # created_at is ISO UTC from Supabase
    for p in participants:
        raw = p.get("created_at", "")
        try:
            # Handle formats like "2026-04-21 11:36:15.123456+00"
            raw_clean = raw.replace(" ", "T").split(".")[0]
            if "+" in raw_clean:
                raw_clean = raw_clean.split("+")[0] + "+00:00"
            dt = datetime.fromisoformat(raw_clean.replace("+00:00", "")).replace(
                tzinfo=timezone.utc
            )
            p["_created_at_unix"] = dt.timestamp()
        except Exception:
            p["_created_at_unix"] = None

    for row in eeg_files:
        filepath = row.get("filepath", row.get("path", ""))
        fname = Path(filepath).name
        eeg_title = row.get("title", fname.split("_")[0])
        if not eeg_title:
            # Extract from filename: everything before first "_EPOCX_"
            m = re.match(r"^(.+?)_EPOCX_", fname)
            eeg_title = m.group(1) if m else fname.split("_")[0]
        sub_study = classify_session(eeg_title, filepath)

        # Get EEG recording start from filename
        eeg_start_utc = parse_emotiv_timestamp_from_filename(fname)
        if not eeg_start_utc:
            ts_raw = row.get("start_timestamp_unix", "")
            if ts_raw:
                try:
                    eeg_start_utc = float(ts_raw)
                except ValueError:
                    pass

        record = {
            "eeg_filepath": filepath,
            "eeg_title": eeg_title,
            "sub_study": sub_study,
            "eeg_recording_start_utc": eeg_start_utc,
            "matched_uuid": None,
            "matched_full_name": None,
            "match_method": None,
            "match_confidence": "UNMATCHED",
            "match_score": 0.0,
            "notes": "",
        }

        # ── Strategy 1: Numbered participant ──
        m = re.match(r"participant_(\d+)", eeg_title.lower())
        if m and sub_study == "main_numbered":
            num = int(m.group(1))
            # Look for participant whose created_at is closest to EEG start
            best_uuid, best_name, best_delta = None, None, math.inf
            for p in participants:
                if p["_created_at_unix"] is None:
                    continue
                delta = abs((p["_created_at_unix"] - eeg_start_utc)) if eeg_start_utc else math.inf
                if delta < best_delta:
                    best_delta = delta
                    best_uuid = p["id"]
                    best_name = p["full_name"]

            # Also try to match by name convention: if participant has name like "Participant 5"
            for p in participants:
                fn = p.get("full_name", "").lower()
                if f"participant {num}" in fn or f"participant{num}" in fn:
                    best_uuid = p["id"]
                    best_name = p["full_name"]
                    best_delta = 0
                    break

            if best_uuid and (best_delta < 3600 or best_delta == 0):  # within 1 hour
                record["matched_uuid"] = best_uuid
                record["matched_full_name"] = best_name
                record["match_method"] = "timestamp_proximity"
                record["match_confidence"] = "HIGH" if best_delta < 300 else "MODERATE"
                record["match_score"] = max(0.0, 1.0 - best_delta / 3600)
                record["notes"] = f"Timestamp delta: {best_delta:.0f}s"
            else:
                record["notes"] = f"No participant within 1 hour of EEG start (best delta: {best_delta:.0f}s)"

        # ── Strategy 2: Named participant ──
        elif sub_study == "main_named":
            best_uuid, best_name, best_score = None, None, 0.0
            for p in participants:
                score = name_similarity(eeg_title, p.get("full_name", ""))
                if score > best_score:
                    best_score = score
                    best_uuid = p["id"]
                    best_name = p["full_name"]

            if best_score >= 1.0:
                confidence = "HIGH"
            elif best_score >= 0.8:
                confidence = "MODERATE"
            elif best_score >= 0.6:
                confidence = "LOW"
            else:
                confidence = "UNMATCHED"
                best_uuid = None

            if best_uuid:
                record["matched_uuid"] = best_uuid
                record["matched_full_name"] = best_name
                record["match_method"] = "name_similarity"
                record["match_confidence"] = confidence
                record["match_score"] = round(best_score, 3)
                record["notes"] = f"Name score: {best_score:.3f}"

        elif sub_study in ("imagination", "stress"):
            record["match_confidence"] = "SEPARATE_SUB_STUDY"
            record["notes"] = f"Sub-study: {sub_study} — excluded from main analysis"

        results.append(record)

    return results


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def write_linkage_csv(records: list[dict], path: Path):
    fields = [
        "eeg_filepath", "eeg_title", "sub_study", "eeg_recording_start_utc",
        "matched_uuid", "matched_full_name", "match_method",
        "match_confidence", "match_score", "notes"
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"  [OK] Linkage written: {path}")


def write_linkage_report(records: list[dict], path: Path):
    conf_counts = {}
    for r in records:
        c = r["match_confidence"]
        conf_counts[c] = conf_counts.get(c, 0) + 1

    main_records = [r for r in records if r["sub_study"] in ("main_numbered", "main_named")]
    matched = [r for r in main_records if r["matched_uuid"]]
    unmatched = [r for r in main_records if not r["matched_uuid"]]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Participant Linkage Report\n\n")
        f.write(f"Total EEG sessions processed: {len(records)}\n")
        f.write(f"Main study (numbered + named): {len(main_records)}\n")
        f.write(f"  - Matched to Supabase UUID: {len(matched)}\n")
        f.write(f"  - Unmatched: {len(unmatched)}\n\n")
        f.write("## Confidence Distribution\n\n")
        for conf, cnt in sorted(conf_counts.items()):
            f.write(f"  {conf}: {cnt}\n")

        f.write("\n## Unmatched Sessions\n\n")
        for r in unmatched:
            f.write(f"  - {r['eeg_title']} | {Path(r['eeg_filepath']).name}\n")
            f.write(f"    Notes: {r['notes']}\n")

        f.write("\n## Matched Sessions (main study)\n\n")
        f.write("| EEG Title | Participant Name | UUID (short) | Method | Confidence |\n")
        f.write("|---|---|---|---|---|\n")
        for r in sorted(matched, key=lambda x: x["eeg_title"]):
            uuid_short = (r["matched_uuid"] or "")[:8]
            f.write(
                f"| {r['eeg_title']} | {r['matched_full_name']} | {uuid_short}... "
                f"| {r['match_method']} | {r['match_confidence']} |\n"
            )

    print(f"  [OK] Linkage report written: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2A: PARTICIPANT LINKAGE")
    print("=" * 60)

    if not INVENTORY_CSV.exists():
        print(f"ERROR: file_inventory.csv not found at {INVENTORY_CSV}")
        print("  Run run_phase1_inspection.py first.")
        sys.exit(1)

    if not PARTICIPANTS_CSV:
        print("ERROR: participants_rows*.csv not found in Data/ tree.")
        sys.exit(1)

    print(f"\n  Loading inventory: {INVENTORY_CSV}")
    inventory = load_inventory(INVENTORY_CSV)
    print(f"  Inventory rows: {len(inventory)}")

    print(f"  Loading participants: {PARTICIPANTS_CSV}")
    participants = load_participants(PARTICIPANTS_CSV)
    print(f"  Participants in Supabase: {len(participants)}")

    print("\n  Running linkage...")
    records = link_participants(inventory, participants)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_linkage_csv(records, OUT_LINKAGE)
    write_linkage_report(records, OUT_LINKAGE_REPORT)

    # Summary
    main = [r for r in records if r["sub_study"] in ("main_numbered", "main_named")]
    matched = [r for r in main if r["matched_uuid"]]
    high = [r for r in matched if r["match_confidence"] == "HIGH"]
    moderate = [r for r in matched if r["match_confidence"] == "MODERATE"]

    print(f"\n  Main study sessions:  {len(main)}")
    print(f"  Matched HIGH:         {len(high)}")
    print(f"  Matched MODERATE:     {len(moderate)}")
    print(f"  Unmatched:            {len(main) - len(matched)}")
    print(f"  Sub-study sessions:   {len(records) - len(main)}")
