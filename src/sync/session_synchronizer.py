"""
Session Synchronizer — Phase 3
Maps each behavioral session to its corresponding EEG recording window.

Algorithm:
  1. Compute activity_start_unix = started_at_unix - time_taken_seconds
  2. Check if this falls within EEG [start_timestamp, stop_timestamp]
  3. Compute EEG-relative offset (seconds into recording)
  4. Classify match confidence
  5. Produce sync_manifest.csv for downstream epoching
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
INVENTORY_CSV = RESULTS_DIR / "file_inventory.csv"
LINKAGE_CSV = RESULTS_DIR / "participant_linkage.csv"
SESSIONS_CSV = RESULTS_DIR / "sessions_enriched.csv"
OUT_MANIFEST = RESULTS_DIR / "sync_manifest.csv"
OUT_REPORT = RESULTS_DIR / "sync_report.md"


def load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def get_eeg_metadata(inventory: list[dict]) -> dict[str, dict]:
    """
    Returns dict: eeg_filepath -> {start_unix, stop_unix, duration_s, sampling_rate, filepath}
    """
    meta = {}
    for row in inventory:
        ft = row.get("file_type", "")
        role = row.get("inferred_role", "")
        path = row.get("filepath", row.get("path", ""))
        if not (ft == "EMOTIV_EEG_CSV" or role == "EEG_PRIMARY"):
            continue
        try:
            start = float(row.get("start_timestamp_unix", 0) or 0)
            stop = float(row.get("stop_timestamp_unix", 0) or 0)
        except (ValueError, TypeError):
            start, stop = 0, 0
        if start <= 0:
            continue
        meta[path] = {
            "filepath": path,
            "eeg_title": row.get("title", ""),
            "start_unix": start,
            "stop_unix": stop,
            "duration_s": round(stop - start, 2) if stop > start else None,
            "sampling_rate": row.get("eeg_sampling_rate_hz", "128"),
        }
    return meta


def build_uuid_to_eeg(linkage: list[dict], eeg_meta: dict[str, dict]) -> dict[str, list[dict]]:
    """
    Returns: participant_uuid -> list of EEG metadata dicts (matched with HIGH or MODERATE confidence)
    """
    mapping: dict[str, list[dict]] = {}
    for row in linkage:
        uuid = row.get("matched_uuid", "")
        confidence = row.get("match_confidence", "")
        filepath = row.get("eeg_filepath", "")
        if not uuid or confidence not in ("HIGH", "MODERATE"):
            continue
        eeg = eeg_meta.get(filepath)
        if eeg is None:
            # Try to find by partial path match
            for k, v in eeg_meta.items():
                if Path(k).name == Path(filepath).name:
                    eeg = v
                    break
        if eeg is None:
            continue
        mapping.setdefault(uuid, []).append({**eeg, "link_confidence": confidence})
    return mapping


def sync_session(session: dict, eeg_files: list[dict]) -> dict:
    """
    Attempt to find the EEG recording window for a single behavioral session.
    Returns a sync record.
    """
    session_id = session.get("session_id", "")
    participant_id = session.get("participant_id", "")
    activity_type = session.get("activity_type", "")
    time_taken_s = float(session.get("time_taken_seconds", 0) or 0)
    activity_start_str = session.get("activity_start_unix_est", "")
    activity_end_str = session.get("activity_end_unix_est", "")

    try:
        activity_start = float(activity_start_str) if activity_start_str else None
        activity_end = float(activity_end_str) if activity_end_str else None
    except ValueError:
        activity_start = activity_end = None

    base = {
        "session_id": session_id,
        "participant_id": participant_id,
        "activity_type": activity_type,
        "time_taken_s": time_taken_s,
        "activity_start_unix_est": activity_start or "",
        "activity_end_unix_est": activity_end or "",
        "matched_eeg_file": "",
        "eeg_start_unix": "",
        "eeg_stop_unix": "",
        "eeg_duration_s": "",
        "eeg_relative_start_s": "",   # seconds from EEG recording start to activity start
        "eeg_relative_end_s": "",     # seconds from EEG recording start to activity end
        "sync_status": "NO_EEG_FOR_PARTICIPANT",
        "sync_confidence": "NONE",
        "sync_notes": "",
    }

    if not eeg_files:
        return base

    if activity_start is None:
        base["sync_status"] = "MISSING_BEHAVIORAL_TIMESTAMP"
        return base

    # ── Try each EEG file for this participant ────────────────────────────────
    best_match = None
    best_score = -1

    for eeg in eeg_files:
        eeg_start = eeg.get("start_unix", 0)
        eeg_stop = eeg.get("stop_unix", 0)

        if not (eeg_start and eeg_stop):
            continue

        # Activity window: [activity_start, activity_end]
        act_end = activity_end if activity_end else (activity_start + time_taken_s)

        # Check overlap between activity window and EEG recording
        overlap_start = max(activity_start, eeg_start)
        overlap_end = min(act_end, eeg_stop)
        overlap_s = max(0.0, overlap_end - overlap_start)

        # Coverage fraction of activity window
        coverage = overlap_s / time_taken_s if time_taken_s > 0 else 0.0

        # Score: prefer high overlap
        score = coverage

        if score > best_score:
            best_score = score
            best_match = (eeg, overlap_s, coverage)

    if best_match is None or best_score < 0.1:
        base["sync_status"] = "UNMATCHED"
        base["sync_notes"] = f"No EEG file covers >10% of activity window (best={best_score:.2f})"
        return base

    eeg, overlap_s, coverage = best_match
    eeg_start = eeg["start_unix"]
    eeg_stop = eeg["stop_unix"]
    act_end = activity_end if activity_end else (activity_start + time_taken_s)

    relative_start = activity_start - eeg_start
    relative_end = act_end - eeg_start

    # Confidence classification
    if coverage >= 0.9:
        confidence = "HIGH"
        status = "MATCHED"
    elif coverage >= 0.7:
        confidence = "MODERATE"
        status = "PARTIAL_MATCH"
    elif coverage >= 0.3:
        confidence = "LOW"
        status = "PARTIAL_MATCH"
    else:
        confidence = "VERY_LOW"
        status = "WEAK_MATCH"

    # Additional sanity checks
    notes = []
    notes.append(f"Coverage: {coverage:.1%}")
    notes.append(f"Overlap: {overlap_s:.0f}s")

    if relative_start < 0:
        notes.append("WARN: activity_start before EEG start (clock drift or timestamp error)")
        confidence = "LOW"
        status = "PARTIAL_MATCH"

    if relative_end > (eeg_stop - eeg_start):
        notes.append("WARN: activity_end after EEG stop (recording ended early)")
        confidence = "LOW"

    return {
        **base,
        "matched_eeg_file": eeg["filepath"],
        "eeg_start_unix": round(eeg_start, 3),
        "eeg_stop_unix": round(eeg_stop, 3),
        "eeg_duration_s": eeg.get("duration_s", ""),
        "eeg_relative_start_s": round(relative_start, 3),
        "eeg_relative_end_s": round(relative_end, 3),
        "sync_status": status,
        "sync_confidence": confidence,
        "sync_notes": " | ".join(notes),
    }


def write_sync_report(manifest: list[dict], path: Path):
    from collections import Counter
    status_counts = Counter(r["sync_status"] for r in manifest)
    conf_counts = Counter(r["sync_confidence"] for r in manifest)
    act_counts = Counter(r["activity_type"] for r in manifest if r["sync_status"] == "MATCHED")

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Synchronization Manifest Report\n\n")
        f.write(f"Total behavioral sessions: {len(manifest)}\n\n")

        f.write("## Sync Status\n\n")
        for status, cnt in sorted(status_counts.items()):
            f.write(f"  {status}: {cnt}\n")

        f.write("\n## Sync Confidence\n\n")
        for conf, cnt in sorted(conf_counts.items()):
            f.write(f"  {conf}: {cnt}\n")

        matched = [r for r in manifest if r["sync_status"] == "MATCHED"]
        f.write(f"\n## Matched Sessions: {len(matched)}\n\n")
        f.write("Activity type distribution of matched sessions:\n")
        for act, cnt in sorted(act_counts.items()):
            f.write(f"  {act}: {cnt}\n")

        if matched:
            durations = [float(r["time_taken_s"]) for r in matched if r["time_taken_s"]]
            if durations:
                f.write(f"\nActivity durations (matched sessions):\n")
                f.write(f"  Min: {min(durations):.0f}s\n")
                f.write(f"  Max: {max(durations):.0f}s\n")
                f.write(f"  Mean: {sum(durations)/len(durations):.0f}s\n")

        f.write("\n## DL Usability Assessment\n\n")
        dl_usable = [
            r for r in manifest
            if r["sync_confidence"] in ("HIGH", "MODERATE")
            and float(r.get("time_taken_s", 0) or 0) >= 60
        ]
        f.write(f"Sessions suitable for DL windowing (HIGH/MODERATE confidence, >=60s): {len(dl_usable)}\n")
        f.write("\nNote: Each session can yield approximately:\n")
        f.write("  - session_duration / 4s = number of 4-second non-overlapping windows\n")
        total_windows_est = sum(
            int(float(r["time_taken_s"] or 0) // 4) for r in dl_usable if r["time_taken_s"]
        )
        f.write(f"  - Estimated total 4s windows from usable sessions: {total_windows_est}\n")

    print(f"  [OK] Sync report written: {path}")


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 3: SESSION-BOUNDARY SYNCHRONIZATION")
    print("=" * 60)

    for path, name in [(INVENTORY_CSV, "file_inventory.csv"),
                       (LINKAGE_CSV, "participant_linkage.csv"),
                       (SESSIONS_CSV, "sessions_enriched.csv")]:
        if not path.exists():
            print(f"ERROR: {name} not found at {path}")
            print(f"  Run the previous phases first.")
            sys.exit(1)

    inventory = load_csv(INVENTORY_CSV)
    linkage = load_csv(LINKAGE_CSV)
    sessions = load_csv(SESSIONS_CSV)

    print(f"  Inventory entries: {len(inventory)}")
    print(f"  Linkage records: {len(linkage)}")
    print(f"  Behavioral sessions: {len(sessions)}")

    eeg_meta = get_eeg_metadata(inventory)
    print(f"  EEG files with timestamps: {len(eeg_meta)}")

    uuid_to_eeg = build_uuid_to_eeg(linkage, eeg_meta)
    print(f"  Participants with linked EEG: {len(uuid_to_eeg)}")

    print("\n  Synchronizing sessions...")
    manifest = []
    for sess in sessions:
        pid = sess.get("participant_id", "")
        eeg_files = uuid_to_eeg.get(pid, [])
        record = sync_session(sess, eeg_files)
        manifest.append(record)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Write manifest CSV
    if manifest:
        fields = list(manifest[0].keys())
        with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(manifest)
        print(f"  [OK] sync_manifest.csv written: {OUT_MANIFEST}")

    write_sync_report(manifest, OUT_REPORT)

    # Console summary
    from collections import Counter
    status_counts = Counter(r["sync_status"] for r in manifest)
    print("\n  Sync Results:")
    for status, cnt in sorted(status_counts.items()):
        print(f"    {status}: {cnt}")

    dl_usable = [
        r for r in manifest
        if r["sync_confidence"] in ("HIGH", "MODERATE")
        and float(r.get("time_taken_s", 0) or 0) >= 60
    ]
    total_windows_est = sum(
        int(float(r["time_taken_s"] or 0) // 4) for r in dl_usable if r["time_taken_s"]
    )
    print(f"\n  DL-usable sessions (>=60s, HIGH/MODERATE sync): {len(dl_usable)}")
    print(f"  Estimated 4-second EEG windows available: {total_windows_est}")
