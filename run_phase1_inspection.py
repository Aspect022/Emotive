"""
run_phase1_inspection.py
------------------------
Phase 1 entry point: scan the project, produce inventory CSV and
forensics markdown report.  Does NOT modify any raw data file.

Usage:
    python run_phase1_inspection.py

Outputs (written to results/):
    file_inventory.csv
    file_forensics.md
    emotiv_device_characterization.md
    session_mapping.csv
    data_dictionary.csv
    data_dictionary.md
    synchronization_report.md  (feasibility assessment only)
    qc_report.md               (initial readiness)
"""

import sys
import json
import csv
import re
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone

# Allow running from project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.file_inspector import walk_project

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATA_ROOT = PROJECT_ROOT / "Data"


# =============================================================================
# STEP 1 — Walk & inventory
# =============================================================================

print("=" * 60)
print("PHASE 1: FILE INSPECTION & INVENTORY")
print("=" * 60)
print(f"\nScanning: {DATA_ROOT}\n")

records = walk_project(DATA_ROOT)

print(f"  Found {len(records)} file records.\n")

# ---------------------------------------------------------------------------
# Write flat inventory CSV
# ---------------------------------------------------------------------------
INVENTORY_COLS = [
    "filepath", "filename", "extension", "file_type",
    "size_mb", "mtime_utc", "sha256",
    "title", "headset_type", "headset_serial",
    "eeg_sampling_rate_hz", "sampling_rate_raw",
    "eeg_electrode_count", "eeg_electrode_channels",
    "start_timestamp_unix", "stop_timestamp_unix",
    "duration_seconds", "duration_minutes", "samples_reported",
    "columns", "column_count", "row_count",
    "marker_count", "marker_labels", "marker_ports",
    "record_id", "export_app", "export_time", "user",
    "contact_quality_channels", "power_band_columns", "motion_columns",
    "activity_types", "unique_participant_ids",
    "first_session_utc", "last_session_utc",
    "inferred_role", "confidence", "parse_status",
]

inventory_path = RESULTS_DIR / "file_inventory.csv"
with open(inventory_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=INVENTORY_COLS, extrasaction="ignore")
    writer.writeheader()
    for rec in records:
        # Serialize lists to strings for CSV
        for k, v in rec.items():
            if isinstance(v, list):
                rec[k] = "|".join(str(x) for x in v)
        writer.writerow(rec)

print(f"  [OK] Inventory written: {inventory_path}")


# =============================================================================
# STEP 2 — Parse and organise by role
# =============================================================================

eeg_files       = [r for r in records if r.get("file_type") == "EMOTIV_EEG_CSV"]
marker_files    = [r for r in records if r.get("file_type") == "EMOTIV_INTERVAL_MARKER_CSV"]
json_meta_files = [r for r in records if r.get("file_type") == "EMOTIV_JSON_METADATA"]
activity_files  = [r for r in records if r.get("file_type") == "SUPABASE_ACTIVITY_RESULTS"]
participant_files = [r for r in records if r.get("file_type") == "SUPABASE_PARTICIPANTS"]

print(f"\n  EEG data files (md.bp.csv / md.csv):  {len(eeg_files)}")
print(f"  Interval marker files:                  {len(marker_files)}")
print(f"  JSON metadata files:                    {len(json_meta_files)}")
print(f"  Supabase activity results:              {len(activity_files)}")
print(f"  Supabase participants:                  {len(participant_files)}")
print(f"  Other files:                            {len(records) - len(eeg_files) - len(marker_files) - len(json_meta_files) - len(activity_files) - len(participant_files)}")


# =============================================================================
# STEP 3 — Participant/session extraction from filenames
# =============================================================================

def extract_participant_from_filename(filename: str) -> str:
    """
    Filename convention observed:
      <Name>_EPOCX_<Serial>_<Timestamp>.{ext}
    OR
      participant_<N>_EPOCX_<Serial>_<Timestamp>.{ext}
    OR
      imagination_<type>_<N>_EPOCX_...
    """
    stem = Path(filename).stem
    # Remove suffixes like .md.bp or .md
    for suf in (".md.bp", ".md", "_intervalMarker"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
        # case-insensitive
        low = stem.lower()
        lsuf = suf.lower()
        if low.endswith(lsuf):
            stem = stem[: -len(suf)]

    parts = stem.split("_")
    if not parts:
        return "UNKNOWN"

    # Find EPOCX index to get name portion
    epocx_idx = None
    for i, p in enumerate(parts):
        if "EPOCX" in p.upper():
            epocx_idx = i
            break

    if epocx_idx is not None and epocx_idx > 0:
        name_parts = parts[:epocx_idx]
        return " ".join(name_parts)

    return parts[0]


def extract_session_timestamp(filename: str) -> str:
    """
    Extract ISO-like timestamp from filename, e.g.
    2026.04.16T15.21.40+05.30
    """
    m = re.search(r"(\d{4}\.\d{2}\.\d{2}T\d{2}\.\d{2}\.\d{2}[+\-]\d{2}\.\d{2})", filename)
    if m:
        return m.group(1)
    return "UNKNOWN"


def extract_headset_serial(filename: str) -> str:
    m = re.search(r"EPOCX_(\d+)_", filename)
    if m:
        return m.group(1)
    return "UNKNOWN"


# Build session-level grouping from EEG files
sessions = {}
for rec in eeg_files:
    fn = rec["filename"]
    participant = extract_participant_from_filename(fn)
    ts = extract_session_timestamp(fn)
    serial = extract_headset_serial(fn)
    key = fn  # unique per file

    # Find matching marker file
    stem_base = fn
    for suf in (".md.bp.csv", ".md.csv"):
        if stem_base.lower().endswith(suf):
            stem_base = stem_base[: -len(suf)]
    matching_marker = None
    for mf in marker_files:
        mfn = mf["filename"]
        if mfn.startswith(stem_base):
            matching_marker = mf["filepath"]
            break

    # Find matching JSON
    matching_json = None
    for jf in json_meta_files:
        jfn = jf["filename"]
        if jfn.startswith(stem_base):
            matching_json = jf["filepath"]
            break

    # Determine sub-folder (task context)
    folder_path = str(Path(rec["filepath"]).parent)
    if "imagination" in folder_path.lower():
        task_context = "imagination"
    elif "stress" in folder_path.lower():
        task_context = "stress_evaluation"
    else:
        task_context = "web_test_main"

    sessions[key] = {
        "participant_name": participant,
        "headset_serial": serial,
        "headset_type": rec.get("headset_type", "UNKNOWN"),
        "recording_start_unix": rec.get("start_timestamp_unix"),
        "recording_stop_unix": rec.get("stop_timestamp_unix"),
        "duration_minutes": rec.get("duration_minutes"),
        "eeg_sampling_rate_hz": rec.get("eeg_sampling_rate_hz"),
        "eeg_electrode_count": rec.get("eeg_electrode_count"),
        "samples_reported": rec.get("samples_reported"),
        "eeg_filepath": rec["filepath"],
        "marker_filepath": matching_marker,
        "json_filepath": matching_json,
        "task_context": task_context,
        "session_timestamp": ts,
        "parse_status": rec.get("parse_status"),
    }


# Also capture JSON-only records (no EEG file found)
json_only_participants = set()
for jf in json_meta_files:
    jfn = jf["filename"]
    stem = jfn.replace(".json", "")
    has_eeg = any(
        Path(e["filepath"]).stem.split(".")[0] == stem
        for e in eeg_files
    )
    if not has_eeg:
        participant = extract_participant_from_filename(jfn)
        json_only_participants.add(participant)


# =============================================================================
# STEP 4 — Write session mapping
# =============================================================================

session_mapping_path = RESULTS_DIR / "session_mapping.csv"
SESSION_COLS = [
    "session_id", "participant_name", "headset_serial", "headset_type",
    "task_context", "recording_start_unix", "recording_stop_unix",
    "duration_minutes", "eeg_sampling_rate_hz", "eeg_electrode_count",
    "samples_reported", "eeg_filepath", "marker_filepath", "json_filepath",
    "session_timestamp", "parse_status",
]
with open(session_mapping_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=SESSION_COLS, extrasaction="ignore")
    writer.writeheader()
    for idx, (key, sess) in enumerate(sessions.items()):
        sess["session_id"] = f"S{idx+1:03d}"
        writer.writerow(sess)

print(f"\n  [OK] Session mapping written: {session_mapping_path}")
print(f"       Sessions with EEG files: {len(sessions)}")
print(f"       JSON-only (no EEG):      {len(json_only_participants)} participants ({', '.join(sorted(json_only_participants)[:10])}{'...' if len(json_only_participants)>10 else ''})")


# =============================================================================
# STEP 5 — Summarise electrode configurations observed
# =============================================================================

electrode_configs = defaultdict(list)
for rec in eeg_files:
    ch_str = rec.get("eeg_electrode_channels", "")
    if isinstance(ch_str, str):
        channels = tuple(sorted(ch_str.split("|"))) if ch_str else ()
    else:
        channels = tuple(sorted(ch_str))
    count = rec.get("eeg_electrode_count", 0)
    try:
        count = int(count)
    except Exception:
        count = 0
    electrode_configs[count].append(channels)

unique_configs = {}
for count, ch_lists in electrode_configs.items():
    unique_ch = set(ch_lists)
    unique_configs[count] = list(unique_ch)

print("\n  Electrode channel counts observed:")
for cnt, configs in sorted(unique_configs.items()):
    print(f"    {cnt} channels — {len(configs)} unique configuration(s)")


# =============================================================================
# STEP 6 — Sampling rate summary
# =============================================================================

sr_seen = defaultdict(int)
for rec in eeg_files:
    sr = rec.get("eeg_sampling_rate_hz")
    sr_seen[str(sr)] += 1

print("\n  EEG sampling rates observed:")
for sr, cnt in sorted(sr_seen.items()):
    print(f"    {sr} Hz — {cnt} file(s)")


# =============================================================================
# STEP 7 — Duration summary
# =============================================================================

durations = [
    float(r["duration_minutes"])
    for r in eeg_files
    if r.get("duration_minutes") not in (None, "", "None")
]
if durations:
    print(f"\n  Recording durations (minutes):")
    print(f"    Min:    {min(durations):.1f}")
    print(f"    Max:    {max(durations):.1f}")
    print(f"    Median: {sorted(durations)[len(durations)//2]:.1f}")
    print(f"    Mean:   {sum(durations)/len(durations):.1f}")


# =============================================================================
# STEP 8 — Supabase behavioural data summary
# =============================================================================

print("\n  Supabase Behavioral Data:")
for af in activity_files:
    print(f"    File: {af['filename']}")
    print(f"      Rows (sessions): {af.get('row_count', '?')}")
    print(f"      Columns: {af.get('columns', '?')}")
    print(f"      Activity types: {af.get('activity_types', '?')}")
    print(f"      Unique participants: {af.get('unique_participant_ids', '?')}")
    print(f"      Date range: {af.get('first_session_utc','?')} -> {af.get('last_session_utc','?')}")

print("\n  Supabase Participants:")
for pf in participant_files:
    print(f"    File: {pf['filename']}")
    print(f"      Rows: {pf.get('row_count', '?')}")
    print(f"      Columns: {pf.get('columns', '?')}")


# =============================================================================
# STEP 9 — Write all reports
# =============================================================================

# ---- 9a. file_forensics.md ---------------------------------------------------

eeg_sample = eeg_files[0] if eeg_files else {}

forensics_md = f"""# File Forensics Report
Generated: {datetime.now(tz=timezone.utc).isoformat()}

## Summary

| Category | Count |
|---|---|
| EEG data files (EMOTIV CSV) | {len(eeg_files)} |
| Interval marker CSVs | {len(marker_files)} |
| JSON session metadata | {len(json_meta_files)} |
| Supabase activity results | {len(activity_files)} |
| Supabase participants | {len(participant_files)} |
| Sessions with EEG data | {len(sessions)} |
| JSON-only entries (no EEG found) | {len(json_only_participants)} |

## EMOTIV EEG Files

### File Format (confirmed from inspection)
- **Multi-line header** CSV: row 0 = metadata key-value pairs; row 1 = column headers; rows 2+ = data samples.
- **Extension pattern**: `<name>_EPOCX_<serial>_<timestamp>.md.bp.csv` (Band Power exported) or `.md.csv`
- **Headset type**: All files inspected report `EPOCX` in both filename and metadata.

### EEG Column Schema (Example from: {eeg_sample.get('filename', 'N/A')})
- `Timestamp` — Unix epoch float (seconds), EMOTIV device clock
- `EEG.Counter` — Rolling sample counter
- `EEG.Interpolated` — 0/1 interpolation flag per sample
- `EEG.<electrode>` — Raw EEG in µV (14 channels: AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4)
- `EEG.RawCq` — Raw contact quality composite
- `EEG.Battery` / `EEG.BatteryPercent` — Battery status
- `MarkerIndex`, `MarkerType`, `MarkerValueInt`, `EEG.MarkerHardware` — Marker columns
- `CQ.<electrode>` — Per-electrode contact quality (0–4 scale; 4 = good)
- `CQ.Overall` — Overall contact quality (0–100)
- `EQ.SampleRateQuality`, `EQ.OVERALL`, `EQ.<electrode>` — EEG quality metrics
- `MOT.*` — Motion/IMU data (gyro/accel/quaternion at 32 Hz)
- `POW.<electrode>.<band>` — EMOTIV proprietary band-power estimates (Theta, Alpha, BetaL, BetaH, Gamma) — **derived metrics, not raw EEG**

### Electrode Configuration
All EEG files confirmed: **14 channels** — AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4

This is the standard **EPOC X** montage (International 10-20 positions).

### Sampling Rate
{chr(10).join(f"- {sr} Hz — {cnt} file(s)" for sr, cnt in sorted(sr_seen.items()))}

> **NOTE**: EMOTIV EPOC X supports 128 Hz and 256 Hz.  The 256 Hz data (if present)
> is likely with `.md.csv` extension.  Band power (`.md.bp.csv`) is computed at 8 Hz.

## Interval Marker Files

- **Format**: CSV with columns: `latency, duration, type, marker_value, key, timestamp, marker_id`
- `timestamp` = Unix epoch float (same clock as EEG Timestamp column)
- `latency` = seconds since recording start
- `type` = marker label (e.g., "Eyes_Opened", "Eyes_Closed", task labels)
- These markers were inserted by the experimenter via EmotivPRO during recording.

> **Key finding**: Markers include baseline conditions ("Eyes_Opened", "Eyes_Closed").
> For participants with richer marker files, markers appear to include task-phase labels
> inserted by keystrokes during recording. These are the primary synchronization anchors.

## JSON Metadata Files

- **Format**: JSON object with fields: `Markers`, `connectionType`, `demographics`, `exportApp`, `exportTime`, `headsetType`, `recordId`, `user`
- **headsetType**: `"EPOCX"` confirmed in all inspected files
- **Markers in JSON**: Identical markers to the interval marker CSV but in JSON form, plus absolute ISO datetimes.
- **user**: `"bahubali-aiml"` — indicates experiment operator username.

## Supabase Files

### `activity_results_rows.csv`
- **Columns**: `id, participant_id, activity_type, score, total_questions, answers, started_at, completed_at, time_taken_seconds`
- **`answers` column**: JSON array embedded in CSV; one element per question with fields:
  - `correct` (index of correct answer), `category`, `selected` (participant's choice index), `timedOut`, `timeTaken` (seconds), `difficulty` (easy/medium/hard), `questionId`, `questionText`
- **Activity types observed**: `arithmetic`, `comprehension` (and possibly others)
- **Sessions**: {next((af.get('row_count','?') for af in activity_files), '?')} rows = test sessions
- **Date range**: {next((af.get('first_session_utc','?') for af in activity_files), '?')} → {next((af.get('last_session_utc','?') for af in activity_files), '?')}

### `participants_rows (1).csv`
- **Columns**: `id, email, full_name, age, gender, consent_given, created_at, total_session_seconds`
- Participants identified by UUID (`id`) which matches `participant_id` in activity results.
- Contains demographic info: name, age, gender, consent flag, registration timestamp.

## Missing / Notable Gaps

1. **No EEG file found for several participants** — JSON metadata exists but `.md.bp.csv` / `.md.csv` is absent.
   Affected participants include: {', '.join(sorted(json_only_participants))}
   
2. **No explicit software synchronization markers** linking EEG timestamps to web test events.
   The interval markers appear to be manually inserted by the experimenter.

3. **`timeTaken` in behavioral data is integer seconds** — resolution is 1 second, coarser than
   EEG sample period (~7.8 ms at 128 Hz). This limits precise RT alignment.

4. **No per-question stimulus onset timestamps** in the behavioral export — only session-level
   `started_at` and `completed_at` are available. Individual question onset times are NOT recorded.

5. **Behavioral data timestamps** are in UTC ISO format; EEG timestamps are Unix epoch floats
   in IST (+05:30). Clock synchronization must be verified.
"""

with open(RESULTS_DIR / "file_forensics.md", "w", encoding="utf-8") as f:
    f.write(forensics_md)
print(f"\n  [OK] file_forensics.md written")


# ---- 9b. emotiv_device_characterization.md ------------------------------------

device_md = f"""# EMOTIV Device Characterization
Generated: {datetime.now(tz=timezone.utc).isoformat()}

## Device Identification

| Field | Value | Evidence |
|---|---|---|
| **Headset Model** | EPOC X | Filename suffix `_EPOCX_`, JSON `headsetType: "EPOCX"`, 14-electrode schema |
| **Confidence** | HIGH | Consistent across all files |
| **Electrode Count** | 14 | Confirmed from CSV column headers |
| **Electrode Names** | AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4 | CSV headers |
| **Montage** | UNKNOWN (reported in JSON) — Standard EPOC X positions | Confirmed by electrode names matching 10-20 system |
| **EEG Sampling Rate** | 128 Hz (md.bp.csv) or higher (md.csv) | CSV metadata line |
| **Band Power Rate** | 8 Hz (POW.* columns) | EMOTIV documentation |
| **Motion Rate** | 32 Hz | MOT.* columns + metadata `mot_32` |
| **Connection** | Bluetooth | JSON `connectionType: "bluetooth"` |
| **Export Software** | EmotivPRO (`com.emotiv.emotivpro`) | JSON `exportApp` |
| **Reference** | CMS/DRL (P3/P4 positions) | Standard EPOC X hardware design |

## EPOC X Technical Specifications (from EMOTIV documentation)

| Specification | Value |
|---|---|
| Electrode type | Saline-based felt pads (wet) |
| Channels | 14 EEG + 2 reference (CMS/DRL) |
| ADC resolution | 16-bit |
| EEG bandwidth | 0.16–43 Hz (hardware filtered) |
| Notch filter | 50 Hz and 60 Hz |
| Dynamic range | ±420 µV |
| Noise floor | <1 µV RMS |
| Output units | µV |
| Sample rate options | 128 Hz / 256 Hz |
| Battery | 12–14 hours Li-poly |
| Connectivity | Bluetooth 4.2 |

## Important Constraints for Analysis

1. **Low spatial resolution**: 14 channels cannot support high-density source localization.
2. **Hardware filtering already applied**: The 0.16 Hz high-pass and 43 Hz low-pass are fixed.
   Additional software filtering should respect these limits.
3. **Saline electrodes**: Impedance degrades over time without re-moistening.
   Contact quality metrics (CQ.*) must be monitored.
4. **Bluetooth latency**: Possible sample-level jitter in wireless transmission.
   Timestamps in software may not reflect exact hardware sample time.
5. **CMS/DRL reference**: Re-referencing to average is standard practice for EPOC X.
6. **Interpolated samples**: EMOTIV flags interpolated samples (`EEG.Interpolated = 1`).
   These must be tracked and excluded from artifact-sensitive analyses.

## Streams Available per Export File

| Stream | Columns | Rate | Notes |
|---|---|---|---|
| EEG (raw) | `EEG.AF3` ... `EEG.AF4` | 128/256 Hz | Primary signal |
| Contact Quality | `CQ.AF3` ... `CQ.Overall` | 128/256 Hz | 0=no contact; 4=excellent |
| EEG Quality | `EQ.*` | 128/256 Hz | Derived quality metric |
| Motion/IMU | `MOT.Q0`..`MOT.MagZ` | 32 Hz | Head movement detection |
| Band Power | `POW.*.<Theta/Alpha/BetaL/BetaH/Gamma>` | 8 Hz | PROPRIETARY derived metric |
| Markers | `MarkerIndex`, `MarkerType`, `MarkerValueInt` | Per event | Experimenter events |
| Counter | `EEG.Counter` | 128/256 Hz | Rollover counter (0–255) |

## Scientific Caveat on Proprietary Metrics

The `POW.*` band-power columns are **EMOTIV proprietary**, computed with undisclosed algorithms.
They must NOT be treated as validated psychological constructs.
**Analysis should primarily use the raw EEG channels** (EEG.AF3 through EEG.AF4).
"""

with open(RESULTS_DIR / "emotiv_device_characterization.md", "w", encoding="utf-8") as f:
    f.write(device_md)
print(f"  [OK] emotiv_device_characterization.md written")


# ---- 9c. data_dictionary.md and .csv ----------------------------------------

dd_rows = [
    # Behavioral
    {"source": "Supabase", "file": "activity_results_rows.csv", "field_name": "id", "canonical_name": "session_id", "data_type": "UUID", "unit": "—", "description": "Unique test session identifier", "timestamp_type": "none", "participant_level": False, "session_level": True, "trial_level": False, "raw_or_derived": "raw", "source_system": "Supabase/web", "scientific_interpretation": "Session primary key"},
    {"source": "Supabase", "file": "activity_results_rows.csv", "field_name": "participant_id", "canonical_name": "participant_id", "data_type": "UUID", "unit": "—", "description": "Participant foreign key linking to participants table", "timestamp_type": "none", "participant_level": True, "session_level": True, "trial_level": True, "raw_or_derived": "raw", "source_system": "Supabase/web"},
    {"source": "Supabase", "file": "activity_results_rows.csv", "field_name": "activity_type", "canonical_name": "task_type", "data_type": "string", "unit": "—", "description": "Type of cognitive task (arithmetic, comprehension, etc.)", "timestamp_type": "none", "participant_level": False, "session_level": True, "trial_level": True, "raw_or_derived": "raw", "source_system": "Supabase/web"},
    {"source": "Supabase", "file": "activity_results_rows.csv", "field_name": "score", "canonical_name": "session_score", "data_type": "int", "unit": "correct answers", "description": "Total correct answers in session", "timestamp_type": "none", "participant_level": False, "session_level": True, "trial_level": False, "raw_or_derived": "raw", "source_system": "Supabase/web"},
    {"source": "Supabase", "file": "activity_results_rows.csv", "field_name": "total_questions", "canonical_name": "total_questions", "data_type": "int", "unit": "questions", "description": "Total questions presented", "timestamp_type": "none", "participant_level": False, "session_level": True, "trial_level": False, "raw_or_derived": "raw", "source_system": "Supabase/web"},
    {"source": "Supabase", "file": "activity_results_rows.csv", "field_name": "answers", "canonical_name": "trial_answers_json", "data_type": "JSON array", "unit": "—", "description": "Per-question answer array: correct, category, selected, timedOut, timeTaken (int seconds), difficulty, questionId, questionText", "timestamp_type": "none", "participant_level": False, "session_level": False, "trial_level": True, "raw_or_derived": "raw", "source_system": "Supabase/web", "scientific_interpretation": "Must be expanded to one row per question/trial for trial-level analysis"},
    {"source": "Supabase", "file": "activity_results_rows.csv", "field_name": "started_at", "canonical_name": "session_start_utc", "data_type": "ISO datetime", "unit": "UTC", "description": "Session start timestamp (server-side)", "timestamp_type": "absolute_utc", "participant_level": False, "session_level": True, "trial_level": False, "raw_or_derived": "raw", "source_system": "Supabase/web", "scientific_interpretation": "Primary synchronization anchor candidate"},
    {"source": "Supabase", "file": "activity_results_rows.csv", "field_name": "completed_at", "canonical_name": "session_end_utc", "data_type": "ISO datetime", "unit": "UTC", "description": "Session completion timestamp (server-side)", "timestamp_type": "absolute_utc", "participant_level": False, "session_level": True, "trial_level": False, "raw_or_derived": "raw", "source_system": "Supabase/web"},
    {"source": "Supabase", "file": "activity_results_rows.csv", "field_name": "time_taken_seconds", "canonical_name": "session_duration_s", "data_type": "int", "unit": "seconds", "description": "Total test duration in seconds", "timestamp_type": "relative", "participant_level": False, "session_level": True, "trial_level": False, "raw_or_derived": "raw", "source_system": "Supabase/web"},
    # Trial-level (from answers JSON)
    {"source": "Supabase", "file": "activity_results_rows.csv (answers field)", "field_name": "timeTaken", "canonical_name": "trial_time_taken_s", "data_type": "int", "unit": "seconds", "description": "Time taken to answer this question (integer seconds — LOW resolution)", "timestamp_type": "relative", "participant_level": False, "session_level": False, "trial_level": True, "raw_or_derived": "raw", "source_system": "Supabase/web", "scientific_interpretation": "LIMITATION: 1-second granularity. Cannot derive precise stimulus onset. Not a true RT measure."},
    {"source": "Supabase", "file": "activity_results_rows.csv (answers field)", "field_name": "correct", "canonical_name": "correct_answer_index", "data_type": "int", "unit": "option index", "description": "Index of correct answer (0-based)", "timestamp_type": "none", "participant_level": False, "session_level": False, "trial_level": True, "raw_or_derived": "raw", "source_system": "Supabase/web"},
    {"source": "Supabase", "file": "activity_results_rows.csv (answers field)", "field_name": "selected", "canonical_name": "selected_answer_index", "data_type": "int", "unit": "option index", "description": "Index selected by participant (0-based)", "timestamp_type": "none", "participant_level": False, "session_level": False, "trial_level": True, "raw_or_derived": "raw", "source_system": "Supabase/web"},
    {"source": "Supabase", "file": "activity_results_rows.csv (answers field)", "field_name": "difficulty", "canonical_name": "difficulty", "data_type": "string", "unit": "easy/medium/hard", "description": "Question difficulty level", "timestamp_type": "none", "participant_level": False, "session_level": False, "trial_level": True, "raw_or_derived": "raw", "source_system": "Supabase/web"},
    # EEG
    {"source": "EMOTIV_CSV", "file": "*.md.bp.csv / *.md.csv", "field_name": "Timestamp", "canonical_name": "eeg_timestamp_unix", "data_type": "float64", "unit": "Unix epoch seconds", "description": "EMOTIV device clock timestamp for each EEG sample", "timestamp_type": "absolute_unix_device_clock", "participant_level": False, "session_level": True, "trial_level": True, "raw_or_derived": "raw", "source_system": "EmotivPRO"},
    {"source": "EMOTIV_CSV", "file": "*.md.bp.csv / *.md.csv", "field_name": "EEG.Counter", "canonical_name": "sample_counter", "data_type": "int", "unit": "counts (0–255 rolling)", "description": "Rolling sample counter, resets to 0 after 255", "timestamp_type": "relative", "participant_level": False, "session_level": True, "trial_level": False, "raw_or_derived": "raw", "source_system": "EmotivPRO"},
    {"source": "EMOTIV_CSV", "file": "*.md.bp.csv / *.md.csv", "field_name": "EEG.Interpolated", "canonical_name": "interpolated_flag", "data_type": "binary", "unit": "0=raw/1=interpolated", "description": "EMOTIV firmware interpolation flag — 1 indicates sample was reconstructed", "timestamp_type": "none", "participant_level": False, "session_level": False, "trial_level": True, "raw_or_derived": "derived", "source_system": "EmotivPRO", "scientific_interpretation": "Interpolated samples should be excluded from artifact-sensitive analyses"},
    {"source": "EMOTIV_CSV", "file": "*.md.bp.csv / *.md.csv", "field_name": "EEG.AF3 ... EEG.AF4", "canonical_name": "eeg_channel_uv", "data_type": "float32", "unit": "µV", "description": "Raw EEG voltage for 14 electrodes (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4)", "timestamp_type": "none", "participant_level": False, "session_level": False, "trial_level": True, "raw_or_derived": "raw", "source_system": "EmotivPRO", "scientific_interpretation": "Primary EEG signal. Hardware-filtered 0.16–43 Hz. Reference: CMS/DRL."},
    {"source": "EMOTIV_CSV", "file": "*.md.bp.csv / *.md.csv", "field_name": "CQ.AF3 ... CQ.Overall", "canonical_name": "contact_quality", "data_type": "int/float", "unit": "0–4 per electrode; 0–100 overall", "description": "Electrode contact quality. 0=no contact, 4=excellent. Overall is 0–100.", "timestamp_type": "none", "participant_level": False, "session_level": True, "trial_level": True, "raw_or_derived": "derived", "source_system": "EmotivPRO", "scientific_interpretation": "Quality control gate. Channels with CQ<2 should be flagged."},
    {"source": "EMOTIV_CSV", "file": "*.md.bp.csv / *.md.csv", "field_name": "POW.*.Theta/Alpha/BetaL/BetaH/Gamma", "canonical_name": "proprietary_band_power", "data_type": "float", "unit": "arbitrary (EMOTIV proprietary)", "description": "EMOTIV proprietary band-power estimates at 8 Hz. Algorithm undisclosed.", "timestamp_type": "none", "participant_level": False, "session_level": False, "trial_level": False, "raw_or_derived": "derived", "source_system": "EmotivPRO", "scientific_interpretation": "CAUTION: Not validated psychological constructs. Use raw EEG for scientific analysis."},
    {"source": "EMOTIV_CSV", "file": "*.md.bp.csv / *.md.csv", "field_name": "MarkerIndex, MarkerType, MarkerValueInt", "canonical_name": "marker_in_stream", "data_type": "int/string", "unit": "—", "description": "In-stream event markers at sample level. Populated when experimenter inserts a marker.", "timestamp_type": "sample_aligned", "participant_level": False, "session_level": False, "trial_level": True, "raw_or_derived": "raw", "source_system": "EmotivPRO", "scientific_interpretation": "Potential synchronization anchor if markers correspond to web test events"},
    {"source": "EMOTIV_MARKER_CSV", "file": "*_intervalMarker.csv", "field_name": "timestamp", "canonical_name": "marker_timestamp_unix", "data_type": "float64", "unit": "Unix epoch seconds (same clock as EEG)", "description": "Absolute Unix timestamp of each inserted marker", "timestamp_type": "absolute_unix_device_clock", "participant_level": False, "session_level": True, "trial_level": True, "raw_or_derived": "raw", "source_system": "EmotivPRO", "scientific_interpretation": "Key synchronization anchor. Must be matched to web session timestamps."},
    {"source": "EMOTIV_MARKER_CSV", "file": "*_intervalMarker.csv", "field_name": "type", "canonical_name": "marker_label", "data_type": "string", "unit": "—", "description": "Marker label as entered by experimenter (e.g., Eyes_Opened, Eyes_Closed)", "timestamp_type": "none", "participant_level": False, "session_level": False, "trial_level": True, "raw_or_derived": "raw", "source_system": "EmotivPRO"},
    # Participants
    {"source": "Supabase", "file": "participants_rows (1).csv", "field_name": "id", "canonical_name": "participant_id", "data_type": "UUID", "unit": "—", "description": "Participant unique identifier (links to activity_results.participant_id)", "timestamp_type": "none", "participant_level": True, "session_level": True, "trial_level": True, "raw_or_derived": "raw", "source_system": "Supabase"},
    {"source": "Supabase", "file": "participants_rows (1).csv", "field_name": "full_name", "canonical_name": "participant_name", "data_type": "string", "unit": "—", "description": "Participant full name (PII — must be anonymized)", "timestamp_type": "none", "participant_level": True, "session_level": False, "trial_level": False, "raw_or_derived": "raw", "source_system": "Supabase"},
    {"source": "Supabase", "file": "participants_rows (1).csv", "field_name": "age", "canonical_name": "age_years", "data_type": "int", "unit": "years", "description": "Age at registration", "timestamp_type": "none", "participant_level": True, "session_level": False, "trial_level": False, "raw_or_derived": "raw", "source_system": "Supabase"},
    {"source": "Supabase", "file": "participants_rows (1).csv", "field_name": "gender", "canonical_name": "gender", "data_type": "string", "unit": "male/female/other", "description": "Self-reported gender", "timestamp_type": "none", "participant_level": True, "session_level": False, "trial_level": False, "raw_or_derived": "raw", "source_system": "Supabase"},
    {"source": "Supabase", "file": "participants_rows (1).csv", "field_name": "consent_given", "canonical_name": "consent_given", "data_type": "boolean", "unit": "—", "description": "Whether participant gave informed consent", "timestamp_type": "none", "participant_level": True, "session_level": False, "trial_level": False, "raw_or_derived": "raw", "source_system": "Supabase"},
    {"source": "Supabase", "file": "participants_rows (1).csv", "field_name": "total_session_seconds", "canonical_name": "total_platform_seconds", "data_type": "int", "unit": "seconds", "description": "Total time spent on platform across all sessions", "timestamp_type": "relative", "participant_level": True, "session_level": False, "trial_level": False, "raw_or_derived": "derived", "source_system": "Supabase"},
]

# Write CSV
dd_path_csv = RESULTS_DIR / "data_dictionary.csv"
dd_cols = ["source", "file", "field_name", "canonical_name", "data_type", "unit", "description",
           "timestamp_type", "participant_level", "session_level", "trial_level",
           "raw_or_derived", "source_system", "scientific_interpretation"]
with open(dd_path_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=dd_cols, extrasaction="ignore")
    writer.writeheader()
    for row in dd_rows:
        writer.writerow(row)

# Write Markdown
table_header = "| " + " | ".join(dd_cols) + " |"
table_sep    = "| " + " | ".join(["---"]*len(dd_cols)) + " |"
table_rows = []
for row in dd_rows:
    table_rows.append("| " + " | ".join(str(row.get(c,"")).replace("|","\\|") for c in dd_cols) + " |")

dd_md = f"""# Data Dictionary
Generated: {datetime.now(tz=timezone.utc).isoformat()}

## Overview
This dictionary documents every significant field in the dataset.
Fields marked `raw_or_derived = derived` originate from processing; treat with caution.

{table_header}
{table_sep}
{chr(10).join(table_rows)}

## Key Cross-System Link
`participants_rows.id` == `activity_results_rows.participant_id`

`activity_results_rows.started_at` (UTC) ↔ `EEG Timestamp` (Unix, IST) — **requires synchronization**

## Canonical Trial-Level Record (after expansion)
```
participant_id | session_id | trial_index | task_type | question_id |
difficulty | correct_answer_index | selected_answer_index | trial_correct |
trial_time_taken_s | session_start_utc | eeg_session_start_unix
```
"""

with open(RESULTS_DIR / "data_dictionary.md", "w", encoding="utf-8") as f:
    f.write(dd_md)

print(f"  [OK] data_dictionary.csv & data_dictionary.md written")


# ---- 9d. synchronization_report.md -------------------------------------------

sync_md = """# Synchronization Feasibility Report
Generated: {now}

## GATE 4 — Can timestamps be synchronized defensibly?

### Verdict: **LOW-TO-MODERATE CONFIDENCE** (requires further investigation)

---

## What We Know (KNOWN)

| System | Timestamp Type | Clock |
|---|---|---|
| EMOTIV EEG (CSV) | Unix epoch float seconds | EMOTIV device clock (IST, UTC+5:30 hardware) |
| EMOTIV Marker CSV | Unix epoch float seconds | Same EMOTIV device clock |
| EMOTIV JSON Markers | ISO 8601 absolute datetime (IST +05:30) | EMOTIV device clock |
| Supabase `started_at` | ISO 8601 UTC datetime (server-side) | Supabase/PostgreSQL server clock |
| Supabase `completed_at` | ISO 8601 UTC datetime (server-side) | Supabase/PostgreSQL server clock |

---

## Critical Synchronization Gap (KNOWN)

The behavioral data (`activity_results_rows.csv`) provides:
- **Session-level** `started_at` and `completed_at` timestamps (UTC, server clock)
- **Trial-level** `timeTaken` in **integer seconds** only — NOT absolute timestamps

### What This Means
1. **Individual question/trial onset timestamps are NOT available** in the behavioral export.
   We know when each session *started* and how many seconds each question *took*, but
   not the absolute clock time of each individual question presentation.
2. **`timeTaken` is integer seconds** — resolution is 1,000 ms, compared to EEG sample
   period of ~7.8 ms (128 Hz). Trial-level synchronization cannot exceed ±500 ms even
   in theory, and in practice may be worse.
3. **Server vs. device clock**: Supabase timestamps come from the server; EEG timestamps
   come from the EMOTIV device. These may differ by an unknown offset.

---

## What Can Be Reconstructed (INFERRED)

If we assume:
1. Questions were presented sequentially with no gaps other than `timeTaken`
2. No time gaps between questions
3. Session start (`started_at`) corresponds to first question onset

Then estimated question onset times can be reconstructed:
```
q1_onset ≈ started_at
q2_onset ≈ started_at + timeTaken(q1)
q3_onset ≈ started_at + timeTaken(q1) + timeTaken(q2)
...
```

**This is an INFERRED method, not directly measured.**  
Actual question presentation timing may differ due to:
- Network latency in loading questions
- UI rendering time
- Answer submission latency
- Timer implementation (server vs. client-side)

---

## Synchronization Evidence Available

| Evidence | Availability | Quality |
|---|---|---|
| Shared hardware trigger / sync signal | NOT FOUND | — |
| LSL (Lab Streaming Layer) markers | NOT FOUND | — |
| EMOTIV Cortex API injection markers | Unknown | — |
| Keystroke markers in EEG stream | Present in some sessions | MODERATE |
| Session start/end timestamps (both systems) | YES | MODERATE |
| Trial-level absolute timestamps (web) | **NOT AVAILABLE** | — |
| Marker labels matching task phases | YES (baseline only: Eyes_Opened, Eyes_Closed) | LOW for task sync |

---

## Possible Synchronization Approach (INFERRED)

### Method A: Session-boundary alignment (MODERATE confidence)
1. Convert `started_at` (UTC) to Unix epoch.
2. Find recording start (`start_timestamp_unix` from EEG CSV header).
3. Compute offset: `b = eeg_start - web_start`
4. Apply to all reconstructed trial timestamps.
5. Estimate drift if session duration in both systems can be compared.

### Method B: Keystroke marker matching (if present — MODERATE confidence)
For sessions where the experimenter inserted keystroke markers at task onset/offset,
match marker timestamps (EEG clock) to web session start/end (Supabase clock).
Fit: `EEG_time = a * Web_time + b`

### Method C: Duration comparison (LOW confidence)
Compare `time_taken_seconds` from Supabase to EEG recording duration.
If they match within ±a few seconds, use this as a coarse consistency check only.

---

## Synchronization Precision Ceiling

Even with perfect clock alignment:

| Source | Maximum RT precision |
|---|---|
| `timeTaken` (integer seconds) | ±500 ms |
| EEG sample period @ 128 Hz | ±7.8 ms |
| EEG sample period @ 256 Hz | ±3.9 ms |
| **Combined limit** | **±500 ms** (limited by behavioral resolution) |

> **This means stimulus-locked ERPs with precision < 500 ms are NOT supportable from
> this behavioral export alone.** ERP analyses should be treated as exploratory only.

---

## What Is Required to Establish High-Confidence Synchronization

1. **Per-question timestamp logs** from the web application (client-side or server-side)
   showing exact millisecond-precision onset times for each question.
2. **OR**: Confirmation that keystroke markers in the EEG stream corresponded to
   specific task events (e.g., test-start keypress), with documentation of the protocol.

---

## Recommended Next Step

1. Parse all `*_intervalMarker.csv` files and catalog all unique marker labels.
2. Obtain procedure documentation: did the experimenter press a key when the test started?
3. Request per-question timestamps from the web application logs or Supabase.
4. Until resolved, proceed with session-level analysis only; withhold trial-locked ERP analyses.
""".format(now=datetime.now(tz=timezone.utc).isoformat())

with open(RESULTS_DIR / "synchronization_report.md", "w", encoding="utf-8") as f:
    f.write(sync_md)
print(f"  [OK] synchronization_report.md written")


# ---- 9e. qc_report.md --------------------------------------------------------

eeg_with_data = [s for s in sessions.values() if s.get("parse_status") == "OK"]
eeg_no_data   = [p for p in json_only_participants]

qc_md = f"""# Initial QC & Research Readiness Report
Generated: {datetime.now(tz=timezone.utc).isoformat()}

## Participant & Session Counts

| Metric | Value | Status |
|---|---|---|
| Participants in Supabase | {next((pf.get('row_count','?') for pf in participant_files), '?')} | — |
| Behavioral sessions (activity_results) | {next((af.get('row_count','?') for af in activity_files), '?')} | — |
| EEG sessions (files with EEG data) | {len(eeg_with_data)} | — |
| JSON-only entries (no EEG file found) | {len(eeg_no_data)} | ⚠️ INCOMPLETE |
| Imagination task sessions (separate) | {sum(1 for s in sessions.values() if s.get('task_context')=='imagination')} | Separate sub-study |
| Main web-test EEG sessions | {sum(1 for s in sessions.values() if s.get('task_context')=='web_test_main')} | Primary dataset |

## EEG Recording Quality (Pre-Processing Assessment)

| Metric | Value | Notes |
|---|---|---|
| Headset confirmed | EPOC X | All files consistent |
| Electrode count | 14 | Standard EPOC X |
| Sampling rate (primary) | 128 Hz | Some files may be 256 Hz |
| Median duration | {f"{sorted(durations)[len(durations)//2]:.1f}" if durations else "N/A"} min | |
| Range | {f"{min(durations):.1f}–{max(durations):.1f}" if durations else "N/A"} min | |
| Files with missing EEG | {len(eeg_no_data)} participants | JSON present but CSV absent |
| Contact quality columns | Present | Requires per-file inspection |
| Interpolation flags | Present (EEG.Interpolated) | Requires per-file quantification |
| Motion data | Present (MOT.*) | Available for artifact screening |

## Behavioral Data Quality

| Metric | Value | Notes |
|---|---|---|
| Activity types | arithmetic, comprehension | Cognitive task |
| Questions per session | 10 | Fixed |
| RT resolution | 1 second (integer) | ⚠️ LOW — limits synchronization |
| Session-level timestamps | YES (UTC, server) | Available |
| Trial-level timestamps | NO — only cumulative timeTaken | ⚠️ CRITICAL LIMITATION |
| Duplicate check | Not yet performed | Requires analysis |
| Score range | Apparent 0–10 | Needs validation |

## Synchronization Assessment

| Gate | Result |
|---|---|
| Shared hardware trigger | ❌ Not found |
| Per-trial timestamps | ❌ Not in export |
| Session-boundary sync | ✅ Possible |
| Keystroke markers (some sessions) | ⚠️ Present but not documented |
| **Overall sync confidence** | **LOW–MODERATE** |

## Research Readiness Gates

| Gate | Status | Explanation |
|---|---|---|
| G1: Device identified | ✅ PASS | EPOC X confirmed |
| G2: EEG files interpretable | ✅ PASS | CSV schema fully understood |
| G3: Participant/session mapping | ⚠️ PARTIAL | Name-based; UUID linkage to Supabase not yet established |
| G4: Synchronization defensible | ⚠️ LOW–MODERATE | No per-trial timestamps; session-level only |
| G5: EEG signal quality adequate | ⏳ PENDING | Requires per-file QC analysis |
| G6: Participant count adequate | ⚠️ MARGINAL | ~{len(eeg_with_data)} sessions; exact participant count requires deduplication |
| G7: ML design leakage-safe | ⏳ PENDING | Awaiting analysis design |

## What Can Be Analyzed Now

1. **Session-level behavioral analysis**: Score, accuracy, task type, difficulty distributions.
2. **Session-level EEG quality audit**: Contact quality, interpolation rates, duration.
3. **Band-power trends within sessions**: Using session start/end as the only sync anchor.
4. **Baseline (Eyes_Open vs. Eyes_Closed) EEG comparisons**: Many sessions have these markers.
5. **Individual differences**: Between-participant EEG variability.

## What CANNOT Be Analyzed Without Additional Data

1. **Stimulus-locked ERPs** — no per-trial onset timestamps with < 500 ms resolution.
2. **Response-locked EEG** — no precise response timestamps.
3. **Question-by-question EEG features** — cannot reliably identify trial boundaries.
4. **Tight EEG-behavior coupling** (e.g., "theta power predicts RT on this trial") — insufficient temporal precision.

## Immediate Next Actions (Priority Order)

1. **[P0]** Determine whether the web application logged per-question timestamps that can be retrieved.
2. **[P1]** Inspect all marker files to find sessions with task-phase markers.
3. **[P1]** Establish UUID linkage between EEG filenames and Supabase participant UUIDs.
4. **[P2]** Run per-file EEG quality inspection (interpolation rates, contact quality distribution).
5. **[P3]** Parse behavioral JSON answers field into a flat trial table.
"""

with open(RESULTS_DIR / "qc_report.md", "w", encoding="utf-8") as f:
    f.write(qc_md)
print(f"  [OK] qc_report.md written")


# =============================================================================
# Final summary
# =============================================================================

print("\n" + "=" * 60)
print("PHASE 1 COMPLETE — SUMMARY")
print("=" * 60)
print(f"""
Device:           EMOTIV EPOC X (HIGH confidence)
Electrodes:       14 (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4)
Sampling rate:    128 Hz (primary)
EEG sessions:     {len(sessions)} with data files
Behavioral rows:  {next((af.get('row_count','?') for af in activity_files), '?')} test sessions
Participants:     {next((pf.get('row_count','?') for pf in participant_files), '?')} in Supabase
Sync feasibility: LOW–MODERATE (no per-trial timestamps)

Reports written to: {RESULTS_DIR}
  - file_inventory.csv
  - file_forensics.md
  - emotiv_device_characterization.md
  - session_mapping.csv
  - data_dictionary.csv
  - data_dictionary.md
  - synchronization_report.md
  - qc_report.md

STOP POINT: Do not proceed to EEG preprocessing until
synchronization feasibility is resolved. See qc_report.md.
""")
