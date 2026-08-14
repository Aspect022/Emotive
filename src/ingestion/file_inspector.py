"""
file_inspector.py
-----------------
Phase 1 inspection module: inventories all raw data files, inspects
headers/schemas without modifying any source file, and produces
structured metadata for downstream analysis.

Scientific principle:  Never assume file contents. Inspect everything.
"""

import os
import json
import hashlib
import csv
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# EEG CSV (.md.bp.csv / .md.csv) inspection
# ---------------------------------------------------------------------------

EMOTIV_CSV_SUFFIXES = (".md.bp.csv", ".md.csv")


def is_emotiv_eeg_csv(path: Path) -> bool:
    name = path.name.lower()
    return any(name.endswith(s) for s in EMOTIV_CSV_SUFFIXES)


def inspect_emotiv_eeg_csv(path: Path) -> dict:
    """
    Parse the EMOTIV EmotivPRO multi-stream CSV.
    Row 0 is a metadata line (key:value pairs).
    Row 1 is the column header.
    Rows 2+ are data.
    """
    result = {
        "filepath": str(path),
        "filename": path.name,
        "extension": path.suffix,
        "file_type": "EMOTIV_EEG_CSV",
        "size_mb": round(file_size_mb(path), 3),
        "mtime_utc": mtime_iso(path),
        "sha256": sha256_file(path),
    }
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            meta_line = f.readline().rstrip("\n")
            header_line = f.readline().rstrip("\n")
            first_data = f.readline().rstrip("\n")

        # --- parse metadata line ---
        meta = {}
        for part in meta_line.split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                meta[k.strip()] = v.strip()

        result["title"] = meta.get("title", "UNKNOWN")
        result["start_timestamp_unix"] = _parse_float(meta.get("start timestamp", None))
        result["stop_timestamp_unix"] = _parse_float(meta.get("stop timestamp", None))
        result["headset_type"] = meta.get("headset type", "UNKNOWN")
        result["headset_serial"] = meta.get("headset serial", "UNKNOWN")
        result["sampling_rate_raw"] = meta.get("sampling rate", "UNKNOWN")
        result["channels_reported"] = _parse_int(meta.get("channels", None))
        result["samples_reported"] = _parse_int(meta.get("samples", None))
        result["export_version"] = meta.get("version", "UNKNOWN")

        # Parse multi-rate: e.g. "eeg_128;mot_32;pow_8"
        sr_raw = meta.get("sampling rate", "")
        eeg_sr = None
        for tok in sr_raw.split(";"):
            if tok.lower().startswith("eeg_"):
                try:
                    eeg_sr = int(tok.split("_")[1])
                except Exception:
                    pass
        result["eeg_sampling_rate_hz"] = eeg_sr

        # --- parse column header ---
        columns = [c.strip() for c in header_line.split(",")]
        result["column_count"] = len(columns)
        result["columns"] = columns

        eeg_cols = [c for c in columns if c.startswith("EEG.") and not c.startswith("EEG.Cq") and not c.startswith("EEG.Battery") and c not in ("EEG.Counter", "EEG.Interpolated", "EEG.RawCq", "EEG.MarkerHardware")]
        # Filter to actual electrode channels
        electrode_cols = [c for c in eeg_cols if c not in (
            "EEG.Counter", "EEG.Interpolated", "EEG.RawCq",
            "EEG.Battery", "EEG.BatteryPercent", "EEG.MarkerHardware",
            "MarkerIndex", "MarkerType", "MarkerValueInt"
        )]
        # Real electrode channels: EEG.AF3, EEG.F7, etc.
        electrode_cols = [c for c in columns if re.match(r"EEG\.[A-Z][A-Z0-9]+$", c)]
        result["eeg_electrode_channels"] = electrode_cols
        result["eeg_electrode_count"] = len(electrode_cols)

        cq_cols = [c for c in columns if c.startswith("CQ.")]
        result["contact_quality_channels"] = cq_cols

        pow_cols = [c for c in columns if c.startswith("POW.")]
        result["power_band_columns"] = len(pow_cols)

        mot_cols = [c for c in columns if c.startswith("MOT.")]
        result["motion_columns"] = mot_cols

        marker_cols = [c for c in columns if "Marker" in c]
        result["marker_columns"] = marker_cols

        # --- timing ---
        if result["start_timestamp_unix"] and result["stop_timestamp_unix"]:
            dur = result["stop_timestamp_unix"] - result["start_timestamp_unix"]
            result["duration_seconds"] = round(dur, 3)
            result["duration_minutes"] = round(dur / 60, 2)
        else:
            result["duration_seconds"] = None
            result["duration_minutes"] = None

        # --- first data row ---
        if first_data:
            vals = first_data.split(",")
            result["first_timestamp_unix"] = _parse_float(vals[0]) if vals else None
        else:
            result["first_timestamp_unix"] = None

        result["parse_status"] = "OK"
        result["inferred_role"] = "EEG_PRIMARY"
        result["confidence"] = "HIGH"

    except Exception as e:
        result["parse_status"] = f"ERROR: {e}"
        result["inferred_role"] = "EEG_PRIMARY_UNPARSED"
        result["confidence"] = "LOW"

    return result


# ---------------------------------------------------------------------------
# Interval marker CSV inspection
# ---------------------------------------------------------------------------

def inspect_interval_marker_csv(path: Path) -> dict:
    result = {
        "filepath": str(path),
        "filename": path.name,
        "extension": path.suffix,
        "file_type": "EMOTIV_INTERVAL_MARKER_CSV",
        "size_mb": round(file_size_mb(path), 3),
        "mtime_utc": mtime_iso(path),
        "sha256": sha256_file(path),
        "inferred_role": "EVENT_MARKER",
        "confidence": "HIGH",
    }
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        result["row_count"] = len(rows)
        result["columns"] = reader.fieldnames
        if rows:
            result["marker_types"] = list({r.get("type", "") for r in rows})
            result["marker_labels"] = list({r.get("type", "") for r in rows})
            # Timestamps
            ts = [_parse_float(r.get("timestamp")) for r in rows if r.get("timestamp")]
            if ts:
                result["first_marker_unix"] = min(ts)
                result["last_marker_unix"] = max(ts)
                result["marker_timespan_seconds"] = round(max(ts) - min(ts), 3)
        result["parse_status"] = "OK"
    except Exception as e:
        result["parse_status"] = f"ERROR: {e}"
    return result


# ---------------------------------------------------------------------------
# JSON metadata file inspection
# ---------------------------------------------------------------------------

def inspect_json_metadata(path: Path) -> dict:
    result = {
        "filepath": str(path),
        "filename": path.name,
        "extension": path.suffix,
        "file_type": "EMOTIV_JSON_METADATA",
        "size_mb": round(file_size_mb(path), 3),
        "mtime_utc": mtime_iso(path),
        "sha256": sha256_file(path),
        "inferred_role": "EEG_METADATA",
        "confidence": "HIGH",
    }
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)

        result["record_id"] = data.get("recordId", "UNKNOWN")
        result["headset_type"] = data.get("headsetType", "UNKNOWN")
        result["export_app"] = data.get("exportApp", "UNKNOWN")
        result["export_time"] = data.get("exportTime", "UNKNOWN")
        result["user"] = data.get("user", "UNKNOWN")
        result["connection_type"] = data.get("connectionType", "UNKNOWN")
        result["montage_type"] = data.get("montageType", "UNKNOWN")

        markers = data.get("Markers", [])
        result["marker_count"] = len(markers)
        if markers:
            result["marker_labels"] = list({m.get("label", "") for m in markers})
            result["marker_ports"] = list({m.get("port", "") for m in markers})
            result["marker_types_json"] = list({m.get("type", "") for m in markers})
            # Timestamps
            starts = [m.get("startDatetime") for m in markers if m.get("startDatetime")]
            if starts:
                result["first_marker_datetime"] = min(starts)
                result["last_marker_datetime"] = max(starts)

        demographics = data.get("demographics", {})
        result["gender"] = demographics.get("gender", "U")
        result["birthday"] = demographics.get("birthday", None)

        result["parse_status"] = "OK"
    except Exception as e:
        result["parse_status"] = f"ERROR: {e}"
    return result


# ---------------------------------------------------------------------------
# Supabase CSV inspection
# ---------------------------------------------------------------------------

def inspect_supabase_csv(path: Path) -> dict:
    result = {
        "filepath": str(path),
        "filename": path.name,
        "extension": path.suffix,
        "size_mb": round(file_size_mb(path), 3),
        "mtime_utc": mtime_iso(path),
        "sha256": sha256_file(path),
    }
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        result["row_count"] = len(rows)
        result["column_count"] = len(reader.fieldnames or [])
        result["columns"] = reader.fieldnames

        # Heuristic classification
        name = path.name.lower()
        if "participant" in name:
            result["file_type"] = "SUPABASE_PARTICIPANTS"
            result["inferred_role"] = "PARTICIPANT_REGISTRY"
            result["confidence"] = "HIGH"
        elif "activity" in name:
            result["file_type"] = "SUPABASE_ACTIVITY_RESULTS"
            result["inferred_role"] = "BEHAVIORAL_TRIALS"
            result["confidence"] = "HIGH"

            # Sample the activity types
            activity_types = list({r.get("activity_type", "") for r in rows if r.get("activity_type")})
            result["activity_types"] = activity_types

            # Timestamp range
            ts_col = "started_at" if "started_at" in (reader.fieldnames or []) else None
            if ts_col:
                ts_vals = [r[ts_col] for r in rows if r.get(ts_col)]
                if ts_vals:
                    result["first_session_utc"] = min(ts_vals)
                    result["last_session_utc"] = max(ts_vals)

            # Unique participants
            pid_col = "participant_id" if "participant_id" in (reader.fieldnames or []) else None
            if pid_col:
                unique_pids = list({r[pid_col] for r in rows if r.get(pid_col)})
                result["unique_participant_ids"] = len(unique_pids)

        result["parse_status"] = "OK"
    except Exception as e:
        result["parse_status"] = f"ERROR: {e}"
        result["file_type"] = "UNKNOWN_CSV"
        result["inferred_role"] = "UNKNOWN"
        result["confidence"] = "LOW"
    return result


# ---------------------------------------------------------------------------
# Generic binary/unknown file inspection
# ---------------------------------------------------------------------------

def inspect_generic(path: Path) -> dict:
    result = {
        "filepath": str(path),
        "filename": path.name,
        "extension": path.suffix,
        "file_type": "UNKNOWN",
        "size_mb": round(file_size_mb(path), 3),
        "mtime_utc": mtime_iso(path),
        "sha256": sha256_file(path),
        "inferred_role": "UNKNOWN",
        "confidence": "LOW",
        "parse_status": "NOT_INSPECTED",
    }
    return result


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def inspect_file(path: Path) -> dict:
    name = path.name.lower()

    if is_emotiv_eeg_csv(path):
        return inspect_emotiv_eeg_csv(path)
    elif name.endswith("_intervalmarker.csv"):
        return inspect_interval_marker_csv(path)
    elif name.endswith(".json"):
        return inspect_json_metadata(path)
    elif name.endswith(".csv"):
        return inspect_supabase_csv(path)
    else:
        return inspect_generic(path)


# ---------------------------------------------------------------------------
# Directory walk
# ---------------------------------------------------------------------------

SCAN_EXTENSIONS = {
    ".csv", ".tsv", ".xlsx", ".xls", ".json",
    ".parquet", ".edf", ".bdf", ".fif", ".mat",
    ".wav", ".txt", ".log", ".sql", ".md",
    ".yaml", ".yml", ".xml", ".pdf",
    ".png", ".jpg", ".jpeg",
}


def walk_project(root: Path, skip_dirs: Optional[list] = None) -> list[dict]:
    """
    Recursively walk root, inspect every file with a recognised extension.
    Skips the results/ and src/ directories to avoid re-scanning outputs.
    """
    skip_dirs = skip_dirs or ["results", "src", "notebooks", "logs", "figures", "tables", ".git", "__pycache__"]
    records = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs in place
        dirnames[:] = [
            d for d in dirnames
            if d not in skip_dirs and not d.startswith(".")
        ]
        for fname in filenames:
            fpath = Path(dirpath) / fname
            ext = fpath.suffix.lower()
            # Also catch multi-part extensions like .md.bp.csv
            full_name_lower = fname.lower()
            is_emotiv = any(full_name_lower.endswith(s) for s in EMOTIV_CSV_SUFFIXES)
            if ext in SCAN_EXTENSIONS or is_emotiv:
                try:
                    rec = inspect_file(fpath)
                    records.append(rec)
                except Exception as e:
                    records.append({
                        "filepath": str(fpath),
                        "filename": fname,
                        "parse_status": f"FATAL: {e}",
                        "inferred_role": "UNKNOWN",
                        "confidence": "LOW",
                    })
    return records


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(str(v).strip())
    except Exception:
        return None


def _parse_int(v) -> Optional[int]:
    if v is None:
        return None
    try:
        return int(str(v).strip().split(".")[0])
    except Exception:
        return None
