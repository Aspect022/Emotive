# Synchronization Feasibility Report
Generated: 2026-08-14T15:59:30.129567+00:00

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
