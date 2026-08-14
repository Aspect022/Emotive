# File Forensics Report
Generated: 2026-08-14T15:59:30.122655+00:00

## Summary

| Category | Count |
|---|---|
| EEG data files (EMOTIV CSV) | 71 |
| Interval marker CSVs | 86 |
| JSON session metadata | 86 |
| Supabase activity results | 1 |
| Supabase participants | 16 |
| Sessions with EEG data | 71 |
| JSON-only entries (no EEG found) | 70 |

## EMOTIV EEG Files

### File Format (confirmed from inspection)
- **Multi-line header** CSV: row 0 = metadata key-value pairs; row 1 = column headers; rows 2+ = data samples.
- **Extension pattern**: `<name>_EPOCX_<serial>_<timestamp>.md.bp.csv` (Band Power exported) or `.md.csv`
- **Headset type**: All files inspected report `EPOCX` in both filename and metadata.

### EEG Column Schema (Example from: imagination_subjct9_EPOCX_792323_2026.05.01T00.46.31+05.30.md.csv)
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
- 128 Hz — 67 file(s)
- 256 Hz — 4 file(s)

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
- **Sessions**: 273 rows = test sessions
- **Date range**: 2026-02-14 20:29:36.81+00 → 2026-04-30 10:44:28.955+00

### `participants_rows (1).csv`
- **Columns**: `id, email, full_name, age, gender, consent_given, created_at, total_session_seconds`
- Participants identified by UUID (`id`) which matches `participant_id` in activity results.
- Contains demographic info: name, age, gender, consent flag, registration timestamp.

## Missing / Notable Gaps

1. **No EEG file found for several participants** — JSON metadata exists but `.md.bp.csv` / `.md.csv` is absent.
   Affected participants include: Aamir, Brijesh 1, Brijesh 2, Brijesh 3, Brijesh 4, Darshini, Dhrshan G, Kushal, Lakshmi, Maanya, Nishanth, Pragati, Prathap, Preetham, Rishab, Sanjay, Suhas, Uma H M, Venkat, aaditya, aishu, akhilesh, amey, bunny, dhanush H U, girish, imagination subjct9, imagination subject1, imagination subject10, imagination subject11, imagination subject3, imagination subject4, imagination subject7, imagination subject8, imagintion subject6, imaginztion subject13, immagination subject1, m1, manu, niranjan k, nitish, participant 1, participant 10, participant 11, participant 12, participant 13, participant 14, participant 15, participant 2, participant 3, participant 4, participant 5, participant 6, participant 7, participant 8, participant 9, pradeep kumar, prajesh, rahul jain, revanth reddy, rishi, rohith v, sankalp, santhosh, shaina, tharun karun, varun, yash raj, yashas, yashwanth
   
2. **No explicit software synchronization markers** linking EEG timestamps to web test events.
   The interval markers appear to be manually inserted by the experimenter.

3. **`timeTaken` in behavioral data is integer seconds** — resolution is 1 second, coarser than
   EEG sample period (~7.8 ms at 128 Hz). This limits precise RT alignment.

4. **No per-question stimulus onset timestamps** in the behavioral export — only session-level
   `started_at` and `completed_at` are available. Individual question onset times are NOT recorded.

5. **Behavioral data timestamps** are in UTC ISO format; EEG timestamps are Unix epoch floats
   in IST (+05:30). Clock synchronization must be verified.
