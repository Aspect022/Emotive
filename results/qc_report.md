# Initial QC & Research Readiness Report
Generated: 2026-08-14T15:59:30.131183+00:00

## Participant & Session Counts

| Metric | Value | Status |
|---|---|---|
| Participants in Supabase | 92370 | — |
| Behavioral sessions (activity_results) | 273 | — |
| EEG sessions (files with EEG data) | 71 | — |
| JSON-only entries (no EEG file found) | 70 | ⚠️ INCOMPLETE |
| Imagination task sessions (separate) | 11 | Separate sub-study |
| Main web-test EEG sessions | 45 | Primary dataset |

## EEG Recording Quality (Pre-Processing Assessment)

| Metric | Value | Notes |
|---|---|---|
| Headset confirmed | EPOC X | All files consistent |
| Electrode count | 14 | Standard EPOC X |
| Sampling rate (primary) | 128 Hz | Some files may be 256 Hz |
| Median duration | 12.0 min | |
| Range | 1.0–30.0 min | |
| Files with missing EEG | 70 participants | JSON present but CSV absent |
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
| G6: Participant count adequate | ⚠️ MARGINAL | ~71 sessions; exact participant count requires deduplication |
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
