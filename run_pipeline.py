"""
Pipeline Orchestrator — Phases 2 through 5
Runs the full EEG → DL dataset pipeline in sequence:

  Phase 2A: Participant Linkage
  Phase 2B: Behavioral Trial Expansion
  Phase 3:  Session Synchronization
  Phase 4:  EEG Preprocessing
  Phase 5:  Windowing & Dataset Compilation

Usage:
  python run_pipeline.py                  # all phases
  python run_pipeline.py --phase 2a       # single phase
  python run_pipeline.py --skip-preproc   # skip Phase 4 (already done)
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def header(msg: str):
    print("\n" + "=" * 60)
    print(msg)
    print("=" * 60)


def run_phase_2a():
    header("PHASE 2A: PARTICIPANT LINKAGE")
    from linkage.participant_linker import (
        load_inventory, load_participants, link_participants,
        write_linkage_csv, write_linkage_report,
        INVENTORY_CSV, PARTICIPANTS_CSV, OUT_LINKAGE, OUT_LINKAGE_REPORT
    )
    if not INVENTORY_CSV.exists():
        print("ERROR: file_inventory.csv not found. Run run_phase1_inspection.py first.")
        return False
    if not PARTICIPANTS_CSV:
        print("ERROR: participants_rows.csv not found.")
        return False

    inventory = load_inventory(INVENTORY_CSV)
    participants = load_participants(PARTICIPANTS_CSV)
    print(f"  Inventory: {len(inventory)} files")
    print(f"  Participants (Supabase): {len(participants)}")

    records = link_participants(inventory, participants)
    write_linkage_csv(records, OUT_LINKAGE)
    write_linkage_report(records, OUT_LINKAGE_REPORT)

    main = [r for r in records if r["sub_study"] in ("main_numbered", "main_named")]
    matched = [r for r in main if r["matched_uuid"]]
    print(f"\n  Main study: {len(main)} EEG sessions")
    print(f"  Matched to Supabase UUID: {len(matched)}")
    print(f"  Unmatched: {len(main) - len(matched)}")
    return True


def run_phase_2b():
    header("PHASE 2B: BEHAVIORAL TRIAL EXPANSION")
    from behavioral.trial_expander import (
        load_activity_results, expand_session, compute_session_enriched,
        DATA_DIR, OUT_DIR
    )
    import csv, json
    from collections import Counter

    sessions = load_activity_results(DATA_DIR)
    print(f"  Sessions: {len(sessions)}")

    all_trials = []
    for sess in sessions:
        all_trials.extend(expand_session(sess))

    print(f"  Trials expanded: {len(all_trials)}")

    enriched = compute_session_enriched(sessions)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    trials_path = OUT_DIR / "trials_flat.csv"
    if all_trials:
        with open(trials_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_trials[0].keys()))
            writer.writeheader()
            writer.writerows(all_trials)
        print(f"  [OK] trials_flat.csv: {trials_path}")

    sessions_path = OUT_DIR / "sessions_enriched.csv"
    if enriched:
        with open(sessions_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(enriched[0].keys()))
            writer.writeheader()
            writer.writerows(enriched)
        print(f"  [OK] sessions_enriched.csv: {sessions_path}")

    act_dist = Counter(t["activity_type"] for t in all_trials)
    print("\n  Trials by activity:")
    for act, cnt in sorted(act_dist.items()):
        print(f"    {act}: {cnt}")
    return True


def run_phase_3():
    header("PHASE 3: SESSION SYNCHRONIZATION")
    from sync.session_synchronizer import (
        load_csv, get_eeg_metadata, build_uuid_to_eeg,
        sync_session, write_sync_report,
        INVENTORY_CSV, LINKAGE_CSV, SESSIONS_CSV,
        OUT_MANIFEST, OUT_REPORT, RESULTS_DIR
    )
    import csv
    from collections import Counter

    for path, name in [(INVENTORY_CSV, "file_inventory.csv"),
                       (LINKAGE_CSV, "participant_linkage.csv"),
                       (SESSIONS_CSV, "sessions_enriched.csv")]:
        if not path.exists():
            print(f"ERROR: {name} not found. Run previous phases first.")
            return False

    inventory = load_csv(INVENTORY_CSV)
    linkage = load_csv(LINKAGE_CSV)
    sessions = load_csv(SESSIONS_CSV)

    eeg_meta = get_eeg_metadata(inventory)
    uuid_to_eeg = build_uuid_to_eeg(linkage, eeg_meta)

    print(f"  Sessions: {len(sessions)}")
    print(f"  EEG files with timestamps: {len(eeg_meta)}")
    print(f"  Participants with linked EEG: {len(uuid_to_eeg)}")

    manifest = []
    for sess in sessions:
        pid = sess.get("participant_id", "")
        eeg_files = uuid_to_eeg.get(pid, [])
        manifest.append(sync_session(sess, eeg_files))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if manifest:
        with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
            writer.writeheader()
            writer.writerows(manifest)
        print(f"  [OK] sync_manifest.csv: {OUT_MANIFEST}")

    write_sync_report(manifest, OUT_REPORT)

    status_counts = Counter(r["sync_status"] for r in manifest)
    print("\n  Sync results:")
    for status, cnt in sorted(status_counts.items()):
        print(f"    {status}: {cnt}")

    dl_usable = [
        r for r in manifest
        if r["sync_confidence"] in ("HIGH", "MODERATE")
        and float(r.get("time_taken_s", 0) or 0) >= 60
    ]
    est_windows = sum(int(float(r["time_taken_s"] or 0) // 4) for r in dl_usable)
    print(f"\n  DL-usable sessions: {len(dl_usable)}")
    print(f"  Estimated 4s windows: {est_windows}")
    return True


def run_phase_4(verbose: bool = False):
    header("PHASE 4: EEG PREPROCESSING")
    from preprocessing.eeg_preprocessor import (
        preprocess_all, write_qc_csv, ROOT
    )
    import numpy as np

    DATA_DIR = ROOT / "Data"
    OUT_DIR = ROOT / "results"
    PREPROCESSED_DIR = ROOT / "data" / "processed" / "preprocessed"
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    eeg_files = []
    for pattern in ["*.md.bp.csv", "*.md.pm.bp.csv", "*.md.csv"]:
        eeg_files.extend(DATA_DIR.rglob(pattern))
    eeg_files = [f for f in eeg_files
                 if "intervalMarker" not in f.name
                 and "participants_rows" not in f.name]

    print(f"  EEG files to process: {len(eeg_files)}")

    results = preprocess_all(eeg_files, verbose=verbose)
    write_qc_csv(results, OUT_DIR / "preprocessing_qc.csv")

    print(f"\n  Saving numpy arrays...")
    for fp_str, res in results.items():
        stem = Path(fp_str).stem.replace(".", "_")
        out_file = PREPROCESSED_DIR / f"{stem}.npz"
        np.savez_compressed(
            out_file,
            timestamps=res["timestamps"],
            eeg_preprocessed=res["eeg_preprocessed"],
            eeg_raw=res["eeg_raw"],
            good_channels=res["good_channels"],
            interpolated_mask=res["interpolated_mask"],
            amplitude_artifact_mask=res["amplitude_artifact_mask"],
        )
    print(f"  Saved {len(results)} preprocessed files to {PREPROCESSED_DIR}")
    return True


def run_phase_5():
    header("PHASE 5+7: WINDOWING & DL DATASET COMPILATION")
    from epoching.windower import (
        load_csv, build_dataset, subject_independent_split,
        RESULTS_DIR, PREPROCESSED_DIR, OUT_DATASET_DIR,
        TASK_LABEL_MAP, EEG_CHANNELS, WINDOW_SIZE_S, NORMALIZE
    )
    import json, csv
    import numpy as np
    from collections import Counter

    SYNC_MANIFEST = RESULTS_DIR / "sync_manifest.csv"
    if not SYNC_MANIFEST.exists():
        print("ERROR: sync_manifest.csv not found. Run Phase 3 first.")
        return False

    sync_manifest = load_csv(SYNC_MANIFEST)
    dataset = build_dataset(sync_manifest, PREPROCESSED_DIR)

    if not dataset or "X" not in dataset:
        print("  Dataset empty — run Phase 4 first, then re-run Phase 5.")
        return False

    OUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    X = dataset["X"]
    y_task = dataset["y_task"]
    participant_ids = dataset["participant_ids"]
    session_ids = dataset["session_ids"]
    meta = dataset["meta"]

    np.save(OUT_DATASET_DIR / "X_raw.npy", X)
    np.save(OUT_DATASET_DIR / "y_task.npy", y_task)
    np.save(OUT_DATASET_DIR / "participant_ids.npy", participant_ids)
    np.save(OUT_DATASET_DIR / "session_ids.npy", session_ids)

    splits = subject_independent_split(participant_ids)
    with open(OUT_DATASET_DIR / "splits.json", "w") as f:
        json.dump(splits, f, indent=2)

    label_map = {
        "task": TASK_LABEL_MAP,
        "task_inv": {str(v): k for k, v in TASK_LABEL_MAP.items()},
        "window_size_s": WINDOW_SIZE_S,
        "n_channels": 14,
        "channel_names": EEG_CHANNELS,
        "normalization": NORMALIZE,
    }
    with open(OUT_DATASET_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    if meta:
        with open(OUT_DATASET_DIR / "window_meta.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
            writer.writeheader()
            writer.writerows(meta)

    inv_map = {v: k for k, v in TASK_LABEL_MAP.items()}
    task_dist = Counter(y_task.tolist())
    print(f"\n  X shape: {X.shape}  (windows x channels x timepoints)")
    print(f"  Task distribution:")
    for lbl, cnt in sorted(task_dist.items()):
        print(f"    {inv_map.get(lbl, lbl)}: {cnt} ({100*cnt/len(y_task):.1f}%)")
    print(f"\n  Files saved to: {OUT_DATASET_DIR}")
    print(f"  Dataset ready for DL training.")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Research Pipeline")
    parser.add_argument("--phase", choices=["2a", "2b", "3", "4", "5", "all"],
                        default="all", help="Which phase to run")
    parser.add_argument("--skip-preproc", action="store_true",
                        help="Skip Phase 4 (EEG preprocessing) if already done")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output during preprocessing")
    args = parser.parse_args()

    t0 = time.time()
    success = True

    if args.phase in ("2a", "all"):
        ok = run_phase_2a()
        if not ok and args.phase == "all":
            print("\nPhase 2A failed. Aborting pipeline.")
            sys.exit(1)

    if args.phase in ("2b", "all"):
        ok = run_phase_2b()

    if args.phase in ("3", "all"):
        ok = run_phase_3()

    if args.phase in ("4", "all") and not args.skip_preproc:
        ok = run_phase_4(verbose=args.verbose)

    if args.phase in ("5", "all"):
        ok = run_phase_5()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Pipeline completed in {elapsed:.1f}s")
    print("=" * 60)
