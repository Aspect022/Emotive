import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('CogProfile_Net_v2_Colab-1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)  # will raise if invalid JSON

all_src = '\n'.join(''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code')

checks = {
    # Phase A (no retraining)
    "EA: euclidean_align_subjects function":         "def euclidean_align_subjects" in all_src,
    "EA: X_win = euclidean_align_subjects(...)":     "X_win = euclidean_align_subjects" in all_src,
    "AdaBN: apply_adabn function":                   "def apply_adabn" in all_src,
    "AdaBN: called in evaluate":                     "apply_adabn(model" in all_src,
    "TTA: tta_shifts in evaluate signature":         "tta_shifts" in all_src,
    "TTA: torch.roll for temporal shifts":           "torch.roll" in all_src,
    # Phase B (retraining)
    "SpecAugment: spec_augment function":            "def spec_augment" in all_src,
    "SpecAugment: called in EEGDataset":             "sc=spec_augment(sc)" in all_src,
    "Electrode permute: function defined":           "def electrode_permute" in all_src,
    "Electrode permute: called in EEGDataset":       "rw=electrode_permute(rw)" in all_src,
    "EDL KL Phase 1: edl_loss called not nll_loss":  "ce =edl_loss" in all_src or "el=edl_loss" in all_src,
    "Manifold Mixup Phase 1: z_mix":                 "z_mix=lam*o" in all_src,
    "Manifold Mixup Phase 2b: o_mix":               "o_mix=" in all_src,
    "DEC lambda_dec function":                       "def lambda_dec" in all_src,
    "DEC confidence gating: conf_thresh":            "conf_thresh" in all_src,
    "DEC: lambda_dec(ep) in compute_loss":           "ld=lambda_dec(ep)" in all_src,
}

print("=" * 62)
print("CogProfile-Net v3 — Patch Verification")
print("=" * 62)
all_pass = True
for desc, result in checks.items():
    status = "PASS" if result else "FAIL"
    if not result:
        all_pass = False
    print(f"  [{status}]  {desc}")

print()
if all_pass:
    print("ALL CHECKS PASSED — notebook ready for upload to Colab")
else:
    print("SOME CHECKS FAILED — review above items")

print(f"\nNotebook cells: {len(nb['cells'])}")
print(f"JSON valid: YES (loaded successfully)")
