import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('CogProfile_Net_v2_Colab-1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def get_src(idx):
    return ''.join(nb['cells'][idx]['source'])

def set_src(idx, new_src):
    nb['cells'][idx]['source'] = [new_src]

# ── FIX 1: Remove unused 'sla' import from EA cell (cell 7) ──────────────────
c7 = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'euclidean_align_subjects' in ''.join(cell['source']):
        c7 = i; break

src7 = get_src(c7)
if 'import scipy.linalg as sla\n' in src7:
    src7 = src7.replace('import scipy.linalg as sla\n', '')
    set_src(c7, src7)
    print(f"FIX 1: Removed unused 'import scipy.linalg as sla' from cell {c7}")
else:
    print(f"FIX 1: 'sla' import not found in cell {c7} (may already be clean)")

# ── FIX 2: apply_adabn — restore BN momentum after stat collection ────────────
# This prevents BN momentum=None from bleeding into training after val evaluation
c15 = None
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'def apply_adabn' in ''.join(cell['source']):
        c15 = i; break

src15 = get_src(c15)

old_adabn = """def apply_adabn(model, loader, device):
    \"\"\"Adaptive Batch Normalization: reset BN running stats using test-subject data.
    Keeps learned γ and β frozen; only updates μ and σ² from test subject's windows.
    One forward pass is enough to collect accurate BN statistics.
    Expected gain: +1.5-2.5% (removes deep feature distribution shift per subject).
    \"\"\"
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.reset_running_stats()
            m.momentum = None   # cumulative moving average (all windows equally)
            m.training = True   # track stats
    with torch.no_grad():
        for sc, rw, bh, _ in loader:
            model(sc.to(device), rw.to(device), bh.to(device))
    model.eval()"""

new_adabn = """def apply_adabn(model, loader, device):
    \"\"\"Adaptive Batch Normalization: reset BN running stats using test-subject data.
    Keeps learned γ and β frozen; only updates μ and σ² from test subject's windows.
    Restores original BN momentum after stats collection to prevent training instability.
    Expected gain: +1.5-2.5% (removes deep feature distribution shift per subject).
    \"\"\"
    # Save original momentum values so training is unaffected after val evaluation
    saved_momentum = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            saved_momentum[name] = m.momentum
            m.reset_running_stats()
            m.momentum = None   # cumulative moving average: all windows contribute equally
            m.training = True   # enable stat tracking
    with torch.no_grad():
        for sc, rw, bh, _ in loader:
            model(sc.to(device), rw.to(device), bh.to(device))
    # Restore original momentum (default 0.1) — critical so training is not contaminated
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.momentum = saved_momentum[name]
    model.eval()"""

if old_adabn in src15:
    src15 = src15.replace(old_adabn, new_adabn)
    set_src(c15, src15)
    print(f"FIX 2: apply_adabn now restores BN momentum after stat collection (cell {c15})")
else:
    print(f"FIX 2: apply_adabn exact match failed — trying partial")
    if 'm.momentum = None   # cumulative moving average (all windows equally)' in src15:
        src15 = src15.replace(
            '    for m in model.modules():\n        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):\n            m.reset_running_stats()\n            m.momentum = None   # cumulative moving average (all windows equally)\n            m.training = True   # track stats\n    with torch.no_grad():\n        for sc, rw, bh, _ in loader:\n            model(sc.to(device), rw.to(device), bh.to(device))\n    model.eval()',
            '    saved_momentum = {}\n    for name, m in model.named_modules():\n        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):\n            saved_momentum[name] = m.momentum\n            m.reset_running_stats()\n            m.momentum = None\n            m.training = True\n    with torch.no_grad():\n        for sc, rw, bh, _ in loader:\n            model(sc.to(device), rw.to(device), bh.to(device))\n    for name, m in model.named_modules():\n        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):\n            m.momentum = saved_momentum[name]\n    model.eval()'
        )
        set_src(c15, src15)
        print(f"FIX 2: partial fix applied")

# Verify fixes
src7_final = get_src(c7)
src15_final = get_src(c15)
print()
print("VERIFY Fix 1 - sla gone:", 'import scipy.linalg as sla' not in src7_final)
print("VERIFY Fix 2 - saved_momentum present:", 'saved_momentum' in src15_final)
print("VERIFY Fix 2 - momentum restored:", 'm.momentum = saved_momentum' in src15_final)

with open('CogProfile_Net_v2_Colab-1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("\nNotebook saved with fixes applied.")
