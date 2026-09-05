import json

with open('CogProfile_Net_v2_Colab-1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixes = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    new_source = []
    for line in cell['source']:

        # Revert N_FREQ 32 → 64 (this was cutting model capacity in half)
        if 'N_FREQ=32' in line and 'SFREQ' in line:
            line = line.replace('N_FREQ=32', 'N_FREQ=64')
            print(f'[Revert A] N_FREQ restored to 64: {line.strip()}')
            fixes += 1

        # Revert ScalogramCNN F=32 → F=64
        if 'def __init__(self, C=14, F=32, T=128' in line:
            line = line.replace('F=32', 'F=64')
            print(f'[Revert C] ScalogramCNN F restored to 64')
            fixes += 1

        # Revert sanity check tensor F=32 → F=64
        if 'torch.randn(4,14,32,128)' in line:
            line = line.replace('torch.randn(4,14,32,128)', 'torch.randn(4,14,64,128)')
            print(f'[Revert D] sanity check tensor restored to 64')
            fixes += 1

        # KEEP batch=128 in compute_all_scalograms (this is the correct GPU fix)
        # Do NOT revert this — it solves the 7GB GPU tensor without hurting accuracy

        new_source.append(line)
    cell['source'] = new_source

with open('CogProfile_Net_v2_Colab-1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'\nDone — {fixes} fix(es) applied.')
print('batch=128 in compute_all_scalograms is KEPT (correct GPU fix).')
print('N_FREQ=64 restored (model capacity back to 1.27M params).')
