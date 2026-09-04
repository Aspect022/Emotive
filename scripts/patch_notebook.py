import json

with open('CogProfile_Net_v2_Colab-1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixes = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    new_source = []
    for line in cell['source']:
        # Revert STEP=64 back to STEP=32
        # STEP=64 caused data starvation (halved windows from ~40k to ~20k)
        # OOM is handled by del X_raw and num_workers=0 instead
        if 'SFREQ=128' in line and 'STEP=64' in line:
            line = line.replace('STEP=64', 'STEP=32')
            print(f'[Fix] STEP reverted 64->32: {line.strip()}')
            fixes += 1
        new_source.append(line)
    cell['source'] = new_source

with open('CogProfile_Net_v2_Colab-1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'\nDone — {fixes} fix(es) applied.')
print('Memory safety still maintained via: del X_raw + num_workers=0 + pin_memory=False')
