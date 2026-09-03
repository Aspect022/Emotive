import json, gc

with open('CogProfile_Net_v2_Colab-1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixes = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue

    src = cell['source']
    new_src = []
    i = 0
    while i < len(src):
        line = src[i]

        # Fix 3: STEP=32 -> STEP=64 (halves window count, halves SCALOGRAMS RAM)
        if 'SFREQ=128' in line and 'STEP=32' in line:
            line = line.replace('STEP=32', 'STEP=64')
            print('[Fix 3] STEP changed 32->64 (halves SCALOGRAMS size)')
            fixes += 1

        # Fix 4: After make_subwindows call, insert del X_raw + gc.collect
        if 'X_win,y_win,trial_ids,subj_ids = make_subwindows' in line:
            new_src.append(line)
            new_src.append('del X_raw, y, pids  # free ~83MB — no longer needed after windowing\n')
            new_src.append('gc.collect()\n')
            new_src.append('print(f"[Memory freed] X_raw deleted. X_win size: {X_win.nbytes/1e6:.1f} MB")\n')
            print('[Fix 4] del X_raw inserted after make_subwindows')
            fixes += 1
            i += 1
            continue

        new_src.append(line)
        i += 1

    cell['source'] = new_src

# Fix 5: Add import gc to the imports cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        joined = ''.join(cell['source'])
        if 'import os, json, time' in joined and 'import gc' not in joined:
            cell['source'] = ['import gc\n'] + cell['source']
            print('[Fix 5] import gc added to imports cell')
            fixes += 1
            break

with open('CogProfile_Net_v2_Colab-1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'\nDone — {fixes} fix(es) applied.')
