import json

with open('CogProfile_Net_v2_Colab-1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

fixes = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    new_source = []
    for line in cell['source']:

        # Fix A: N_FREQ 64 → 32 in config line
        if 'N_FREQ=64' in line:
            line = line.replace('N_FREQ=64', 'N_FREQ=32')
            print(f'[Fix A] N_FREQ set to 32: {line.strip()}')
            fixes += 1

        # Fix B: scalogram compute batch 512 → 128
        if 'compute_all_scalograms(X_win, filters_fft, batch=512)' in line:
            line = line.replace('batch=512', 'batch=128')
            print(f'[Fix B] scalogram batch 512->128: {line.strip()}')
            fixes += 1
        if 'def compute_all_scalograms(X_win, filters_fft, batch=512)' in line:
            line = line.replace('batch=512', 'batch=128')
            print(f'[Fix B2] default batch 512->128 in function def')
            fixes += 1

        # Fix C: ScalogramCNN F=64 → F=32 in class default
        if 'def __init__(self, C=14, F=64, T=128' in line:
            line = line.replace('F=64', 'F=32')
            print(f'[Fix C] ScalogramCNN F default 64->32')
            fixes += 1

        # Fix D: model sanity check tensor F=64 → F=32
        if 'torch.randn(4,14,64,128)' in line:
            line = line.replace('torch.randn(4,14,64,128)', 'torch.randn(4,14,32,128)')
            print(f'[Fix D] sanity check tensor 64->32')
            fixes += 1
        if 'torch.zeros(1,C,F,T)' in line and 'F=32' not in ''.join(new_source[-5:]):
            # This computes flat size dynamically so no change needed
            pass

        new_source.append(line)
    cell['source'] = new_source

with open('CogProfile_Net_v2_Colab-1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'\nDone — {fixes} fix(es) applied.')
print('Memory impact:')
print('  GPU peak during CWT: ~7GB -> ~875MB (batch 128 x N_FREQ 32)')
print('  SCALOGRAMS RAM: ~2.6GB -> ~650MB')
