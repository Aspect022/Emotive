import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('CogProfile_Net_v2_Colab-1.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

def get_src(idx):
    return ''.join(nb['cells'][idx]['source'])

def set_src(idx, new_src):
    nb['cells'][idx]['source'] = [new_src]

# Find EEGDataset cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'class EEGDataset' in ''.join(cell['source']):
        c13 = i
        break

src13 = get_src(c13)

# Fix: Add SpecAugment + electrode_permute AFTER augment_pair in EEGDataset
old_aug_block = """        if self.aug:
            (sc,rw),_=augment_pair(sc,rw)  # reuse augment, take only view 1
        return (torch.from_numpy(sc),torch.from_numpy(rw),
                torch.from_numpy(self.beh),torch.tensor(self.y[r],dtype=torch.long))"""

new_aug_block = """        if self.aug:
            (sc,rw),_=augment_pair(sc,rw)  # reuse augment, take only view 1
            sc=spec_augment(sc)             # SpecAugment: freq+time masking (+1-2%)
            rw=electrode_permute(rw)        # bilateral electrode swap (+0.5%)
        return (torch.from_numpy(sc),torch.from_numpy(rw),
                torch.from_numpy(self.beh),torch.tensor(self.y[r],dtype=torch.long))"""

if old_aug_block in src13:
    src13 = src13.replace(old_aug_block, new_aug_block)
    set_src(c13, src13)
    print("SUCCESS: EEGDataset SpecAugment + electrode_permute injected")
else:
    print("ERROR: exact match not found, trying partial")
    if "(sc,rw),_=augment_pair(sc,rw)" in src13:
        src13 = src13.replace(
            "(sc,rw),_=augment_pair(sc,rw)  # reuse augment, take only view 1",
            "(sc,rw),_=augment_pair(sc,rw)  # reuse augment, take only view 1\n            sc=spec_augment(sc)\n            rw=electrode_permute(rw)"
        )
        set_src(c13, src13)
        print("SUCCESS: partial match injection applied")

# Verify
if 'spec_augment(sc)' in get_src(c13):
    print("VERIFIED: spec_augment call found in EEGDataset cell")
else:
    print("FAILED: spec_augment still not in EEGDataset")

# Save
with open('CogProfile_Net_v2_Colab-1.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Notebook saved.")
