#!/usr/bin/env python3

#SBATCH --time=00:20:00
#SBATCH --partition=broadwl
#SBATCH --mem-per-cpu=56GB
#SBATCH --output=logs/grand-average-ffr_%j.log

import mne
import glob
import numpy as np
import pandas as pd

files = glob.glob('../data/bids/derivatives/preprocessing/sub-*/*fif.gz')
first = True
conditions = ['11', '12', '13', '21', '22', '23', '31', '32', '33']
all_evokeds = {}

for file in files:
    epochs = mne.read_epochs(file)

    if first: # handle first loop
        conditions = list(epochs.event_id.keys())
        all_evokeds = {c:[epochs[c].average()] for c in conditions}
        first = False
        continue

    if epochs.info['nchan'] != 62: # handle files with incorrect number of channels
        print(f"Skipping {file}, incorrect number of channels.")
        continue

    evokeds = {c:[epochs[c].average()] for c in conditions}
    for condition in conditions:
        print(f"Concatenating evokeds from condition {condition}")
        all_evokeds[condition].append(evokeds[condition][0])
        break
    break

for condition in conditions:
    combined_evokeds = mne.combine_evoked(all_evokeds[condition], "equal")
    
    print(f"Saving to evokeds_{condition}.fif.gz")
    combined_evokeds.save(f'../data/bids/derivatives/grand-average-ffr/evokeds_{condition}.fif.gz', overwrite = True)

