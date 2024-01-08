#!/usr/bin/env python3

#SBATCH --time=00:10:00
#SBATCH --partition=broadwl
#SBATCH --mem-per-cpu=32GB
#SBATCH --output=logs/grand-average-ffr_%j.log

import mne
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.fft import fftshift

files = glob.glob('../data/bids/derivatives/preprocessing/sub-*/*fif.gz')
epochs = []
conditions = ['11', '12', '13', '21', '22', '23', '31', '32', '33']
i = 0

#for condition in conditions:
for file in files:
    print(i)
    sub_epochs = mne.read_epochs(file)

    if not epochs: # handle first loop
        epochs = sub_epochs
        continue

    if sub_epochs.info['nchan'] != 62: # handle files with incorrect number of channels
        print(f"Skipping {file}, incorrect number of channels.")
        continue

    #sub_epochs = sub_epochs[condition]
    epochs = mne.concatenate_epochs([epochs, sub_epochs])

    # Compute grand average 
    #print("Compute grand average")
    #conditions = list(epochs.event_id.keys())
    #evokeds = {c:epochs[c].average() for c in conditions}
    #evokeds = epochs[condition].average()

    # Save
    #print(f"Saving to evokeds_{condition}.csv")
    #evokeds = pd.concat(evokeds)
    #evokeds.to_csv(f'evokeds_{condition}.csv', sep = '\t', index = False)
    i += 1
    if i > 3:
        break

conditions = list(epochs.event_id.keys())
for condition in conditions:
    print("Compute grand average for condition {condition}")
    evokeds = epochs[condition].average()

    print(f"Saving to evokeds_{condition}.csv")
    evokeds.to_csv(f'evokeds_{condition}.csv', sep = '\t', index = False)

