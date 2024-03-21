#!/usr/bin/env python3

#SBATCH --account=pi-hcn1
#SBATCH --time=02:00:00
#SBATCH --partition=bigmem
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=128G
#SBATCH --mail-type=all
#SBATCH --mail-user=letitiayhho@uchicago.edu
#SBATCH --output=logs/erp_%j.log

from matplotlib import pyplot as plt
from itertools import product
import pandas as pd
import os.path as op
import numpy as np
import mne
import re
from scipy import signal
from scipy.fft import fftshift
from bids import BIDSLayout

def read_epochs(sub, desc):
    '''
    reads and concatenates epochs across runs
    '''
    from bids import BIDSLayout
    layout = BIDSLayout(BIDS_ROOT, derivatives = True)
    run = lambda f: int(re.findall('run-(\w+)_', f)[0])
    fnames = layout.get(
        return_type = 'filename',
        subject = sub, 
        desc = desc
        )
    print(fnames)
    fnames.sort(key = run)
    epochs_all = [mne.read_epochs(f) for f in fnames]
    epochs = mne.concatenate_epochs(epochs_all)
    epochs = epochs.pick('eeg')
    return epochs

BIDS_ROOT = '../data/bids'
layout = BIDSLayout(BIDS_ROOT, derivatives = True, regex_search = 'forERP')
subs = layout.get_subjects(scope = 'erp')
subs.sort(key = int)
evokeds = pd.DataFrame()

for sub in subs:
    
    # Read epochs
    epochs = read_epochs(sub, 'forERP')
    
    # Compute evokeds 
    conditions = list(epochs.event_id.keys())
    chans = ['Cz', 'Fz', 'FCz', 'CPz', 'Pz']
    for c in conditions:
        for chan in chans:
            uV = np.squeeze(epochs[c].average(picks = chan).get_data())
            t = np.arange(-300, 300 + (1000/5000), 1000/5000)
            d = {
                'sub': sub,
                'chan': chan,
                'msec': t,
                'uV': uV
            }

            # Bind into dataframe
            evoked = pd.DataFrame(d)
            evokeds = pd.concat([evokeds, evoked])
            evokeds = evokeds.reset_index(drop = True)

evokeds.to_csv('evokeds.csv', index = False)