#!/usr/bin/env python3

#SBATCH --time=00:04:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-type=all
#SBATCH --mail-user=letitiayhho@uchicago.edu
#SBATCH --output=logs/compute_stft_%j.log

import sys
import numpy as np
import mne
import pandas as pd
from scipy import signal
from scipy import signal
from util.io.stft import *

def main(fpath, sub, task, run, save_fpath):
    FS = 5000
    CONDITION_FREQS = [130, 200, 280]
    
    # Read data
    epochs = mne.read_epochs(fpath)
    epochs = epochs.get_data()
    
    # Get metadata
    n_freqs = len(CONDITION_FREQS)
    n_epochs = np.shape(epochs)[0]
    n_chans = np.shape(epochs)[1]
    
    # Compute stft across all channels
    Zxxs = np.empty([n_epochs, n_chans, n_freqs, 19]) # n_epochs, n_chans, n_freqs, n_windows
    for chan in range(n_chans):
        x = pd.DataFrame(epochs[:, chan, :])
        f, t, Zxx = get_stft_for_one_channel(x, FS, n_epochs, CONDITION_FREQS)
        Zxxs[:, chan, :, :] = Zxx

    # Save powers and events
    print('Saving scores to: ' + save_fpath)
    np.save(save_fpath, Zxxs)
        
    return (Zxxs)

__doc__ = "Usage: ./compute_stft.py <fpath> <sub> <task> <run> <save_fpath>"

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print(__doc__)
        sys.exit(1)
    FPATH = sys.argv[1]
    SUB = sys.argv[2]
    TASK = sys.argv[3]
    RUN = sys.argv[4]
    SAVE_FPATH = sys.argv[5]
    main(FPATH, SUB, TASK, RUN, SAVE_FPATH)
