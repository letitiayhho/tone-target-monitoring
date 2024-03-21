#!/usr/bin/env python3

#SBATCH --time=00:30:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=all
#SBATCH --mail-user=letitiayhho@uchicago.edu
#SBATCH --output=logs/plot-aux-psd_%j.log

from mne_bids import BIDSPath, write_raw_bids, get_anonymization_daysback
import random
import numpy as np
import itertools
import mne
import os
import sys
import re
from bids import BIDSLayout
from util.io.iter_raw_paths import iter_raw_paths

def plot_aux(tone_freq, epochs, sub, FIGS_DIR, aux):
    print(f"---------- Tag {tone_freq}: Audio: {aux} ----------")
    plt = epochs.plot_psd(picks = aux, fmin = 100, fmax = 300) # thought this was left
    figname = f'{FIGS_DIR}/sub-{sub}_tone-{tone_freq}.png'
    print(f"saving to {figname}")
    plt.savefig(figname)
    
RAW_DIR = '../data/raw/'
FIGS_DIR = '../figs'

def main() -> None:
    for (fname, sub, task, run) in iter_raw_paths(RAW_DIR):
        raw = mne.io.read_raw_brainvision(RAW_DIR + fname)
        raw.load_data()
        events, event_ids = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, tmin = -0.2, tmax = .3, baseline = (-0.2, 0))
        del raw
        plot_aux('130', epochs['11', '21', '31'], sub, FIGS_DIR, 'Aux1')
        plot_aux('200', epochs['12', '22', '32'], sub, FIGS_DIR, 'Aux1')
        plot_aux('280', epochs['13', '23', '33'], sub, FIGS_DIR, 'Aux1')
        
if __name__ == "__main__":
    main()