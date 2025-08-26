#!/usr/bin/env python3

#SBATCH --time=00:30:00
#SBATCH --partition=bigmem2
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=96G
#SBATCH --mail-type=all
#SBATCH --mail-user=letitiayhho@uchicago.edu
#SBATCH --output=logs/plot-grand-average-ffr_%j.log


from matplotlib import pyplot as plt
from itertools import product
import numpy as np
import pandas as pd
import os.path as op
import argparse
import re
import matplotlib.pyplot as plt
import mne
from scipy import signal
from scipy.fft import fftshift
from bids import BIDSLayout

fs = 1200

def read_epochs(sub, desc):
    '''
    reads and concatenates epochs across runs
    '''
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

def compute_spectrum_dB(epo):
    '''
    Computes power spectrum of frequency following response
    '''
    poststim = epo.average().compute_psd(tmin = 0., tmax = 0.25)
    baseline = epo.average().compute_psd(tmin = -0.25, tmax = 0.)
    power = 10 * np.log10(poststim.get_data() / baseline.get_data())
    power = np.squeeze(power)
    return poststim.freqs, power

def plot_power_spectrum(evokeds, stim_freq):
    poststim = evokeds.compute_psd(tmin = 0., tmax = 0.2)
    baseline = evokeds.compute_psd(tmin = -0.2, tmax = 0.)
    power = 10 * np.log10(poststim.get_data() / baseline.get_data())
    power = np.squeeze(power)
    freqs = poststim.freqs

    plt.plot(freqs, power, label = str(stim_freq))
    plt.xlabel('frequency')
    plt.ylabel('dB')
    plt.xlim(0, 300)
    plt.ylim(0, 40)
    plt.axvline(stim_freq, linestyle = '--', color = 'grey')

# Load data
BIDS_ROOT = '../data/bids'
layout = BIDSLayout(BIDS_ROOT, derivatives = True)
subs = layout.get_subjects(scope = 'preprocess_FFR')
subs.sort(key = int)
epochs = []

for sub in subs:
    if epochs == []:
        epochs = read_epochs(sub, 'forFFR')
    sub_epochs = read_epochs(sub, 'forFFR')
    epochs = mne.concatenate_epochs([epochs, sub_epochs])
    
# Compute grand average 
evokeds = {}
evokeds['130'] = epochs['11', '21', '31'].average()
evokeds['200'] = epochs['12', '22', '32'].average()
evokeds['280'] = epochs['31', '32', '33'].average()

plot_power_spectrum(evokeds['130'], 130)
plot_power_spectrum(evokeds['200'], 200)
plot_power_spectrum(evokeds['280'], 280)
plt.legend(title = 'Stim frequency (Hz)')
plt.savefig('grand-average-ffr.jpg'%stim_freq, dpi = 300, bbox_inches = 'tight')