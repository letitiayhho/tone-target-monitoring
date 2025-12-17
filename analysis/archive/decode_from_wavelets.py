#!/usr/bin/env python3

#SBATCH --time=00:08:00 # only need 15 minutes for regular logreg? need like 4 hrs for logregcv, 30 min for logreg no crop
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=32G # this doesn't seem to influence run time much
#SBATCH --mail-type=all
#SBATCH --mail-user=letitiayhho@uchicago.edu
#SBATCH --output=logs/decode_from_wavelets_%j.log

import gc
import sys
import mne
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from typing import Tuple, Iterator
from mne_bids import BIDSPath, read_raw_bids, print_dir_tree
from mne.time_frequency import tfr_morlet
from bids import BIDSLayout

from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import SlidingEstimator, cross_val_multiscore

def main(fpath, sub, task, run, cond, scores_fpath):
    BIDS_ROOT = '../data/bids'
    FIGS_ROOT = '../figs'
    STIM_FREQS = np.array([130, 200, 280])

    np.random.seed(0)

    print("---------- Load data ----------")
    print(fpath)
    epochs = mne.read_epochs(fpath)
    events = epochs.events
    
    print("---------- Subset epochs ----------")
    if cond == 'target':
        condition_epochs = epochs
    elif cond == 'tone_target':
        CONDS = ['11', '22', '33']
        condition_epochs = epochs[CONDS]
    elif cond == 'tone_nontarget':
        CONDS = ['12', '13', '21', '23', '31', '32']
        condition_epochs = epochs[CONDS]
    else:
        CONDS = {'1': ['11', '12', '13'], # subset the trials belonging to each target tone
                 '2': ['21', '22', '23'],
                 '3': ['31', '32', '33'],}
        condition_epochs = epochs[CONDS[cond[0]]]
    events = condition_epochs.events
    print(condition_epochs.event_id)
    print(condition_epochs)

    del epochs
    gc.collect()

    print("---------- Compute power ----------")
    n_cycles = STIM_FREQS / 7 # different number of cycle per frequency
                               # higher constant, fewer windows, maybe?
    power = tfr_morlet(condition_epochs,
                       freqs = STIM_FREQS,
                       n_cycles = n_cycles,
                       use_fft = True,
                       return_itc = False,
                       decim = 3,
                       n_jobs = 1,
                       average = False)
    power = np.log10(power)
    
    del condition_epochs
    gc.collect()

    # Get some information
    n_epochs = np.shape(power)[0]
    n_channels = np.shape(power)[1]
    n_freqs = np.shape(power)[2]
    n_windows = np.shape(power)[3]
    print("n_windows: " + str(n_windows))

    print("---------- Prepare for decoder ----------")
    # Reshape for classifier
    X = power.reshape((n_epochs, n_freqs * n_channels, n_windows)) # Set order to preserve epoch order

    # Create array of condition labels
    labels = pd.Series(events[:, 2])
    EVENT_DICTS = {'tone_target': {10001 : 1, 10005: 2, 10009: 3},
                   'tone_nontarget': {10002 : 1, 10003 : 1, 10004: 2, 10006: 2, 10007: 3, 10008: 3},
                   '11': {10001: 1, 10002: 0, 10003: 0},
                   '12': {10001: 0, 10002: 1, 10003: 0},
                   '13': {10001: 0, 10002: 0, 10003: 1},
                   '21': {10004: 1, 10005: 0, 10006: 0},
                   '22': {10004: 0, 10005: 1, 10006: 0},
                   '23': {10004: 0, 10005: 0, 10006: 1},
                   '31': {10007: 1, 10008: 0, 10009: 0},
                   '32': {10007: 0, 10008: 1, 10009: 0},
                   '33': {10007: 0, 10008: 0, 10009: 1},
                   'target': {10001 : 1, 10002 : 0, 10003 : 0, 10004: 0, 10005: 1, 10006: 0, 10007: 0, 10008: 0, 10009: 1}}
                    # FOR REFERENCE {'11': 10001, '12': 10002, '13': 10003, '21': 10004, 
                    #'22': 10005, '23': 10006, '31': 10007, '32': 10008, '33': 10009}
    y = labels.replace(EVENT_DICTS[cond])
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    print("---------- Decode ----------")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver = 'liblinear')
    )

    print("Creating sliding estimators")
    time_decod = SlidingEstimator(clf,
                                 scoring = 'balanced_accuracy')

    print("Fit estimators")
    scores = cross_val_multiscore(
        time_decod,
        X, # a trials x features x time array
        y, # an (n_trials,) array of integer condition labels
        cv = 5, # use stratified 5-fold cross-validation
        n_jobs = -1, # use all available CPU cores
    )
    scores = np.mean(scores, axis = 0) # average across cv splits

    print("---------- Save decoder scores ----------")
    print('Saving scores to: ' + scores_fpath)
    np.save(scores_fpath, scores)

    print("---------- Plot ----------")
    n_stimuli = 3 
    fig, ax = plt.subplots()
    ax.plot(range(len(scores)), scores, label = 'score')
    ax.axhline(1/n_stimuli, color = 'k', linestyle = '--', label = 'chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.set_title('Sensor space decoding')

    # Save plot
    fig_fpath = FIGS_ROOT + '/sub-' + sub + '_run-' + run + '_wavelet_' + cond + '.png'
    print('Saving figure to: ' + fig_fpath)
    plt.savefig(fig_fpath)

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print(__doc__)
        sys.exit(1)
    fpath = sys.argv[1]
    sub = sys.argv[2]
    task = sys.argv[3]
    run = sys.argv[4]
    cond = sys.argv[5]
    scores_fpath = sys.argv[6]
    main(fpath, sub, task, run, cond, scores_fpath)
