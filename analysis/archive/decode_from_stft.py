#!/usr/bin/env python3

#SBATCH --time=00:10:00
#SBATCH --partition=broadwl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=24G
#SBATCH --mail-type=all
#SBATCH --mail-user=letitiayhho@uchicago.edu
#SBATCH --output=logs/decode_from_stft_%j.log

import mne
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from mne.decoding import SlidingEstimator, cross_val_multiscore

from util.io.bids import DataSink

def main(sub, task, run):
    BIDS_ROOT = '../data/bids'
    DERIV_ROOT = '../data/bids/derivatives'
    FIGS_ROOT = '../figs'

    # Load events
    epochs_fpath = f'/project2/hcn1/pitch_tracking_attention/scripts/../data/bids/derivatives/preprocessing/sub-{sub}/sub-{sub}_task-pitch_run-{run}_desc-clean_epo.fif.gz'
    print(f'Loading events from {epochs_fpath}')
    epochs = mne.read_epochs(epochs_fpath)
    events = epochs.events

    # Load stft results        
    DERIV_ROOT = '../data/bids/derivatives'
    sink = DataSink(DERIV_ROOT, 'decoding')
    stft_fpath = sink.get_path(
        subject = sub,
        task = task,
        run = run,
        desc = 'stft',
        suffix = 'power',
        extension = 'npy',
    )
    print(f'Loading stft from {stft_fpath}')
    Zxxs = np.load(stft_fpath)

    # Reshape for decoder
    n_epochs = np.shape(Zxxs)[0]
    if n_epochs != np.shape(events)[0]:
        sys.exit('Incorrect number of epochs')
    n_freqs = 3
    n_chans = 62
    n_windows = 19 ## CHANGE THIS
    Zxxs = Zxxs.reshape((n_epochs, n_freqs*n_chans, n_windows)) # n_epochs, n_freqs*n_chans, n_windows

    # Create target array
    labels = pd.Series(events[:, 2])
    y = labels.replace({10001 : 0, 10002 : 1, 10003 : 2, 10004 : 3, 10005 : 4}) # CHANGE THIS
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(y)

    # Decode
    print("---------- Decode ----------")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(solver = 'liblinear')
    )

    print("Creating sliding estimators")
    time_decod = SlidingEstimator(clf)

    print("Fit estimators")
    scores = cross_val_multiscore(
        time_decod,
        Zxxs, # a trials x features x time array
        y, # an (n_trials,) array of integer condition labels
        cv = 5, # use stratified 5-fold cross-validation
        n_jobs = -1, # use all available CPU cores
    )
    scores = np.mean(scores, axis = 0) # average across cv splits

    # Save decoder score_shape
    print("---------- Save decoder scores ----------")
    sink = DataSink(DERIV_ROOT, 'decoding')
    scores_fpath = sink.get_path(
        subject = sub,
        task = task,
        run = run,
        desc = 'stft',
        suffix = 'scores',
        extension = 'npy',
    )
    print('Saving scores to: ' + scores_fpath)
    np.save(scores_fpath, scores)

    # Plot
    print("---------- Plot ----------")
    fig, ax = plt.subplots()
    ax.plot(range(len(scores)), scores, label = 'score')
    ax.axhline(1/n_stimuli, color = 'k', linestyle = '--', label = 'chance')
    ax.set_xlabel('Times')
    ax.set_ylabel('Accuracy')  # Area Under the Curve
    ax.legend()
    ax.set_title('Sensor space decoding')

    # Save plot
    fig_fpath = FIGS_ROOT + '/sub-' + sub + '_run-' + run + '_stft' + '.png'
    print('Saving figure to: ' + fig_fpath)
    plt.savefig(fig_fpath)

__doc__ = "Usage: ./decode_from_stft.py <sub> <task> <run>"

if __name__ == "__main__":
    print(len(sys.argv))
    print("Argument List:", str(sys.argv))
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit("Incorrect call to script")
    print("Reading args")
    SUB = sys.argv[1]
    TASK = sys.argv[2]
    RUN = sys.argv[3]
    main(SUB, TASK, RUN)
