#!/usr/bin/env python3

#SBATCH --time=00:12:00
#SBATCH --partition=broadwl
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=all
#SBATCH --mail-user=letitiayhho@uchicago.edu
#SBATCH --output=logs/microstates_%j.log

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os.path as op
import argparse
import re
# EEG utilities
import mne
from pycrostates.cluster import ModKMeans
from pycrostates.preprocessing import extract_gfp_peaks
# BIDS utilities
from util.io.bids import DataSink
from bids import BIDSLayout

# constants
BIDS_ROOT = '../data/bids'
DERIV_ROOT = op.join(BIDS_ROOT, 'derivatives')
TASK = 'pitch'
N_MICROSTATES = 4

def main(sub):
    '''
    For one subject, Compiles epochs across all runs and computes
    subject-level microstate topographies. 

    Parameters
    -----------
    sub : str
        Subject ID as in BIDS dataset.
    '''

    # load epochs and concatenate across runs
    print("----------- load epochs and concatenate across runs -----------")
    layout = BIDSLayout(BIDS_ROOT, derivatives = True)
    run = lambda f: int(re.findall('run-(\w+)_', f)[0])
    fs = layout.get(
        return_type = 'filename',
        subject = sub, desc = 'forMicrostate'
        )
    fs.sort(key = run)
    epochs_all = [mne.read_epochs(f) for f in fs]
    epochs = mne.concatenate_epochs(epochs_all)
    epochs = epochs.pick('eeg')

    # cluster observed topographies to derive microstates
    print("----------- cluster observed topographics to derive microstates -----------")
    peaks = extract_gfp_peaks(epochs) # topographies w/ highest signal-to-noise
    ModK = ModKMeans(n_clusters = N_MICROSTATES, random_state = 0)
    ModK.fit(peaks, n_jobs = -1)

    fig_topos, axs = plt.subplots(1, N_MICROSTATES)
    ModK.plot(axes = axs)

    segmentation = ModK.predict(epochs, reject_edges = False)
    parameters = segmentation.compute_parameters()

    print("----------- Plot global explained variance (ratio) -----------")
    x = ModK.cluster_names
    y = [parameters[elt + "_gev"] for elt in x]
    fig_var, ax = plt.subplots()
    sns.barplot(x = x, y = y, ax = ax)
    ax.set_xlabel("Microstates")
    ax.set_ylabel("Global explained Variance (ratio)")

    print("----------- Plot time coverage (ratio) -----------")
    y = [parameters[elt + "_timecov"] for elt in x]
    fig_cov, ax = plt.subplots()
    ax = sns.barplot(x = x, y = y, ax = ax)
    ax.set_xlabel("Microstates")
    ax.set_ylabel("Time Coverage (ratio)")

    print("----------- Save microstates -----------")
    sink = DataSink(DERIV_ROOT, 'microstates')
    solution_fpath = sink.get_path(
        subject = sub,
        task = TASK,
        desc = 'microstates',
        suffix = 'ModKMeans',
        extension = '.fif.gz'
    )
    ModK.save(solution_fpath)

    print("----------- Generate report figures -----------")
    report = mne.Report(verbose = True)
    report.add_figure(
        fig_topos,
        title = 'Topographies',
        section = 'Microstates (subject-level)'
    )
    report.add_figure(
        fig_var,
        title = 'Variance explained',
        section = 'Microstates (subject-level)'
    )
    report.add_figure(
        fig_cov,
        title = 'Time coverage',
        section = 'Microstates (subject-level)'
    )
    
    print("----------- Save report -----------")
    report.save(op.join(sink.deriv_root, 'sub-%s_microstates.html'%sub), overwrite = True)

    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sub', type = str)
    args = parser.parse_args()
    main(args.sub)
