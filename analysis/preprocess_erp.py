#!/usr/bin/env python3

#SBATCH --account=pi-hcn1
#SBATCH --time=03:00:00 # 2 hrs enough for almost all
#SBATCH --partition=bigmem
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=300G # 300G enough for most
#SBATCH --mail-type=all
#SBATCH --mail-user=letitiayhho@uchicago.edu
#SBATCH --output=logs/preprocess_erp-%j.log

import numpy as np
import os.path as op
import os
import sys

cwd = os.getcwd()
sys.path.append(cwd)
print(sys.path)

from pprint import pformat
import argparse
# EEG utilities
import mne
from mne.preprocessing import ICA, create_eog_epochs
from pyprep.prep_pipeline import PrepPipeline
from autoreject import get_rejection_threshold, validation_curve
# BIDS utilities
from mne_bids import BIDSPath, read_raw_bids
from util.io.bids import DataSink

# constants
BIDS_ROOT = '../data/bids'
DERIV_ROOT = op.join(BIDS_ROOT, 'derivatives')
ERP_PASSBAND = (0.1, 40)
TASK = 'pitch'
TMIN = -0.3
TMAX = 0.3

def main(sub, run):
    '''
    Parameters
    ----------
    sub : str
        Subject ID as in BIDS dataset
    '''
    print('----------------- load data ------------------')
    bids_path = BIDSPath(
        root = BIDS_ROOT,
        subject = sub,
        task = TASK,
        run = run,
        datatype = 'eeg'
        )
    print(bids_path)
    raw = read_raw_bids(bids_path, verbose = False)
    events, event_ids = mne.events_from_annotations(raw)

    print('----------------- re-reference eye electrodes to become bipolar EOG ------------------')
    raw.load_data()
    def reref(dat):
        dat[0,:] = (dat[1,:] - dat[0,:])
        return dat
    raw = raw.apply_function(
        reref,
        picks = ['leog', 'Fp2'],
        channel_wise = False
    )
    raw = raw.apply_function(
        reref,
        picks = ['reog', 'Fp1'],
        channel_wise = False
    )
    raw = raw.set_channel_types({'leog': 'eog', 'reog': 'eog'})

    print('----------------- run PREP pipeline ------------------') # notch, exclude bad chans, and re-reference
    raw.load_data()
    np.random.seed(int(sub))
    lf = raw.info['line_freq']
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(lf, ERP_PASSBAND[1], lf)
    }
    prep = PrepPipeline(
        raw,
        prep_params,
        raw.get_montage(),
        ransac = False,
        random_state = int(sub)
        )
    prep.fit()

    print('----------------- Extract data from PREP ------------------')
    prep_eeg = prep.raw_eeg # get EEG channels from PREP
    prep_non_eeg = prep.raw_non_eeg # get non-EEG channels from PREP
    raw_data = np.concatenate((prep_eeg.get_data(), prep_non_eeg.get_data())) # combine data from the two
    
    # Create info object for post-PREP data
    print('Create info object for post-PREP data')
    new_ch_names = prep_eeg.info['ch_names'] + prep_non_eeg.info['ch_names']
    raw = raw.reorder_channels(new_ch_names) # modify the channel names on the original raw data
    raw_info = raw.info # use the modified info from the original raw data object
     
    # Combine post-prep data and new info
    print('Create new raw object')
    raw = mne.io.RawArray(raw_data, raw_info) # replace original raw object
    
    print('----------------- Filter ------------------') 
    raw = raw.filter(*ERP_PASSBAND)

    ## now prepare non-epoched data for ERP analysis
    # identify bad ICs on weakly highpassed data
    print('----------------- Epoch data for ERP analysis ------------------')
    epochs = mne.Epochs(
        raw,
        events, # same events as FFR epochs
        tmin = TMIN,
        tmax = TMAX, # only prestim
        event_id = event_ids,
        baseline = None,
        preload = True
    )

    print('----------------- Downsample ------------------') 
    epochs = epochs.resample(1000) # resample after epoching to avoid adding jitter to event triggers

    print('----------------- Run ICA ------------------')
    ica = ICA(n_components = 15, random_state = 0)
    ica.fit(epochs, picks = ['eeg', 'eog'])
    
    print('----------------- Apply ICA ------------------')
    eog_indices, eog_scores = ica.find_bads_eog(epochs, threshold = 1.96)
    ica.exclude = eog_indices
    ica.apply(epochs) # transforms in place 

    if ica.exclude: # if we found any bad components
        fig_ica_removed = ica.plot_components(ica.exclude)

    # now we no longer need EOG channels
    epochs = epochs.drop_channels('leog')
    epochs = epochs.drop_channels('reog')

    print('----------------- Baseline correct ------------------')
    epochs = epochs.apply_baseline((TMIN, 0.))

    print('----------------- Reject bad trials ------------------')
    thres = get_rejection_threshold(epochs)
    print(thres)
    epochs.drop_bad(reject = thres)

    print('----------------- Save ------------------')
    sink = DataSink(DERIV_ROOT, 'erp')
    erp_fpath = sink.get_path(
        subject = sub,
        task = TASK,
        run = run,
        desc = 'forERP',
        suffix = 'epo',
        extension = 'fif.gz'
    )
    print(f'Saving epochs for ERP analysis to: {erp_fpath}')
    epochs.save(erp_fpath, overwrite = True)

    print('----------------- generate a report ------------------')
    report = mne.Report(verbose = True)
    report.parse_folder(op.dirname(erp_fpath), pattern = '*epo.fif.gz', render_bem = False)
    if ica.exclude:
        fig_ica_removed = ica.plot_components(ica.exclude, show = False)
        report.add_figure(
            fig_ica_removed,
            title = 'Removed ICA Components',
            section = 'ICA'
        )
    bads = prep.noisy_channels_original
    html_lines = []
    for line in pformat(bads).splitlines():
        html_lines.append('<br/>%s' % line)
    html = '\n'.join(html_lines)
    report.add_html(html, title = 'Interpolated Channels', section = 'channels')
    report.add_html(epochs.info._repr_html_(), title = 'Epochs Info (FFR)', section = 'info')
    report.add_html(epochs.info._repr_html_(), title = 'Epochs Info (ERP)', section = 'info')
    report.save(op.join(sink.deriv_root, 'sub-%s.html'%sub), overwrite = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sub', type = str)
    parser.add_argument('run', type = str)
    args = parser.parse_args()
    main(args.sub, args.run)
