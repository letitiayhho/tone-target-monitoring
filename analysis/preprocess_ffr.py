#!/usr/bin/env python3

#SBATCH --time=01:00:00 # 40 min enough for most
#SBATCH --partition=bigmem2
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=120G # 96 enough for most
#SBATCH --mail-type=all
#SBATCH --mail-user=letitiayhho@uchicago.edu
#SBATCH --output=logs/preprocess_ffr_%j.log

import numpy as np
import os.path as op
from pprint import pformat
import argparse
# EEG utilities
import mne
from mne.preprocessing import ICA, create_eog_epochs
from pyprep.prep_pipeline import PrepPipeline
# BIDS utilities
from mne_bids import BIDSPath, read_raw_bids
from util.io.bids import DataSink

# constants
BIDS_ROOT = '../data/bids'
DERIV_ROOT = op.join(BIDS_ROOT, 'derivatives')
FFR_PASSBAND = (100., 300.)
MICROSTATE_PASSBAND = (1., 30.)
TASK = 'pitch'
TMIN = -0.25
TMAX = 0.25

def main(sub, run):
    '''
    Parameters
    ----------
    sub : str
        Subject ID as in BIDS dataset
    '''
    # load data
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

    # re-reference eye electrodes to become bipolar EOG
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

    # run PREP pipeline (notch, exclude bad chans, and re-reference)
    print('----------------- run PREP pipeline ------------------')
    raw, events = raw.resample(int(4*FFR_PASSBAND[1]), events = events) # resample to 1200 Hz
    np.random.seed(int(sub))
    lf = raw.info['line_freq']
    prep_params = {
        "ref_chs": "eeg",
        "reref_chs": "eeg",
        "line_freqs": np.arange(lf, FFR_PASSBAND[1], lf)
    }
    prep = PrepPipeline(
        raw,
        prep_params,
        raw.get_montage(),
        ransac = False,
        random_state = int(sub)
        )
    prep.fit()
    
    # Extract data from PREP
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

    ## reconstruct the typical FFR/ABR montage we know and love
    print('----------------- reconstruct the typical FFR/ABR montage ------------------')
    raw_for_ffr = raw.copy().pick(['Cz', 'TP9', 'TP10']) # Cz and mastoids
    raw_for_ffr.set_eeg_reference(ref_channels = ['TP9', 'TP10'])
    raw_for_ffr = raw_for_ffr.pick(['Cz'])
    # and filter (just the highpass, since lowpass applied when downsampling)
    raw_for_ffr = raw_for_ffr.filter(l_freq = FFR_PASSBAND[0], h_freq = None)
    # then epoch
    epochs = mne.Epochs(
        raw_for_ffr,
        events,
        tmin = TMIN,
        tmax = TMAX,
        event_id = event_ids,
        baseline = (TMIN, 0.),
        preload = True
    )
    # drop bad epochs
    epochs.drop_bad(reject = dict(eeg = 35e-6))
    # then save epochs for later
    sink = DataSink(DERIV_ROOT, 'preprocess_ffr')
    ffr_fpath = sink.get_path(
        subject = sub,
        task = TASK,
        run = run,
        desc = 'forFFR2',
        suffix = 'epo',
        extension = 'fif.gz'
    )
    print(f'Saving epochs object to: {ffr_fpath}')
    epochs.save(ffr_fpath, overwrite = True)

    ## now prepare non-epoched data for microstate jazz
    # identify bad ICs on weakly highpassed data
    print('----------------- prepare non-epoched data for microstate jazz ------------------')
    raw_for_ica = raw.copy().filter(l_freq = 1., h_freq = None)
    epochs_for_ica = mne.Epochs(
        raw_for_ica,
        epochs.events, # same events as FFR epochs
        tmin = TMIN,
        tmax = .0, # only prestim
        event_id = event_ids,
        baseline = None,
        preload = True
    )
    
    print('----------------- identify bad ICs on weakly highpassed data ------------------')
    ica = ICA(n_components = 15, random_state = 0)
    ica.fit(epochs_for_ica, picks = ['eeg', 'eog'])
    eog_indices, eog_scores = ica.find_bads_eog(epochs_for_ica, threshold = 1.96)
    ica.exclude = eog_indices
    
    # filter to desired bandwidth and remove bad ICs
    print('----------------- filter to desired bandwidth and remove bad ICs ------------------')
    raw = raw.filter(*MICROSTATE_PASSBAND)
    epochs_for_micro = mne.Epochs(
        raw,
        epochs.events, # same events as FFR epochs
        tmin = TMIN,
        tmax = .0, # only prestim
        event_id = event_ids,
        baseline = None,
        preload = True
    )
    
    # apply ICA
    print('----------------- apply ICA ------------------')
    ica.apply(epochs_for_micro) # transforms in place
    # now we no longer need EOG channels
    epochs_for_micro = epochs_for_micro.drop_channels('leog')
    epochs_for_micro = epochs_for_micro.drop_channels('reog')
    # and save
    sink = DataSink(DERIV_ROOT, 'microstates')
    micro_fpath = sink.get_path(
        subject = sub,
        task = TASK,
        run = run,
        desc = 'forMicrostate2',
        suffix = 'epo',
        extension = 'fif.gz'
    )
    
    # Save data for microstate jazz
    print(f'Saving epochs for microstate jazz to: {micro_fpath}')
    epochs_for_micro.save(micro_fpath, overwrite = True)

    # generate a report
    print('----------------- generate a report ------------------')
    report = mne.Report(verbose = True)
    report.parse_folder(op.dirname(ffr_fpath), pattern = '*epo.fif.gz', render_bem = False)
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
    report.add_html(epochs_for_micro.info._repr_html_(), title = 'Epochs Info (Microstates)', section = 'info')
    report.save(op.join(sink.deriv_root, 'sub-%s.html'%sub), overwrite = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('sub', type = str)
    parser.add_argument('run', type = str)
    args = parser.parse_args()
    main(args.sub, args.run)
