#!/usr/bin/env python3

import os
import subprocess
import argparse
from bids import BIDSLayout
from util.io.bids import DataSink
from util.io.iter_BIDSPaths import *

def main(subs, skips) -> None:
    BIDS_ROOT = '../data/bids'
    layout = BIDSLayout(BIDS_ROOT, derivatives = True)
    fpaths = layout.get(scope = 'preprocessing',
                    suffix='epo',
                    extension = 'fif.gz',
                    return_type = 'filename')
    
    for (fpath, sub, task, run) in iter_BIDSPaths(fpaths):
        # Don't run if file already exists
        DERIV_ROOT = '../data/bids/derivatives'
        sink = DataSink(DERIV_ROOT, 'decoding')
        save_fpath = sink.get_path(
            subject = sub,
            task = task,
            run = run,
            desc = 'stft',
            suffix = 'power',
            extension = 'npy',
        )
        if os.path.isfile(save_fpath):
            print(f"Stft already computed for {sub} run {run}")
            continue
        
        # if subs were given but sub is not in subs, don't preprocess
        if bool(subs) and sub not in subs:
            continue

        # if sub in skips, don't preprocess
        if sub in skips:
            continue

        # Get stft
        print('subprocess.check_call("sbatch ./compute_stft.py %s %s %s %s %s" % (fpath, sub, task, run, save_fpath), shell=True)')
        subprocess.check_call("sbatch ./compute_stft.py %s %s %s %s %s" % (fpath, sub, task, run, save_fpath), shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run compute_stft.py over given subjects')
    parser.add_argument('--subs', 
                        type = str, 
                        nargs = '*', 
                        help = 'subjects to compute stft for (e.g. 3 14 8), provide no argument to run over all subjects', 
                        default = [])
    parser.add_argument('--skips', 
                        type = str, 
                        nargs = '*', 
                        help = 'subjects NOT to stft for (e.g. 1 9)', 
                        default = [])
    args = parser.parse_args()
    subs = args.subs
    skips = args.skips
    print(f"subs: {subs}, skips : {skips}")
    if bool(subs) & bool(skips):
        raise ValueError('Cannot specify both subs and skips')
    main(subs, skips)
