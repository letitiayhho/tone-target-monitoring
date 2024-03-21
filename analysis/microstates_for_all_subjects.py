#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
from util.io.bids import DataSink
from util.io.iter_BIDSPaths import *

def main(force, subs, skips) -> None:
    BIDS_ROOT = '../data/bids'
    DERIV_ROOT = '../data/bids/derivatives/'
    layout = BIDSLayout(BIDS_ROOT, derivatives = True)
    subs = layout.get_subjects()
    BADS = []

    for sub in subs:
        if bool(subs) and sub not in subs:
            continue

        # if sub in skips, don't preprocess
        if sub in skips:
            continue

        # if sub is bad, don't preprocess
        if sub in BADS:
            continue

        # skip if subject is already preprocessed
        sink = DataSink(DERIV_ROOT, 'microstates')
        solution_fpath = sink.get_path(
            subject = sub,
            task = 'pitch',
            desc = 'microstates',
            suffix = 'ModKMeans',
            extension = '.fif.gz'
        )
        if os.path.isfile(solution_fpath) and not force:
            print(f"Subject {sub} is already preprocessed")
            continue

        print("subprocess.check_call(\"sbatch ./microstates.py %s\" % (sub), shell=True)")
        subprocess.check_call("sbatch ./microstates.py %s" % (sub), shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run preprocess.py over given subjects')
    parser.add_argument('--force', 
                        type = bool, 
                        nargs = 1, 
                        help = 'overwrite existing output file', 
                        default = False)
    parser.add_argument('--subs', 
                        type = str, 
                        nargs = '*', 
                        help = 'subjects to preprocess (e.g. 3 14 8), provide no argument to run over all subjects', 
                        default = [])
    parser.add_argument('--skips', 
                        type = str, 
                        nargs = '*', 
                        help = 'subjects NOT to preprocess (e.g. 1 9)', 
                        default = [])
    args = parser.parse_args()
    force = args.force
    subs = args.subs
    skips = args.skips
    print(f"subs: {subs}, skips : {skips}, force : {force}")
    if bool(subs) & bool(skips):
        raise ValueError('Cannot specify both subs and skips')
    main(force, subs, skips)
