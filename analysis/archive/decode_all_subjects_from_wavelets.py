#!/usr/bin/env python3

import os
import sys
import subprocess
import argparse
from bids import BIDSLayout
from util.io.iter_BIDSPaths import *
from util.io.bids import DataSink

def main(cond, subs, skips) -> None:
    BIDS_ROOT = '../data/bids'
    DERIV_ROOT = '../data/bids/derivatives'
    if cond == 'target':  
        CONDS = ['target']
    elif cond == 'tone':
        CONDS = ['tone_target', 'tone_nontarget']
    elif cond == 'binary':
        CONDS = ['11', '12', '13', '21', '22', '23', '31', '32', '33']

    layout = BIDSLayout(BIDS_ROOT, derivatives = True)
    fpaths = layout.get(scope = 'preprocessing',
                    suffix='epo',
                    extension = 'fif.gz',
                    return_type = 'filename')
    
    
    for (fpath, sub, task, run) in iter_BIDSPaths(fpaths):
        
        # if subs were given but sub is not in subs, don't preprocess
        if bool(subs) and sub not in subs:
            continue

        # if sub in skips, don't preprocess
        if sub in skips:
            continue
           
        for cond in CONDS:

            # skip if subject is already decoded
            sink = DataSink(DERIV_ROOT, 'decoding')
            scores_fpath = sink.get_path(
                subject = sub,
                task = task,
                run = run,
                desc = 'wavelet_' + cond,
                suffix = 'scores',
                extension = 'npy',
            )
            if os.path.isfile(scores_fpath) and sub not in subs:
                print(f"Subject {sub} run {run} is already preprocessed")
                continue
             
            print(f"subprocess.check_call('sbatch ./decode_from_wavelets.py %s %s %s %s %s %s' % ({fpath}, {sub}, {task}, {run}, {cond}, {scores_fpath}), shell=True)")
            subprocess.check_call("sbatch ./decode_from_wavelets.py %s %s %s %s %s %s" % (fpath, sub, task, run, cond, scores_fpath), shell=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run decode_from_wavelets.py over given subjects')
    parser.add_argument('cond', 
                        type = str, 
                        nargs = 1, 
                        help = 'condition, either <target> (decode if sound is a target) <binary> (for each condition decode if the tone is a target or distractor) or <tone> (for targets vs distractors, decode the identity of the tone', 
                        default = [])
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
    cond = args.cond[0]
    subs = args.subs
    skips = args.skips
    print(f"cond : '{cond}', subs: {subs}, skips : {skips}")
    if bool(subs) & bool(skips):
        raise ValueError('Cannot specify both subs and skips')
    if cond not in ['target', 'binary', 'tone']:
        parser.print_help(sys.stderr)
        sys.exit(1)
    main(cond, subs, skips)
