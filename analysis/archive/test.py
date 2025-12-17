#!/usr/bin/env python3

#SBATCH --account=pi-hcn1
#SBATCH --time=00:00:30
#SBATCH --mem=5G
#SBATCH --output=logs/test-%j.log

import os
print(os.getcwd())
cwd = os.getcwd()

import subprocess

#subprocess.run('source activate /project/hcn1/.conda/envs/mne', shell = True)

import sys
print(sys.path)

sys.path.append(cwd)

print(sys.path)
from util.io.bids import DataSink
