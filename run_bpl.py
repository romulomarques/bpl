import sys
import os
import numpy as np
from bpl import run_bpl

WDIR = ['DATA_LOOP_04', 'DATA_LOOP_08', 'DATA_LOOP_12']
# NUM = [10, 100, 1000]
NUM = [1000]

if __name__ == "__main__":
    # run the 'run_bpl' function for each file in each directory,
    # and for each number in 'NUM'
    for wdir in WDIR:
        for num in NUM:
            for fdat in os.listdir(wdir):
                if not fdat.endswith('.dat'):
                    continue
                fdat = os.path.join(wdir, fdat)
                run_bpl(fdat, num)