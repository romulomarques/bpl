import os
import numpy as np

interval_len = 0.5
os.system('python extract_loops.py')
os.system('python create_clp_essential_hydro.py %f' % interval_len)
os.system('python run_bpl.py')
os.system('python generate_results_table.py')