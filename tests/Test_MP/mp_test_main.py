import os
filename = os.environ.get('PYTHONSTARTUP')
if filename and os.path.isfile(filename):
    with open(filename) as fobj:
        startup_file = fobj.read()
    exec(startup_file)

import matplotlib.pyplot as plt

import numpy as np
import time
from multiprocessing import Pool, RawArray

import mp_test_module as mp_mod
import os, psutil

import vir.mpct as mpct
"""
if __name__ == '__main__':
    mp_mod.mp_process()

"""



m1 = mpct.ps_mem()
X_shape = (16, 1024*1024*32)
#X_shape = (16, 1024*32)

# Randomly generate some data
data = np.random.randn(*X_shape)
print('Parent Data Address:',data.ctypes.data)

m2 = mpct.ps_mem()

if __name__ == '__main__':
    mp_mod.mp_unlock(data)

mpct.ps_info(out='Parent', m=[m1,m2])



"""
if __name__ == '__main__':
    mp_mod.mp_lock(data)
"""

