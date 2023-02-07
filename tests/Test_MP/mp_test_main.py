import numpy as np
import time
from multiprocessing import Pool, RawArray

import mp_test_module as mp_mod
import os, psutil


"""
if __name__ == '__main__':
    mp_mod.mp_process()

"""


#process = psutil.Process(os.getpid())
#print(process.memory_info().rss/1024/1024)

X_shape = (16, 100000)
# Randomly generate some data

data = np.random.randn(*X_shape)
if __name__ == '__main__':
    mp_mod.mp_unlock(data)


if __name__ == '__main__':
    mp_mod.mp_lock(data)


