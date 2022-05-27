import numpy as np
import time
from multiprocessing import Pool, RawArray

import mp_test_module as mp_mod



"""
if __name__ == '__main__':
    mp_mod.mp_process()

"""

X_shape = (16, 1000000)
# Randomly generate some data


data = np.random.randn(*X_shape)
if __name__ == '__main__':
    mp_mod.mp_unlock(data)


if __name__ == '__main__':
    mp_mod.mp_lock(data)


