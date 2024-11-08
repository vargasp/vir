import os
filename = os.environ.get('PYTHONSTARTUP')
if filename and os.path.isfile(filename):
    with open(filename) as fobj:
        startup_file = fobj.read()
    exec(startup_file)


import numpy as np
import mp_test_module as mp_mod

"""

if __name__ == '__main__':
        
    #X_shape = (16, 1024*1024*32)
    X_shape = (16, 1024*32)
    
    # Randomly generate some data
    data = np.random.randn(*X_shape)

    
    #Test function using a shared array that read only
    mp_mod.mp_unlock(data)

<<<<<<< HEAD
mpct.ps_info(out='Parent', m=[m1,m2])

"""

"""
if __name__ == '__main__':
    mp_mod.mp_lock(data)
"""

if __name__ == '__main__':
    mp_mod.mp_star_mp()
=======
    #Test function using a shared array that read/write
    mp_mod.mp_lock(data)
>>>>>>> cc2c63ad8c17edc2a7a52b0dbccffed4e765d3dc
