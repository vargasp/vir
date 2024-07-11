import os
filename = os.environ.get('PYTHONSTARTUP')
if filename and os.path.isfile(filename):
    with open(filename) as fobj:
        startup_file = fobj.read()
    exec(startup_file)


import numpy as np
import mp_test_module as mp_mod



if __name__ == '__main__':
        
    #X_shape = (16, 1024*1024*32)
    X_shape = (16, 1024*32)
    
    # Randomly generate some data
    data = np.random.randn(*X_shape)

    
    #Test function using a shared array that read only
    mp_mod.mp_unlock(data)

    #Test function using a shared array that read/write
    mp_mod.mp_lock(data)