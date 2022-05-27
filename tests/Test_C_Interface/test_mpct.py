
import ctypes 
import numpy as np

import vir.mpct as mpct


var = np.array([5,4], dtype=np.float32)
sino2 = np.zeros([4,5,6], dtype=np.float32)
isinstance(var[0], np.float32) 
isinstance(sino2[0], np.float32) 


print(var.dtype)

test = mpct.ctypes_vars(var)




test = mpct.ctypes_vars(sino2)
