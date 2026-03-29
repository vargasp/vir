# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:26:42 2026

@author: varga
"""

import numpy as np
import matplotlib.pyplot as plt
from vir.filtration import ramp_filter



n = 1024
du =  1
cutoff = 1
zp = False
hs = False


if zp:
    nfu = int(2**np.ceil(np.log2(2 * n)))
else:
    nfu=n

if hs:
    v = np.fft.rfftfreq(nfu, d=du)
else:
    v = np.fft.fftfreq(nfu,d=du)
    
H_real = ramp_filter(n,du,real_space=True,zero_pad=zp,half_spectrum=hs,cutoff=cutoff)
H_freq = ramp_filter(n,du,real_space=False,zero_pad=zp,half_spectrum=hs,cutoff=cutoff)




ind = np.argsort(v)
v = v[ind]
H_real = H_real[ind]
H_freq = H_freq[ind]

print(np.max(np.abs(H_real - H_freq)))

plt.title(f"n: {n:d}, du: {du:1.2f}, cutoff: {cutoff:1.2f}")
plt.plot(v,H_real, label='Real')
plt.plot(v,H_freq, label='Freq')
plt.legend()
plt.show()












