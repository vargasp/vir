# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 19:26:42 2026

@author: varga
"""

import numpy as np
import matplotlib.pyplot as plt
from vir.filtration import ramp_filter, window_filter



n = 1024
du =  1
cutoff = 1
zp = True
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





H, f = ramp_filter(n,du,real_space=False,zero_pad=zp,half_spectrum=hs,cutoff=cutoff,
                   return_freq=True)

H_r =  window_filter(H.copy(),f,du=du,filter_type="ram-lak")
H_hn =  window_filter(H.copy(),f,du=du,filter_type="hann")
H_hm =  window_filter(H.copy(),f,du=du,filter_type="hamming")
H_b =  window_filter(H.copy(),f,du=du,filter_type="blackman")
H_s =  window_filter(H.copy(),f,du=du,filter_type="shepp-logan")
H_c =  window_filter(H.copy(),f,du=du,filter_type="cosine")
H_g =  window_filter(H.copy(),f,du=du,filter_type="gaussian")


ind = np.argsort(f)
f = f[ind]
H = H[ind]
H_r = H_r[ind]
H_hm = H_hm[ind]
H_hn = H_hn[ind]
H_s = H_s[ind]
H_c = H_c[ind]
H_g = H_g[ind]
H_b = H_b[ind]

plt.title(f"n: {n:d}, du: {du:1.2f}, cutoff: {cutoff:1.2f}")
plt.plot(f,H, label='Ramp')
plt.plot(f,H_r, label='ram-lak')
plt.plot(f,H_hn, label='hann')
plt.plot(f,H_hm, label='hamming')
plt.plot(f,H_b, label='blackman')
plt.plot(f,H_s, label='shepp-logan')
plt.plot(f,H_c, label='cosine')
plt.plot(f,H_g, label='gaussian')
plt.legend()
plt.show()











