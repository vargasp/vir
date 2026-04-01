# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:17:00 2026

@author: varga
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale

import vir.filtration as filt


image = shepp_logan_phantom()
image = rescale(image, scale=0.4, mode='reflect', channel_axis=None)



na = 200


theta = np.linspace(0.0, 180.0,na, endpoint=False)

sino = radon(image, theta=theta)
nu, _ = sino.shape

fsino = filt.filter_sino(sino.T, filter_type='cosine' ).T


bp = iradon(sino, theta=theta, filter_name=None)
rec1 = iradon(sino, theta=theta, filter_name='cosine')
rec2 = iradon(fsino, theta=theta, filter_name=None)


plt.plot(rec1[:,80])
plt.plot(rec2[:,80])
plt.show()
         

plt.plot((rec1 - rec2)[:,80])
plt.show()
         