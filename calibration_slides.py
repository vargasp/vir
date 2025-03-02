#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 09:01:01 2025

@author: pvargas21
"""

import numpy as np
import matplotlib.pyplot as plt
from vir.phantoms import discrete_circle

phantom = np.load('/Users/pvargas21/Desktop/derenzo_phantom3d512_512_128.npy')

phantom = phantom + discrete_circle(radius=175, upsample=10)[:,:,np.newaxis]
p_mid = phantom[367:398,239:273,:].copy()
p_mid[:10,:5,:] = 1.0
p_mid[:10,-5:,:] = 1.0
p_mid[-10:,:5,:] = 1.0
p_mid[-10:,-5:,:] = 1.0
phantom[241:272,239:273] = p_mid


phantom = np.tile(phantom.astype(np.float32),[1,1,4])





plt.imshow(phantom[:,:,128],origin='lower')
plt.show()





plt.imshow(phantom[:,:,128])
plt.show()


plt.imshow(phantom[367:398,239:273,128],origin='lower')
plt.show()


plt.imshow(phantom[300:400,200:,126],origin='lower')
plt.show()



for i in np.arange(10,20):
    plt.imshow(phantom[:,:,i])
    plt.show()



plt.plot(phantom[:,256,13])
plt.show()





plt.plot(phantom[:,256,128])
plt.show()
