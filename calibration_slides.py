#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 09:01:01 2025

@author: pvargas21
"""

import numpy as np
import matplotlib.pyplot as plt

import vt
import vir
import vir.sino_calibration as sc
from vir.phantoms import discrete_circle

def gen_phantom(f):
    #phantom = np.load('/Users/pvargas21/Desktop/derenzo_phantom3d512_512_128.npy')
    phantom = np.load('C:\\Users\\varga\\Desktop\\Samples\\derenzo_phantom3d512_512_128.npy')
    
    phantom = phantom + discrete_circle(radius=175, upsample=10)[:,:,np.newaxis]
    p_mid = phantom[367:398,239:273,:].copy()
    p_mid[:10,:5,:] = 1.0
    p_mid[:10,-5:,:] = 1.0
    p_mid[-10:,:5,:] = 1.0
    p_mid[-10:,-5:,:] = 1.0
    phantom[241:272,239:273] = p_mid
    
    return np.tile(phantom.astype(np.float32),[1,1,4])
   


"""Paralell"""
phantom = gen_phantom(1)
phantom = vir.rebin(phantom, [256,256,512])
phantom = np.dstack([np.zeros([256,256,128]),phantom,np.zeros([256,256,128])])
nX, nY, nZ = phantom.shape


nAngs = 256
nRots = 3
nViews = nRots*nAngs
Views = np.linspace(0, nRots*2*np.pi,nViews,endpoint=False, dtype=np.float32)
sino = sc.forward_project_phantom_misalign(phantom, Views)

    
"""Axis of rotaion, rotated with respect to z-axis"""
trans_X = 10.5
sinoT = sc.forward_project_phantom_misalign(phantom, Views, trans_X=trans_X)    
    
    
"""Axis of rotaion, rotated with respect to z-axis"""
angX_Z,angY_Z = (0.0,0.05)
center_Z = (nX/2.-0.5, nY/2.-0.5, .5)
sinoRz = sc.forward_project_phantom_misalign(phantom, Views, \
                                        angX_Z=angX_Z,angY_Z=angY_Z,center_Z=center_Z)    
    
"""Precessing"""
angX_A,angY_A = (0.0,0.04)
center_A = (nX/2.-0.5, nY/2.-0.5, 64-.5)
sinoRa = sc.forward_project_phantom_misalign(phantom, Views, \
                                        angX_A=angX_A,angY_A=angY_A,center_A=center_A)
    
    
"""Axis of rotaion - rotated and translated with respect to z-axis and precessing"""
sinoTRaRz = sc.forward_project_phantom_misalign(phantom, Views, trans_X=trans_X,\
                                        angX_Z=angX_Z,angY_Z=angY_Z,center_Z=center_Z,\
                                        angX_A=angX_A,angY_A=angY_A,center_A=center_A)    
    
    
    
    
    
    
    
    
    
    
    vt.CreateImage(sino[0,:,:], title='Projection View 0',\
               xtitle='Detector Cols',ytitle='Detector Rows',\
               coords=(-128,128,0,nViews),aspect=1)
    
    
    
    

    

vt.animated_gif(sino, "sino_m", fps=48)



vt.animated_gif(sino, "sino", fps=48)

view_sinos(sino)

ph

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
