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

def Images(p):
    nX, nY, nZ = p.shape
    
    Z = [128,384,640]
    vert_label = ['Bottom','Middle','Top']

    for (z,label)in zip(Z,vert_label):
        """XY Planes Images"""
        vt.CreateImage(p[:,:,z], title='Phantom XY-Plane '+label,\
           xtitle='X Bins',ytitle='Y Bins',\
           coords=(-128,128,-128,128),aspect=1)
        
        vt.CreateImage(p[118:138,118:138,z], title='Phantom XY-Plane '+label+' (Zoomed Center)',\
           xtitle='X Bins',ytitle='Y Bins',\
           coords=(-10,10,-10,10),aspect=1)
    
        vt.CreateImage(p[182:202,118:138,z], title='Phantom XY-Plane '+label+' (Zoomed Edge)',\
               xtitle='X Bins',ytitle='Y Bins',\
               coords=(-10,10,54,74),aspect=1)    

        """YZ Planes Images"""
        vt.CreateImage(p[128,118:138,(z-10):(z+10)].T, title='Phantom YZ-Plane '+label+' (Zoomed Center)',\
           xtitle='Y Bins',ytitle='Z Bins',\
           coords=(-10,10,z-nZ/2-10,z-nZ/2+10),aspect=1)

        vt.CreateImage(p[192,118:138,(z-10):(z+10)].T, title='Phantom YZ-Plane '+label+' (Zoomed Edge)',\
           xtitle='Y Bins',ytitle='Z Bins',\
           coords=(54,74,z-nZ/2-10,z-nZ/2+10),aspect=1)

def Sinos(s):
    nAngs, nRows, nCols = s.shape
        
    vt.CreateImage(s[0,:,:], title='Projection View 0',\
           xtitle='Detector Cols',ytitle='Detector Rows',\
           coords=(-nCols/2,nCols/2,-nRows/2,nRows/2),aspect=1)
    
    Z = [128,384,640]
    vert_label = ['Bottom','Middle','Top']

    for (z,label)in zip(Z,vert_label):
        vt.CreateImage(s[:,z,:], title='Sinogram '+label,\
           xtitle='Detector Cols',ytitle='Angles',\
           coords=(-nCols/2,nCols/2,0,nAngs),aspect=1)

    sino = s[:,np.add.outer(np.array(Z), np.arange(-5,5)).flatten(),:]
    
    tomopy(sino)
    return 


data_dir = '/Users/pvargas21/Desktop/Wobble/'
data_dir = 'C:\\Users\\varga\\Desktop\\Wobble\\'
    

"""Phantom"""
phantom = np.load(data_dir+'wobble_phantom1.npy') + \
    np.load(data_dir+'wobble_phantom2.npy')/10

"""Parallel"""
sinoP = np.load(data_dir+'sinoP1.npy') + \
    np.load(data_dir+'sinoP2.npy')/10
sinoT = np.load(data_dir+'sinoT1.npy') + \
    np.load(data_dir+'sinoT2.npy')    
sinoRz = np.load(data_dir+'sinoRz1.npy') + \
    np.load(data_dir+'sinoRz2.npy')
sinoRa = np.load(data_dir+'sinoRa1.npy') + \
    np.load(data_dir+'sinoRa2.npy') 


nX, nY, nZ = phantom.shape
nAngs, nRows, nCols = sinoP.shape
Angs = np.linspace(0, 2*np.pi,nAngs,endpoint=False, dtype=np.float32)

"""Axis of rotaion, rotated with respect to z-axis"""
trans_X = 10.5
sinoTC = sc.calib_proj_orient(sinoT.copy(),Angs,transX=-trans_X)
Sinos(sinoTC)


    
"""Axis of rotaion, rotated with respect to z-axis"""
angX_Z,angY_Z = (0.0,0.05)
center_Z = (nX/2.-0.5, nY/2.-0.5, .5)

"""Precessing"""
angX_A,angY_A = (0.0,0.04)
center_A = (nX/2.-0.5, nY/2.-0.5, 64-.5)


"""Axis of rotaion - rotated and translated with respect to z-axis and precessing"""
sinoTRaRz = np.load('C:\\Users\\varga\\Desktop\\Wobble\sinoTRaRz.npy', )
    
    


    

    
Sinos(sinoRz)

    
    
    
    

    
    

    

vt.animated_gif(sinoRz, "sino_m", fps=128)



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
