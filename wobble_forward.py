# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 09:15:01 2025

@author: varga
"""

import numpy as np
import matplotlib.pyplot as plt

import vir
import vir.sino_calibration as sc
from vir.phantoms import discrete_circle
import vir.phantoms as pt

def gen_phantom2(f):
    #phantom = np.load('/Users/pvargas21/Desktop/derenzo_phantom3d512_512_128.npy')
    phantom = np.load('C:\\Users\\varga\\Desktop\\Samples\\derenzo_phantom3d512_512_128.npy')
    
    phantom = phantom + discrete_circle(radius=175, upsample=10)[:,:,np.newaxis]
    p_mid = phantom[367:398,239:273,:].copy()
    p_mid[:10,:5,:] = 1.0
    p_mid[:10,-5:,:] = 1.0
    p_mid[-10:,:5,:] = 1.0
    p_mid[-10:,-5:,:] = 1.0
    phantom[241:272,239:273] = p_mid
    phantom = np.tile(phantom.astype(np.float32),[1,1,5])
    phantom = vir.rebin(phantom, [256,256,512])
    phantom = np.dstack([np.zeros([256,256,128]),phantom,np.zeros([256,256,128])])
    
    return phantom


def gen_phantom(f):
    s1 = ([-.8,1.385640646,0,.8,1],\
          [.8,1.385640646,0,.8,1],\
          [-1.6,0,0,.8,1],\
          [0,0,0,.8,1],\
          [1.6,0,0,.8,1],\
          [-.8,-1.385640646,0,.8,1],\
          [.8,-1.385640646,0,.8,1])
    s2 = ([-.8,0.692820323,1.306394529,.8,1],\
          [.8,0.692820323,1.306394529,.8,1],\
          [-.8,-0.692820323,1.306394529,.8,1],\
          [.8,-0.692820323,1.306394529,.8,1])
    
        
    z=32
    dZ = 1.306394529
    #Number of z planes for radii set    
    nZ1 = int(2*((z/2) // dZ) + 1)
    nZ2 = int(2*(((z/2 - dZ/2) // dZ) + 1))
    
    Z1 = vir.censpace(nZ1,d=dZ)
    Z2 = vir.censpace(nZ2,d=dZ)
        
    S1s = np.tile(s1, (nZ1,1))
    S2s = np.tile(s2, (nZ2,1))
    S1s[:,2] = np.repeat(Z1,7)
    S2s[:,2] = np.repeat(Z2,4)
    
    S = np.vstack([S1s,S2s]) 
    S[:,:3] *=8
    S[:,3] *=4
    
            

    t = pt.DiscretePhantom()
    t.updatePhantomDiscrete(S)
    
    phantom = t.phantom.copy()
    phantom = phantom + np.roll(phantom,128,axis=0) +\
              np.roll(phantom,-128,axis=0) +\
              np.roll(phantom.transpose([1,0,2]),128,axis=1) +\
              np.roll(phantom.transpose([1,0,2]),-128,axis=1)
    
    #phantom = phantom + discrete_circle(radius=175, upsample=10)[:,:,np.newaxis]
    
    phantom = np.tile(phantom.astype(np.float32),[1,1,3])
    phantom = phantom[:,:,64:-64]
    phantom2 = np.ones(phantom.shape)
    phantom2 = phantom2*discrete_circle(radius=175, upsample=10)[:,:,np.newaxis]
    
    phantom = np.dstack([np.zeros([512,512,64]),phantom,np.zeros([512,512,64])])
    phantom2 = np.dstack([np.zeros([512,512,64]),phantom2,np.zeros([512,512,64])])

    phantom = vir.rebin(phantom, [256,256,768])
    phantom2 = vir.rebin(phantom2, [256,256,768])
    phantom = phantom.astype(np.float32)
    phantom2 = phantom2.astype(np.float32)
    
    return phantom, phantom2
   

"""Paralell"""
phantom1, phantom2 = gen_phantom(1)
np.save('C:\\Users\\varga\\Desktop\\Wobble\\wobble_phantom1',phantom1)
np.save('C:\\Users\\varga\\Desktop\\Wobble\\wobble_phantom2',phantom2)
#phantom = np.load('C:\\Users\\varga\\Desktop\\Wobble\wobble_phantom.npy')
nX, nY, nZ = phantom1.shape


nAngs = 512
nRots = 1
nViews = nRots*nAngs
Views = np.linspace(0, nRots*2*np.pi,nViews,endpoint=False, dtype=np.float32)
sino1 = sc.forward_project_phantom_misalign(phantom1, Views)
sino2 = sc.forward_project_phantom_misalign(phantom2, Views)
np.save('C:\\Users\\varga\\Desktop\\Wobble\sinoP1', sino1)
np.save('C:\\Users\\varga\\Desktop\\Wobble\sinoP2', sino2)
    
"""Axis of rotaion, rotated with respect to z-axis"""
trans_X = 10.5
sinoT1 = sc.forward_project_phantom_misalign(phantom1, Views, trans_X=trans_X)    
sinoT2 = sc.forward_project_phantom_misalign(phantom2, Views, trans_X=trans_X)
np.save('C:\\Users\\varga\\Desktop\\Wobble\sinoT1', sinoT1)
np.save('C:\\Users\\varga\\Desktop\\Wobble\sinoT2', sinoT2)

    
"""Axis of rotaion, rotated with respect to z-axis"""
angX_Z,angY_Z = (0.0,0.05)
center_Z = (nX/2.-0.5, nY/2.-0.5, 127)
sinoRz1 = sc.forward_project_phantom_misalign(phantom1, Views, \
                                        angX_Z=angX_Z,angY_Z=angY_Z,center_Z=center_Z)    
sinoRz2 = sc.forward_project_phantom_misalign(phantom2, Views, \
                                        angX_Z=angX_Z,angY_Z=angY_Z,center_Z=center_Z)    
np.save('C:\\Users\\varga\\Desktop\\Wobble\sinoRz1', sinoRz1)
np.save('C:\\Users\\varga\\Desktop\\Wobble\sinoRz2', sinoRz2)





"""Precessing"""
angX_A,angY_A = (0.0,0.04)
center_A = (nX/2.-0.5, nY/2.-0.5, 64-.5)
sinoRa = sc.forward_project_phantom_misalign(phantom, Views, \
                                        angX_A=angX_A,angY_A=angY_A,center_A=center_A)
np.save('C:\\Users\\varga\\Desktop\\Wobble\sinoRa', sinoRa)
    
    
"""Axis of rotaion - rotated and translated with respect to z-axis and precessing"""
sinoTRaRz = sc.forward_project_phantom_misalign(phantom, Views, trans_X=trans_X,\
                                        angX_Z=angX_Z,angY_Z=angY_Z,center_Z=center_Z,\
                                        angX_A=angX_A,angY_A=angY_A,center_A=center_A)    
np.save('C:\\Users\\varga\\Desktop\\Wobble\sinoTRaRz', sinoTRaRz)
    
    
    


            
