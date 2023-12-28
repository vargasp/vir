#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:01:39 2023

@author: pvargas21
"""


import numpy as np
import matplotlib.pyplot as plt

import vir
import vir.sinogram as sg
import vt



def gen_phantom(nX,nY,nZ,f=1):
    sX = slice(int(nX/2-1*f), int(nX/2+1*f))
    sY = slice(int(nY/2-1*f), int(nY/2+1*f))
    
    phantom = np.zeros([nX*f, nY*f])
    phantom[sX,sY] = 1
    phantom = np.tile(phantom, (f*nZ,1,1))
    phantom = phantom.transpose([1,2,0])


    return phantom


def gen_sino(phantom,angs,phi=0,theta=0,center=None):
    nX, nY, nZ = phantom.shape

    if center is None:
        center = (nX/2.-.5,nY/2.-.5,nZ/2.-.5)
    
    return sg.forward_project_wobble(phantom, angs, phi, theta, center=center)


def gen_params(params):
    center = params[0,:].mean()
    m, y_int = np.polyfit(np.arange(128), params[1,:], 1)
    z_center = -y_int/m
    phi = np.arctan(m)
    theta = params[2,:].mean()

    return center, z_center, phi, theta



nX = 128
nY = 128
nZ = 160

nAng = 256
angs = np.linspace(0,np.pi*2,nAng,endpoint=False)
phantom = gen_phantom(nX,nY,nZ)


Xs = vir.censpace(5).astype(int)
Ys = vir.censpace(5).astype(int)
phi = 10*np.pi/180
theta = 0

params_arr = np.zeros((Xs.size,Ys.size,3,128 ))
center_arr = np.zeros((Xs.size,Ys.size))
z_center_arr = np.zeros((Xs.size,Ys.size))
phi_arr = np.zeros((Xs.size,Ys.size))
theta_arr = np.zeros((Xs.size,Ys.size))


for i, x, in enumerate(Xs): 
    for j, y, in enumerate(Ys): 

        phant_shift = np.roll(phantom,(x,y),axis=(0,1))
        sino = gen_sino(phant_shift,angs,phi=0,theta=0)[:,16:144,:]
        
        params = sg.estimate_wobble(sino,angs)
        center, z_center, phi, theta = gen_params(params)        
        print(f'Center: {center:.2f}, Z Center: {z_center:.2f}, Phi: {phi/np.pi*180:.2f}, Theta:{theta*np.pi/180:2f}')

        params_arr[i,j,:,:] = params
        center_arr[i,j] = center
        z_center_arr[i,j] = z_center
        phi_arr[i,j] = phi
        theta_arr[i,j] = theta


