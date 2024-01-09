#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 15:01:39 2023

@author: pvargas21
"""


import numpy as np
import matplotlib.pyplot as plt

import vir
import vir.phantoms as pt
import vir.sinogram as sg
import vt



def gen_phantom(nX,nY,nZ,f=1):
    sX = slice(int(nX/2-1*f), int(nX/2+1*f))
    sY = slice(int(nY/2-1*f), int(nY/2+1*f))
    
    phantom = pt.discrete_circle(nPixels=(nX,nY), radius=nX/8, upsample=4)
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
    m, y_int = np.polyfit(np.arange(params.shape[1]), params[1,:], 1)
    z_center = -y_int/m
    phi = np.arctan(m)
    theta = params[2,:].mean()

    return center, z_center, phi, theta





nX = 32
nY = 32
nZ = 32

nAng = int(nX*2)
angs = np.linspace(0,np.pi*2,nAng,endpoint=False)
phantom = gen_phantom(nX,nY,int(nZ*1.25))

Xs = vir.censpace(13).astype(int)
Ys = vir.censpace(13).astype(int)
Xs = np.array([0])
Ys = np.array([0])

phis = vir.censpace(13,2.5*np.pi/180)
thetas = vir.censpace(13,2.5*np.pi/180)

params_arr = np.zeros((Xs.size,Ys.size,thetas.size,phis.size,3,nZ ))
center_arr = np.zeros((Xs.size,Ys.size,thetas.size,phis.size))
z_center_arr = np.zeros((Xs.size,Ys.size,thetas.size,phis.size))
phi_arr = np.zeros((Xs.size,Ys.size,thetas.size,phis.size))
theta_arr = np.zeros((Xs.size,Ys.size,thetas.size,phis.size))

for i, x, in enumerate(Xs): 
    for j, y, in enumerate(Ys):
        center_rot = (nX/2.-.5+x,nY/2.-.5+y,nZ/2.-.5)
        phant_shift = np.roll(phantom,(x,y),axis=(0,1))

        for k, theta, in enumerate(thetas): 
            for l, phi, in enumerate(phis): 

                c = int(0.125*nZ)
                sino = gen_sino(phant_shift,angs,phi=phi,theta=theta,center=center_rot)[:,c:(nZ+c),:]
        
                params = sg.estimate_wobble(sino,angs)
                center_e, z_center_e, phi_e, theta_e = gen_params(params)
                print(i,j,k,l)
                #print(f'Center: {center_e:.2f}, Z Center: {z_center_e:.2f}, Phi: {phi_e/np.pi*180:.2f}, Theta:{theta_e*np.pi/180:2f}')
                #print('')
                params_arr[i,j,k,l,:,:] = params
                center_arr[i,j,k,l] = center_e
                z_center_arr[i,j,k,l] = z_center_e
                phi_arr[i,j,k,l] = phi_e
                theta_arr[i,j,k,l] = theta_e
        



        

params_arr2 = np.zeros((Xs.size,Ys.size,thetas.size,phis.size,3,128 ))
center_arr2 = np.zeros((Xs.size,Ys.size,thetas.size,phis.size))
z_center_arr2 = np.zeros((Xs.size,Ys.size,thetas.size,phis.size))
phi_arr2 = np.zeros((Xs.size,Ys.size,thetas.size,phis.size))
theta_arr2 = np.zeros((Xs.size,Ys.size,thetas.size,phis.size))

for i, x, in enumerate(Xs): 
    for j, y, in enumerate(Ys): 
        phant_shift = np.roll(phantom,(x,y),axis=(0,1))

        for k, theta, in enumerate(thetas): 
            for l, phi, in enumerate(phis): 

                sino = gen_sino(phant_shift,angs,phi=phi,theta=theta)[:,26:164,:]
        
                params = sg.estimate_wobble(sino,angs)
                center, z_center, phi, theta = gen_params(params)
                print(i,j,k,l)
                print(f'Center: {center:.2f}, Z Center: {z_center:.2f}, Phi: {phi/np.pi*180:.2f}, Theta:{theta*np.pi/180:2f}')
                print('')
                params_arr2[i,j,k,l,:,:] = params
                center_arr2[i,j,k,l] = center
                z_center_arr2[i,j,k,l] = z_center
                phi_arr2[i,j,k,l] = phi
                theta_arr2[i,j,k,l] = theta
        
        