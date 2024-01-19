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


def gen_sino(phantom,angs,phi=0,theta=0, psi=0,center=None):
    nX, nY, nZ = phantom.shape

    if center is None:
        center = (nX/2.-.5,nY/2.-.5,nZ/2.-.5)
    
    return sg.forward_project_wobble(phantom, angs+theta, phi, psi, center=center)


def gen_params(params):
    center = params[0,:].mean()
    m, y_int = np.polyfit(np.arange(params.shape[1]), params[1,:], 1)
    z_center = -y_int/m
    phi = -np.arctan(m)
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
psis = vir.censpace(13,2.5*np.pi/180)
thetas = np.linspace(0,2*np.pi, 13)

params_arr = np.zeros((Xs.size,Ys.size,thetas.size,phis.size,3,nZ ))
center_arr = np.zeros((Xs.size,Ys.size,thetas.size,phis.size))
z_center_arr = np.zeros((Xs.size,Ys.size,thetas.size,phis.size))
phi_arr = np.zeros((Xs.size,Ys.size,thetas.size,phis.size))
theta_arr = np.zeros((Xs.size,Ys.size,thetas.size,phis.size))

for i, x, in enumerate(Xs): 
    for j, y, in enumerate(Ys):
        center_rot = (nX/2.-.5+x,nY/2.-.5+y,(nZ*1.25)/2.-.5)
        phant_shift = np.roll(phantom,(x,y),axis=(0,1))

        for k, theta, in enumerate(thetas): 
            for l, phi, in enumerate(phis): 

                c = int(0.125*nZ)
                sino = gen_sino(phant_shift,angs,phi=phi,theta=theta,center=center_rot)[:,c:(nZ+c),:]
        
                params = sg.estimate_wobble(sino,angs)
                center_e, z_center_e, phi_e, theta_e = gen_params(params)
                #print(i,j,k,l,"Modes:",np.sum(np.histogram(params[2,:],bins=np.linspace(0,np.pi*2,8,endpoint=True))[0] >0))
                #print(f'Center: {center_e:.2f}, Z Center: {z_center_e:.2f}, Phi: {phi_e/np.pi*180:.2f}, Theta:{theta_e*np.pi/180:2f}')
                #print('')
                print(i,j,k,l,f"Phi: {phi:.2f},{phi_e:.2f}; Theta: {theta:.2f},{theta_e:.2f}")
                
                params_arr[i,j,k,l,:,:] = params
                center_arr[i,j,k,l] = center_e
                z_center_arr[i,j,k,l] = z_center_e
                phi_arr[i,j,k,l] = phi_e
                theta_arr[i,j,k,l] = theta_e
                
                sino_c = sg.correct_wobble(sino, angs, phi_e, theta_e, center=(z_center_e, 15.5))

        


sino1 = gen_sino(phant_shift,angs,phi=phis[0],theta=thetas[0],center=center_rot)[:,c:(nZ+c),:]
params1 = sg.estimate_wobble(sino1,angs)

