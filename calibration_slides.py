#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 09:01:01 2025

@author: pvargas21
"""

from sys import path

path.append('C:\\Users\\varga\\Codebase\\Libraries')

import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import iradon


import vt
import vir
import vir.sino_calibration as sc
from vir.phantoms import discrete_circle


def sino_par2hel(sinoP, nRows, nAngs, pitch, transD=0.0, rD=0.0):
    nViews, nRowsP, nCols = sinoP.shape
    
    dView = (Views[-1] - Views[0])/(nViews-1)
    dZ = pitch*nRows*dView/(np.pi*2)
    c =nRowsP/2 - nRows/2 +1
    Z = vir.censpace(nViews,c=c,d=dZ)

    sinoH = np.zeros([Views.size,nRows,nCols], dtype=np.float32)


    for i in range(nViews):
        coords = af.coords_array((1,nRows,nCols), ones=True)
        coords[:,:,0,:] = i
        coords[:,:,1,:] = coords[:,:,1,:] + Z[i]
        
        center=(0.0,Z[i]+nRows/2-0.5,nCols/2-0.5)
        T, R = sc.proj_orient_TM(rD, transD, center)  

        RTC = (np.linalg.inv(R @ T) @ coords)

        sinoH[i,:,:] = af.coords_transform(sinoP, np.round(RTC,6))

    return sinoH



def Images(p, Z = [128,384,640]):
    nX, nY, nZ = p.shape
    
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


def Recs(s,Views):
    nAngs, nRows, nCols = s.shape
    
    Z = [128,384,640]
    sino = s[:,np.add.outer(np.array(Z), np.arange(-10,10)).flatten(),:]
    recs = np.zeros([nCols,nCols,60])
    
    for i in range(60):
        recs[:,:,i] = iradon(sino[:,i,:].T, theta=Views/np.pi*180, \
                             filter_name='ramp')

    return recs



def Plots(rec, recC, plabel):
    Z = [10,30,50]
    vert_label = ['Bottom','Middle','Top']
    labels = ['No degradation',plabel]

    for (z,label)in zip(Z,vert_label):
    
        y = np.vstack([rec[128,118:138,z],recC[128,118:138,z]]).T
        vt.CreatePlot(y,xs=vir.censpace(20),labels=labels, ylims=(0,1.1),\
                      title='X Profile '+label+' (Zoomed Center)',\
                      xtitle='X Pixels', ytitle='Intensity')

        y = np.vstack([rec[128,182:202,z],recC[128,182:202,z]]).T
        vt.CreatePlot(y,xs=vir.censpace(20,c=64),labels=labels, ylims=(0,1.1), \
                      title='X Profile '+label+' (Zoomed Edge)',\
                      xtitle='X Pixels', ytitle='Intensity')



        y = np.vstack([rec[128,128,(z-10):(z+10)],recC[128,128,(z-10):(z+10)]]).T
        vt.CreatePlot(y,xs=vir.censpace(20),labels=labels, ylims=(0,1.1), \
                      title='Z Profile '+label+' (Zoomed Center)',\
                      xtitle='Z Pixels', ytitle='Intensity')








#data_dir = '/Users/pvargas21/Desktop/Wobble/'
data_dir = 'C:\\Users\\varga\\Desktop\\Wobble\\'
    

"""Phantom"""
phantom = np.load(data_dir+'wobble_phantom1.npy') + \
    np.load(data_dir+'wobble_phantom2.npy')/10

"""Parallel"""
sinoP = np.load(data_dir+'sinoP1.npy') + \
    np.load(data_dir+'sinoP2.npy')/10
sinoT = np.load(data_dir+'sinoT1.npy') + \
    np.load(data_dir+'sinoT2.npy')/10    
sinoRz = np.load(data_dir+'sinoRz1.npy') + \
    np.load(data_dir+'sinoRz2.npy')/10
sinoRa = np.load(data_dir+'sinoRa1.npy') + \
    np.load(data_dir+'sinoRa2.npy')/10 
sinoTRaRz = np.load(data_dir+'sinoTRaRz1.npy') + \
    np.load(data_dir+'sinoTRaRz2.npy')/10 


"""Correct Sinograms"""
nX, nY, nZ = phantom.shape
nAngs, nRows, nCols = sinoP.shape
Angs = np.linspace(0, 2*np.pi,nAngs,endpoint=False, dtype=np.float32)

"""Axis of rotaion, rotated with respect to z-axis"""
trans_X = 10.5
sinoTC = sc.calib_proj_orient(sinoT.copy(),Angs,transX=-trans_X)
Sinos(sinoTC)

"""Axis of rotaion, rotated with respect to z-axis"""
angX_Z,angY_Z = (0.0,0.05)
center_Z = (nX/2.-0.5, nY/2.-0.5, 127)
sinoRzC = sc.calib_proj_orient(sinoRz.copy(),Angs,rZ=-angY_Z,cenZ_y=center_Z[2])
Sinos(sinoRzC)

"""Precessing""" 
angX_A,angY_A = (0.0,0.075)
center_A = (nX/2.-0.5, nY/2.-0.5, 383.0)
sinoRaC = sc.calib_proj_orient(sinoRa.copy(),Angs,phi=angY_A,cenA_y=center_A[2])
print(  np.sqrt(np.sum((sinoRaC - sinoP)**2)))




Sinos(sinoRa)
Sinos(sinoRaC)

"""Axis of rotaion - rotated and translated with respect to z-axis and precessing"""
sinoTRaRzC = sc.calib_proj_orient(sinoTRaRz.copy(),Angs, transX=-trans_X,\
                                      rZ=-angY_Z,cenZ_y=center_Z[2],\
                                      phi=angY_A,cenA_y=center_A[2])
Sinos(sinoTRaRzC)


"""Reconstruct Sinograms"""
recP = Recs(sinoP,Angs)
recT = Recs(sinoT,Angs)
recTC = Recs(sinoTC,Angs)
recRa = Recs(sinoRa,Angs)
recRaC = Recs(sinoRaC,Angs)
recRz = Recs(sinoRz,Angs)
recRzC = Recs(sinoRzC,Angs)
recTRaRz = Recs(sinoTRaRz,Angs)
recTRaRzC = Recs(sinoTRaRzC,Angs)

"""Images"""
Images(recP,  Z = [10,30,50])
Images(recTC,  Z = [10,30,50])
Images(recRzC,  Z = [10,30,50])
Images(recRaC,  Z = [10,30,50])
Images(recTRaRzC,  Z = [10,30,50])


"""Plots"""
label = 'Translation Corrected'
Plots(recP,recTC,label)
 
label = 'Z-rotation Corrected'
Plots(recP,recRzC,label)

label = 'Precession Corrected'
Plots(recP,recRaC,label)

label = 'All Corrected'
Plots(recP,recTRaRzC,label)


"""Helical"""













"""Animations"""


vt.animated_gif(sinoRz, "sino_m", fps=128)



vt.animated_gif(sino, "sino", fps=48)
