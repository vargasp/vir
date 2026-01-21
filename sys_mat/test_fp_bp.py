#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:57:48 2025

@author: vargasp
"""


import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 600
import numpy as np

import vir.sys_mat.dd as dd
import vir.sys_mat.rd as rd
import vir.sys_mat.pd as pd



#Fan Beam Geometry
DSO = 1e8
DSD = 1e8+32



#Image params - Pixels
nx = 32
ny = 32
d_pix = 1

#Sino params 
na = 32
nu = 64
du = 1
su = 0.0


ang_arr = np.linspace(0, np.pi*2, na, endpoint=False)#, dtype=np.float32)
u_arr = du*(np.arange(nu) - nu/2.0 + 0.5 + su)

#Test image
img = np.zeros((nx, ny), dtype=np.float32)
#img[14:18, 14:18] = 1.0  # center impulse
img[4:8, 4:8] = 1.0  # center impulse
#img[:] = 1.0  # center impulse

"""
plt.figure(figsize=(3,3))
plt.subplot(1,1,1)
plt.imshow(img.T, cmap='gray', aspect='auto', origin='lower')
plt.title("Image Phantom")
plt.xlabel("X Pixels")
plt.ylabel("Y Pixels")

"""

#Test Sino
r = 4
x0 = 0
y0 = 0
sino = np.zeros((na, nu))
for ia, ang in enumerate(ang_arr):
    shift = x0 * np.cos(ang) + y0 * np.sin(ang)
    s = u_arr - shift
    sino[ia,:] = 2 * np.sqrt((r**2 - s**2).clip(0))

plt.imshow(sino, cmap='gray', aspect='auto', origin='lower')


"""

sino1p = dd.dd_fp_par_2d(img, angles, n_dets, du=d_det, su=su, d_pix=d_pix)
sino1f = dd.dd_fp_fan_2d(img, angles, n_dets, DSO, DSD, du=d_det, su=su, d_pix=d_pix)
sino2p = rd.aw_fp_par_2d(img, angles, n_dets, du=d_det, su=su, d_pix=d_pix)
sino2f = rd.aw_fp_fan_2d(img, angles, n_dets, DSO, DSD, du=d_det, su=su, d_pix=d_pix)
sino3p = rd.aw_fp_par_2d(img, angles, n_dets, du=d_det, su=su, d_pix=d_pix,joseph=True)
sino3f = rd.aw_fp_fan_2d(img, angles, n_dets, DSO, DSD, du=d_det, su=su, d_pix=d_pix,joseph=True)
sino4p = pd.pd_fp_par_2d(img, angles, n_dets, du=d_det, su=su, d_pix=d_pix)
sino4f = pd.pd_fp_par_2d(img, angles, n_dets, du=d_det, su=su, d_pix=d_pix)



sinos = [sino1p,sino1f,sino2p,sino2f,sino3p,sino3f,sino4p,sino4f]
titles = ["DD Parallel","SD Parallel","JO Parallel","PD Parallel",
          "DD Fanbeam","SD Fanbeam","JO Fanbeam","PD Fanbeam"]
plt.figure(figsize=(16,8))
for i, (sino,title) in enumerate(zip(sinos,titles)):
    plt.subplot(2,4,i+1)
    plt.imshow(sino, cmap='gray', aspect='auto', origin='lower')
    plt.title(title)
    if i % 4 ==0: 
        plt.ylabel("Angle")
    if i > 3:
        plt.xlabel("Detector Bin")
 
plt.show()



fractions = [0, 1/8, 1/4, 3/8,1/2]
plt.figure(figsize=(20,4))
for i, fraction in enumerate(fractions):
    plt.subplot(1,len(fractions),i+1)
    plt.plot(sino1p[int(fraction*na),:], label='DD Parallel')
    plt.plot(sino1f[int(fraction*na),:], label='DD Fanbeam')
    plt.plot(sino2p[int(fraction*na),:], label='AW Parallel')
    plt.plot(sino2f[int(fraction*na),:], label='AW Fanbeam')
    plt.plot(sino3p[int(fraction*na),:], label='JO Parallel')
    plt.plot(sino3f[int(fraction*na),:], label='JO Fanbeam')
    plt.plot(sino4p[int(fraction*na),:], label='PD Parallel')
    #plt.legend()
    plt.title("Angle "+ str(int(fraction*360))+" profile")
    plt.xlabel("Detector Bin")
    plt.ylabel("Intensity")
plt.show()


#print("Siddons/AW Diff:", (sino1-sino4).max())
#print("Siddons/Joe Diff:", (sino1-sino2).max())
#print("Siddons/DD Diff:", (sino1-sino3).max())



"""




#sino = sino[0:1,:]




rec1p = dd.dd_bp_par_2d(sino, ang_arr, (nx,ny), d_pix=d_pix, du=du)
rec1f = dd.dd_bp_fan_2d(sino, ang_arr, (nx,ny), DSO, DSD, d_pix=d_pix, du=du)
rec2p = dd.dd_bp_par_2d(sino, ang_arr, (nx,ny), d_pix=d_pix, du=du)
rec2f = dd.dd_bp_fan_2d(sino, ang_arr, (nx,ny), DSO, DSD, d_pix=d_pix, du=du)
rec3p = dd.dd_bp_par_2d(sino, ang_arr, (nx,ny), d_pix=d_pix, du=du)
rec3f = dd.dd_bp_fan_2d(sino, ang_arr, (nx,ny), DSO, DSD, d_pix=d_pix, du=du)
rec4p = dd.dd_bp_par_2d(sino, ang_arr, (nx,ny), d_pix=d_pix, du=du)
rec4f = dd.dd_bp_fan_2d(sino, ang_arr, (nx,ny), DSO, DSD, d_pix=d_pix, du=du)


recs = [rec1p,rec1f,rec2p,rec2f,rec3p,rec3f,rec4p,rec4f]
titles = ["DD Parallel","SD Parallel","JO Parallel","PD Parallel",
          "DD Fanbeam","SD Fanbeam","JO Fanbeam","PD Fanbeam"]
plt.figure(figsize=(16,8))
for i, (rec,title) in enumerate(zip(recs,titles)):
    plt.subplot(2,4,i+1)
    plt.imshow(rec, cmap='gray', aspect='auto', origin='lower')
    plt.title(title)
    if i % 4 ==0: 
        plt.ylabel("Pixels")
    if i > 3:
        plt.xlabel("Pixels")
 
plt.show()



plt.figure(figsize=(20,4))
    plt.subplot(1,len(fractions),i+1)

plt.plot(rec1p[15:17,:].mean(axis=0)

         

"""

fractions = [0, 1/8, 1/4, 3/8,1/2]
plt.figure(figsize=(20,4))
for i, fraction in enumerate(fractions):
    plt.subplot(1,len(fractions),i+1)
    plt.plot(sino1p[int(fraction*na),:], label='DD Parallel')
    plt.plot(sino1f[int(fraction*na),:], label='DD Fanbeam')
    plt.plot(sino2p[int(fraction*na),:], label='AW Parallel')
    plt.plot(sino2f[int(fraction*na),:], label='AW Fanbeam')
    plt.plot(sino3p[int(fraction*na),:], label='JO Parallel')
    plt.plot(sino3f[int(fraction*na),:], label='JO Fanbeam')
    plt.plot(sino4p[int(fraction*na),:], label='PD Parallel')
    #plt.legend()
    plt.title("Angle "+ str(int(fraction*360))+" profile")
    plt.xlabel("Detector Bin")
    plt.ylabel("Intensity")
plt.show()

"""



print("Par Max:", rec1p.max())
print("Fan Max:", rec1f.max())

