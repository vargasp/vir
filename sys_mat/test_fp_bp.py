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


#Image params - Pixels
nx = 32
ny = 32
d_pix = 1


#Fan Beam Geometry - Parallel
DSO = 1e8
DSD = 1e8 + max(nx,ny)/2

#Fan Beam Geometry - Parallel
DSO = max(nx,ny)*np.sqrt(2)/2 
DSD = DSO*2



#Sino params 
na = 64
nu = 32
du = 1
su = 0.0


ang_arr = np.linspace(0, np.pi*2, na, endpoint=False)#, dtype=np.float32)
u_arr = du*(np.arange(nu) - nu/2.0 + 0.5 + su)

#Test image
img = np.zeros((nx, ny), dtype=np.float32)
img[14:18, 14:18] = 1.0  # center impulse
#img[3:6, 3:6] = 1.0  # center impulse
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





sino1p = dd.dd_fp_par_2d(img, ang_arr, nu, du=du, su=su, d_pix=d_pix)
sino1f = dd.dd_fp_fan_2d(img, ang_arr, nu, DSO, DSD, du=du, su=su, d_pix=d_pix)
sino2p = rd.aw_fp_par_2d(img, ang_arr, nu, du=du, su=su, d_pix=d_pix)
sino2f = rd.aw_fp_fan_2d(img, ang_arr, nu, DSO, DSD, du=du, su=su, d_pix=d_pix)
sino3p = rd.aw_fp_par_2d(img, ang_arr, nu, du=du, su=su, d_pix=d_pix,joseph=True)
sino3f = rd.aw_fp_fan_2d(img, ang_arr, nu, DSO, DSD, du=du, su=su, d_pix=d_pix,joseph=True)
sino4p = pd.pd_fp_par_2d(img, ang_arr, nu, du=du, su=su, d_pix=d_pix)
sino4f = pd.pd_fp_fan_2d(img, ang_arr, nu, DSO, DSD, du=du, su=su, d_pix=d_pix)



sinos = [sino1p,sino2p,sino3p,sino4p,sino1f,sino2f,sino3f,sino4f]
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
    plt.plot(sino4f[int(fraction*na),:], label='PD Fanbeam')
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
#ang_arr = [ang_arr[0]]


rec1p = dd.dd_bp_par_2d(sino, ang_arr, (nx,ny), d_pix=d_pix, du=du,su=su)
rec1f = dd.dd_bp_fan_2d(sino, ang_arr, (nx,ny), DSO, DSD, d_pix=d_pix, du=du,su=su)
rec2p = rd.aw_bp_par_2d(sino, ang_arr, (nx,ny), d_pix=d_pix, du=du,su=su)
rec2f = rd.aw_bp_fan_2d(sino, ang_arr, (nx,ny), DSO, DSD, d_pix=d_pix, du=du,su=su)
rec3p = rd.aw_bp_par_2d(sino, ang_arr, (nx,ny), d_pix=d_pix, du=du,su=su,joseph=True)
rec3f = rd.aw_bp_fan_2d(sino, ang_arr, (nx,ny), DSO, DSD, d_pix=d_pix, du=du,su=su, joseph=True)
rec4p = dd.dd_bp_par_2d(sino, ang_arr, (nx,ny), d_pix=d_pix, du=du,su=su)
rec4f = dd.dd_bp_fan_2d(sino, ang_arr, (nx,ny), DSO, DSD, d_pix=d_pix, du=du,su=su)

rec4p[:,:] = 0.0
rec4f[:,:] = 0.0

#rec2p[:,:] = 0 
#rec2f[:,:] = 0 
#rec3p[:,:] = 0 
#rec3f[:,:] = 0 
#rec4p[:,:] = 0 
#rec4f[:,:] = 0 


recs = [rec1p,rec2p,rec3p,rec4p,rec1f,rec2f,rec3f,rec4f]
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
plt.subplot(1,4,1)
plt.title("X Center")
plt.plot(rec1p[:,15:17].mean(axis=1), label='DD P')
plt.plot(rec1f[:,15:17].mean(axis=1), label='DD F')
plt.plot(rec2p[:,15:17].mean(axis=1), label='SD P')
plt.plot(rec2f[:,15:17].mean(axis=1), label='SD F')
plt.plot(rec3p[:,15:17].mean(axis=1), label='JO P')
plt.plot(rec3f[:,15:17].mean(axis=1), label='JO F')
plt.legend()

plt.subplot(1,4,2)
plt.title("Y Center")
plt.plot(rec1p[15:17,:].mean(axis=0), label='DD P')
plt.plot(rec1f[15:17,:].mean(axis=0), label='DD F')
plt.plot(rec2p[15:17,:].mean(axis=0), label='SD P')
plt.plot(rec2f[15:17,:].mean(axis=0), label='SD F')
plt.plot(rec3p[15:17,:].mean(axis=0), label='JO P')
plt.plot(rec3f[15:17,:].mean(axis=0), label='JO F')

plt.subplot(1,4,3)
plt.title("XY Center")
plt.plot(rec1p[np.arange(32),np.arange(32)], label='DD P')
plt.plot(rec1f[np.arange(32),np.arange(32)], label='DD F')
plt.plot(rec2p[np.arange(32),np.arange(32)], label='SD P')
plt.plot(rec2f[np.arange(32),np.arange(32)], label='SD F')
plt.plot(rec3p[np.arange(32),np.arange(32)], label='JO P')
plt.plot(rec3f[np.arange(32),np.arange(32)], label='JO F')

plt.subplot(1,4,4)
plt.title("YX Center")
plt.plot(rec1p[np.arange(32), np.arange(32)[::-1]], label='DD P')
plt.plot(rec1f[np.arange(32), np.arange(32)[::-1]], label='DD F')
plt.plot(rec2p[np.arange(32), np.arange(32)[::-1]], label='SD P')
plt.plot(rec2f[np.arange(32), np.arange(32)[::-1]], label='SD F')
plt.plot(rec3p[np.arange(32), np.arange(32)[::-1]], label='JO P')
plt.plot(rec3f[np.arange(32), np.arange(32)[::-1]], label='JO F')
plt.show()


print("Par Max:", rec1p.max())
print("Fan Max:", rec1f.max())

"""