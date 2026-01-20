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
nX = 32
nY = 32
d_pix = 1

#Sino params 
na = 32
n_dets = 64
d_det = .5
su = 0.0


angles = np.linspace(0, np.pi*2, na, endpoint=False)#, dtype=np.float32)
Dets = d_det*(np.arange(n_dets) - n_dets / 2.0 + 0.5)

#Test image
img = np.zeros((nX, nY), dtype=np.float32)
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



#Test Sino
r = 10
x0 = 14
y0 = 14
sino = np.zeros((nAngs, n_dets))
for i, theta in enumerate(angles):
    shift = x0 * np.cos(theta) + y0 * np.sin(theta)
    s = Dets - shift
    sino[i] = 2 * np.sqrt((r**2 - s**2).clip(0))



sino1 = proj_2d.siddons_fp_2d(img, angles, n_dets, d_det=d_det, d_pix=d_pix)
sino2 = jp.joseph_fp_2d( img, angles, n_dets, d_det=d_det, d_pix=d_pix)
sino3 = dd.dd_fp_par_2d(      img, angles, n_dets, d_det=d_det, d_pix=d_pix)
sino4 = aw.aw_fp_2d(     img, angles, n_dets, d_det=d_det, d_pix=d_pix)



plt.figure(figsize=(6,6))
plt.subplot(2,2,1)
plt.imshow(sino1, cmap='gray', aspect='auto', origin='lower')
plt.title("Siddons")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(2,2,2)
plt.imshow(sino2, cmap='gray', aspect='auto', origin='lower')
plt.title("Joseph")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(2,2,3)
plt.imshow(sino3, cmap='gray', aspect='auto', origin='lower')
plt.title("Distance-Driven")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(2,2,4)
plt.imshow(sino4, cmap='gray', aspect='auto', origin='lower')
plt.title("AW Siddons")
plt.xlabel("Detector bin")
plt.ylabel("Angle")
plt.tight_layout()
plt.show()


plt.subplot(1,1,1)
plt.plot(sino1[0,:], label='Siddon Parallel')
plt.plot(sino2[0,:], label='Joseph Parallel')
plt.plot(sino3[0,:], label='DD Parallel')
plt.plot(sino4[0,:], label='AW Parallel')
plt.title("Angle 0 profile")
plt.xlabel("Detector Bin")
plt.ylabel("Intensity")
plt.legend()
plt.show()


print("Siddons/AW Diff:", (sino1-sino4).max())
print("Siddons/Joe Diff:", (sino1-sino2).max())
print("Siddons/DD Diff:", (sino1-sino3).max())

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



"""

print(sino2f[0,32:])



plt.subplot(1,1,1)
plt.plot(sino1p[:,32])
plt.plot(sino1f[:,32])
plt.plot(sino2p[:,32])
plt.plot(sino2f[:,32])
plt.show()

plt.subplot(1,1,1)
plt.plot(sino1p[:,31])
plt.plot(sino1f[:,31])
plt.plot(sino2p[:,31])
plt.plot(sino2f[:,31])
plt.show()









rec1p = dd.dd_bp_par_2d(sino, Angs, nX, nY, d_pix=d_pix, d_det=dDet)
rec1f = dd.dd_bp_fan_2d(sino, Angs, nX, nY, DSO, DSD, d_pix=d_pix, d_det=dDet)
# plt.imshow(rec)

print("Par Max:", rec1p.max())
print("Fan Max:", rec1f.max())

plt.figure(figsize=(4,2))
plt.subplot(1,2,1)
plt.imshow(rec1p, cmap='gray', aspect='auto', origin='lower')
plt.title("DD Parallel")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(1,2,2)
plt.imshow(rec1f, cmap='gray', aspect='auto', origin='lower')
plt.title("DD Fanbean")
plt.xlabel("Detector bin")
plt.ylabel("Angle")


#plt.plot(rec1p[20,:])
#plt.plot(rec1f[20,:])


rec1p = dd.dd_bp_par_2d(sino, Angs, nX, nY, d_pix=d_pix, d_det=dDet)
rec1f = dd.dd_bp_fan_2d(sino, Angs, nX, nY, DSO, DSD, d_pix=d_pix, d_det=dDet)
# plt.imshow(rec)

print("Par Max:", rec1p.max())
print("Fan Max:", rec1f.max())

plt.figure(figsize=(4,2))
plt.subplot(1,2,1)
plt.imshow(rec1p, cmap='gray', aspect='auto', origin='lower')
plt.title("DD Parallel")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(1,2,2)
plt.imshow(rec1f, cmap='gray', aspect='auto', origin='lower')
plt.title("DD Fanbean")
plt.xlabel("Detector bin")
plt.ylabel("Angle")


rec2 = proj_2d.aw_bp_2d_fan_flat(sino, angles, (nX,nY), DSO, DSD, dPix=1.0, dDet=1.0)

"""