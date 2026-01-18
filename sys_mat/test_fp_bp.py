#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:57:48 2025

@author: vargasp
"""


import matplotlib.pyplot as plt
import numpy as np

import vir.sys_mat.dd as dd
import vir.sys_mat.aw as aw
import vir.sys_mat.joseph as jp
import vir.sys_mat.fp_2d_par as proj_2d



#Fan Beam Geometry
DSO = 1e8
DSD = 1e8+32



#Image params - Pixels
nX = 32
nY = 32
d_pix = 1

#Sino params 
nAngs = 32
n_dets = 32
d_det = 1

angles = np.linspace(0, np.pi*2, nAngs, endpoint=False)#, dtype=np.float32)
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

sino1p = dd.dd_fp_par_2d(img, angles, n_dets, du=d_det, d_pix=d_pix)
sino1f = dd.dd_fp_fan_2d(img, angles, n_dets, DSO, DSD, du=d_det, d_pix=d_pix)
sino2p = aw.aw_fp_par_2d(img, angles, n_dets, du=d_det, d_pix=d_pix)
sino2f = aw.aw_fp_fan_2d(img, angles, n_dets, DSO, DSD, du=d_det, d_pix=d_pix)
sino3p = jp.joseph_fp_2d     (img, angles, n_dets, d_det=d_det, d_pix=d_pix)
sino3f = jp.joseph_fp_fan_2d (img, angles, n_dets, DSO, DSD, d_det=d_det, d_pix=d_pix)


plt.figure(figsize=(6,4))
plt.subplot(2,3,1)
plt.imshow(sino1p, cmap='gray', aspect='auto', origin='lower')
plt.title("DD Parallel")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(2,3,2)
plt.imshow(sino1f, cmap='gray', aspect='auto', origin='lower')
plt.title("DD Fanbean")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(2,3,3)
plt.imshow(sino2p, cmap='gray', aspect='auto', origin='lower')
plt.title("AW Parallel")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(2,3,4)
plt.imshow(sino2f, cmap='gray', aspect='auto', origin='lower')
plt.title("AW Fanbean")
plt.xlabel("Detector bin")
plt.ylabel("Angle")
plt.tight_layout()


plt.subplot(2,3,5)
plt.imshow(sino3p, cmap='gray', aspect='auto', origin='lower')
plt.title("Joseph Parallel")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(2,3,6)
plt.imshow(sino3f, cmap='gray', aspect='auto', origin='lower')
plt.title("Jospeh Fanbean")
plt.xlabel("Detector bin")
plt.ylabel("Angle")
plt.tight_layout()

plt.show()




"""
plt.subplot(1,1,1)
plt.plot(sino1p[0,:], label='DD Parallel')
plt.plot(sino1f[0,:], label='DD Fanbeam')
plt.plot(sino2p[0,:], label='AW Parallel')
plt.plot(sino2f[0,:], label='AW Fanbeam')
plt.legend()
plt.title("Angle 0 profile")
plt.xlabel("Detector Bin")
plt.ylabel("Intensity")
plt.show()



plt.subplot(1,1,1)
plt.plot(sino1p[12,:], label='DD Parallel')
plt.plot(sino1f[12,:], label='DD Fanbeam')
plt.plot(sino2p[12,:], label='AW Parallel')
plt.plot(sino2f[12,:], label='AW Fanbeam')
plt.legend()
plt.title("Angle 45 profile")
plt.xlabel("Detector Bin")
plt.ylabel("Intensity")
plt.show()




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