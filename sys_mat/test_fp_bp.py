#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:57:48 2025

@author: vargasp
"""


import matplotlib.pyplot as plt
import numpy as np

import vir.sys_mat.fp_2d_par as proj_2d

"""
img = np.zeros((32, 32))
#img[4:28, 4:28] = 1.0  # center impulse
img[4:8, 4:8] = 1.0  # center impulse

angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
#angles = np.array([45*np.pi/4.0])

#angles = np.array([np.pi/4])
nDets = 64
dDet = .5

sino1 = proj_2d.siddons_fp_2d(img, angles, nDets=nDets,dDet=dDet)
sino2 = proj_2d.joseph_fp_2d(img, angles, nDets=nDets,dDet=dDet)
sino3 = proj_2d.distance_driven_fp_2d(img, angles, nDets=nDets,dDet=dDet)
sino4 = proj_2d.separable_footprint_fp_2d(img, angles, nDets,dDet=dDet)

plt.figure(figsize=(6,6))
plt.subplot(2,2,1)
plt.imshow(sino1, cmap='gray', aspect='auto')
plt.title("Siddons")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(2,2,2)
plt.imshow(sino2, cmap='gray', aspect='auto')
plt.title("Joseph")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(2,2,3)
plt.imshow(sino3, cmap='gray', aspect='auto')
plt.title("Distance-Driven")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(2,2,4)
plt.imshow(sino4, cmap='gray', aspect='auto')
plt.title("Seperable Footprint")
plt.xlabel("Detector bin")
plt.ylabel("Angle")
plt.tight_layout()
plt.show()


#Disatance to source iso cneter
DSO =1e8-1

#Det 2 source
DSD = 1e8

sino1p = proj_2d.dd_fp_par_2d(img, angles, nDets, d_pix=1.0, d_det=1.0)
sino1f = proj_2d.dd_fp_fan_2d(img, angles, nDets, DSO, DSD, d_pix=1.0, d_det=1.0)

plt.figure(figsize=(4,2))
plt.subplot(1,2,1)
plt.imshow(sino1p, cmap='gray', aspect='auto')
plt.title("DD Parallel")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(1,2,2)
plt.imshow(sino1f, cmap='gray', aspect='auto')
plt.title("DD Fanbean")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

#plt.plot(sino1p[0,:])
#plt.plot(sino1f[0,:])


"""
#Disatance to source iso cneter
DSO = 1

#Det 2 source
DSD = 1

dDet = .5
nDets = 64
nAngs = 32
r = 2
x0 = 4
y0 =4
d_pix = 1.5
Dets = dDet*(np.arange(nDets) - nDets / 2.0 + 0.5)


Angs = np.linspace(0, np.pi*2, 32, endpoint=False)
nX = 32
nY = 32


sino = np.zeros((nAngs, nDets))

for i, theta in enumerate(Angs):
    shift = x0 * np.cos(theta) + y0 * np.sin(theta)
    s = Dets - shift
    sino[i] = 2 * np.sqrt((r**2 - s**2).clip(0))



rec1p = proj_2d.dd_bp_2d(sino, Angs, nX, nY, d_pix=d_pix, d_det=dDet)
rec1f = proj_2d.dd_bp_fan_2d(sino, Angs, nX, nY, DSO, DSD, d_pix=d_pix, d_det=dDet)
# plt.imshow(rec)

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



