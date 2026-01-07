#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:57:48 2025

@author: vargasp
"""


import matplotlib.pyplot as plt
import numpy as np

import vir.sys_mat.fp_2d_par as proj_2d


img = np.zeros((32, 32))
img[4:8, 4:8] = 1.0  # center impulse

angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
#angles = np.array([np.pi/4])
nDets = 64
dDet = .5

"""
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
"""

#Disatance to source iso cneter
DSO =500

#Det 2 source
DSD = 750

sino1f = proj_2d.dd_fp_fan_2d(img, angles, nDets, DSO, DSD, dPix=1.0, dDet=1.0)
plt.imshow(sino1f)

"""

dDet = .5
nDets = 64
nAngs = 32
r = 12.5

Dets = dDet*(np.arange(nDets) - nDets / 2.0 + 0.5)


proj = 2*np.sqrt((r**2 - Dets**2).clip(0))
sino = np.tile(proj, [nAngs, 1])

Angs = np.linspace(0, np.pi*2, 32, endpoint=False)
nX = 32
nY = 32

rec = proj_2d.dd_bp_2d(sino, Angs, nX, nY, dPix=1.0, dDet=dDet)

# plt.imshow(rec)

plt.plot(rec[:, 16])
plt.plot(rec[16, :])
plt.show()

plt.plot(rec[:, 16])
plt.plot(rec[:, 15])
plt.show()



"""