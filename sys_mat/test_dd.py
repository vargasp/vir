# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 20:27:09 2026

@author: varga
"""



import matplotlib.pyplot as plt
import numpy as np

import vir.sys_mat.dd as dd

#Fan Beam Geometry
DSO =1e8-1
DSD = 1e8

#Image params - Pixels
nX = 64
nY = 64
d_pix = 1

#Sino params 
nAngs = 64
nDets = 64
dDet = 1


angles = np.linspace(0, np.pi*2, nAngs, endpoint=False)
#Angs = np.array([45*np.pi/4.0])
Dets = dDet*(np.arange(nDets) - nDets / 2.0 + 0.5)

#Test image
img = np.zeros((nX, nY))
#img[4:28, 4:28] = 1.0  # center impulse

img[12:16, 12:16] = 1.0  # center impulse


#Test Sino
r = 10
x0 = 14
y0 = 14
sino = np.zeros((nAngs, nDets))
for i, theta in enumerate(angles):
    shift = x0 * np.cos(theta) + y0 * np.sin(theta)
    s = Dets - shift
    sino[i] = 2 * np.sqrt((r**2 - s**2).clip(0))


sino1p = dd.dd_fp_par_2d(img, angles, nDets, d_pix=d_pix, d_det=dDet)
sino1f = dd.dd_fp_fan_2d(img, angles, nDets, DSO, DSD, d_pix=d_pix, d_det=dDet)

plt.figure(figsize=(4,2))
plt.subplot(1,2,1)
plt.imshow(sino1p, cmap='gray', aspect='auto')
plt.title("DD Parallel")

plt.subplot(1,2,2)
plt.imshow(sino1f, cmap='gray', aspect='auto')
plt.title("DD Fanbean")

plt.tight_layout()
plt.show()

sino1p = dd.dd_fp_par_2d(img, angles, nDets, d_pix=d_pix, d_det=dDet)
sino1f = dd.dd_fp_fan_2d(img, angles, nDets, DSO, DSD, d_pix=d_pix, d_det=dDet)

plt.figure(figsize=(4,2))
plt.subplot(1,2,1)
plt.imshow(sino1p, cmap='gray', aspect='auto')
plt.title("DD Parallel")

plt.subplot(1,2,2)
plt.imshow(sino1f, cmap='gray', aspect='auto')
plt.title("DD Fanbean")

plt.tight_layout()
plt.show()





print("Par Sino Max:", sino1p.max())
print("Fan Sino Max:", sino1f.max())



sino = np.zeros((1, nDets))
sino[0,10:-10] = 1
angles = np.array([10*np.pi/180])

rec1p = dd.dd_bp_par_2d(sino, angles, nX, nY, d_pix=d_pix, d_det=dDet)
rec1f = dd.dd_bp_fan_2d(sino, angles, nX, nY, DSO, DSD, d_pix=d_pix, d_det=dDet)
# plt.imshow(rec)

print("Par Rec Max:", rec1p.max())
print("Fan Rec Max:", rec1f.max())

plt.figure(figsize=(4,2))
plt.subplot(1,2,1)
plt.imshow(rec1p.clip(.98,1), cmap='gray', aspect='auto', origin='lower')
plt.title("DD Parallel")


plt.subplot(1,2,2)
plt.imshow(rec1f.clip(.90,1), cmap='gray', aspect='auto', origin='lower')
plt.title("DD Fanbean")



#plt.plot(rec1p[20,:])
#plt.plot(rec1f[20,:])



