#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 14:02:51 2026

@author: pvargas21
"""


import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 600
import numpy as np

import vir.sys_mat.dd as dd
import vir.sys_mat.rd as rd
import vir.sys_mat.pd as pd


#Image params - Pixels
nx = 33
ny = 33
nz = 33
d_pix = 1


#Fan Beam Geometry - Parallel
DSO = 1e8
DSD = 1e8 + max(nx,ny)/2

#Fan Beam Geometry - Parallel
DSO = max(nx,ny)*np.sqrt(2)/2 
DSD = DSO*2



#Sino params 
na = 8
nu = 33
nv = 33
du = 1
dv = 1
su = 0.0
sv = 0.0
x = 13
y = 16
z = 23
ang_arr = np.linspace(0, np.pi/2, na, endpoint=False)#, dtype=np.float32)
u_arr = du*(np.arange(nu) - nu/2.0 + 0.5 + su)

#Test image
img3d = np.zeros((nx, ny, nz), dtype=np.float32)
img3d[x, y, z] = 1.0  # center impulse




sino1c = dd.dd_fp_cone_3d(img3d, ang_arr, nu, nv, DSO, DSD, du=du, dv=dv,su=su, sv=sv, d_pix=1.0)
sino2c = rd.aw_fp_cone_3d(img3d, ang_arr, nu, nv, DSO, DSD, du=du, d_pix=d_pix)
sino3c = rd.aw_fp_cone_3d(img3d, ang_arr, nu, nv, DSO, DSD, du=du, d_pix=d_pix,joseph=True)
sino4c = pd.pd_fp_cone_3d(img3d, ang_arr, nu, nv, DSO, DSD, du=du, dv=dv,su=su, sv=sv,d_pix=1.0)
sinos = [sino1c,sino2c,sino3c,sino4c]

plt.figure(figsize=(20,8))
plt.subplot(2,6,1)
plt.imshow(img3d[:,:,z].T, cmap='gray', origin='lower')
plt.title("XY Plane")
plt.xlabel("X Pixels")
plt.ylabel("Y Pixels")

plt.subplot(2,5,2)
plt.imshow(sino1c[0,:,:].T, cmap='gray', origin='lower')
plt.title("DD Angle 0")

plt.subplot(2,5,3)
plt.imshow(sino1c[0,:,:].T, cmap='gray', origin='lower')
plt.title("AW Angle 0")

plt.subplot(2,5,4)
plt.imshow(sino1c[0,:,:].T, cmap='gray', origin='lower')
plt.title("JO Angle 0")

plt.subplot(2,5,5)
plt.imshow(sino1c[0,:,:].T, cmap='gray', origin='lower')
plt.title("PD Angle 0")

plt.subplot(2,5,6)
plt.imshow(img3d[:,y,:].T, cmap='gray', origin='lower')
plt.title("XZ Plane")
plt.xlabel("X Pixels")
plt.ylabel("Z Pixels")

plt.subplot(2,5,7)
plt.imshow(sino1c[2,:,:].T, cmap='gray', origin='lower')
plt.title("DD Angle 90")

plt.subplot(2,5,8)
plt.imshow(sino1c[2,:,:].T, cmap='gray', origin='lower')
plt.title("AW Angle 90")

plt.subplot(2,5,9)
plt.imshow(sino1c[2,:,:].T, cmap='gray', origin='lower')
plt.title("JO Angle 90")

plt.subplot(2,5,10)
plt.imshow(sino1c[2,:,:].T, cmap='gray', origin='lower')
plt.title("PD Angle 90")
plt.show()

"""

lt.plot(sino2c[,:,16], label='AW')
plt.plot(sino3c[,:,16], label='JO')
plt.plot(sino4c[,:,16], label='PD')
plt.legend()





plt.figure(figsize=(16,8))
fractions = [0, 1/4, 2/4, 3/4]
for a, fraction in enumerate(fractions):
    plt.subplot(2,len(fractions),a+1)
    plt.plot(sino1c[a*2,:,16], label='DD')
    plt.plot(sino2c[a*2,:,16], label='AW')
    plt.plot(sino3c[a*2,:,16], label='JO')
    plt.plot(sino4c[a*2,:,16], label='PD')
    plt.legend()
    plt.title("u det: Angle "+ str(int(fraction*360))+" profile")

for a, fraction in enumerate(fractions):
    plt.subplot(2,len(fractions),a+5)
    plt.plot(sino1c[a*2,16,:], label='DD')
    plt.plot(sino2c[a*2,16,:], label='AW')
    plt.plot(sino3c[a*2,16,:], label='JO')
    plt.plot(sino4c[a*2,16,:], label='PD')
    plt.legend()
    plt.title("v det: Angle "+ str(int(fraction*360))+" profile")
plt.show()



plt.figure(figsize=(16,8))
fractions = [1/8, 3/8, 5/8, 7/8]
for a, fraction in enumerate(fractions):
    plt.subplot(2,len(fractions),a+1)
    plt.plot(sino1c[a*2+1,12:20,16], label='DD')
    plt.plot(sino2c[a*2+1,12:20,16], label='AW')
    plt.plot(sino3c[a*2+1,12:20,16], label='JO')
    plt.plot(sino4c[a*2+1,12:20,16], label='PD')
    plt.legend()
    plt.title("u det: Angle "+ str(int(fraction*360))+" profile")

for a, fraction in enumerate(fractions):
    plt.subplot(2,len(fractions),a+5)
    plt.plot(sino1c[a*2+1,16,:], label='DD')
    plt.plot(sino2c[a*2+1,16,:], label='AW')
    plt.plot(sino3c[a*2+1,16,:], label='JO')
    plt.plot(sino4c[a*2+1,16,:].clip(13.25), label='PD')
    plt.legend()
    plt.title("v det: Angle "+ str(int(fraction*360))+" profile")
plt.show()







titles = ["DD Conebeam","SD Conebeam","JO Conebeam","PD Conebeam"]
fractions = [0, 1/4, 2/4, 3/4]
plt.figure(figsize=(16,16))
for i, (sino,title) in enumerate(zip(sinos,titles)):
    for a in range(4):
        plt.subplot(4,4,i*4+a+1)
        plt.imshow(sino[a*2,:,:].T, cmap='gray', aspect='auto', origin='lower')
        
        title2 = " Angle "+ str(int(fractions[a]*360))
        plt.title(titles[i]+title2)
plt.show()







titles = ["DD Conebeam","SD Conebeam","JO Conebeam","PD Conebeam"]
fractions = [0, 1/4, 2/4, 3/4]
plt.figure(figsize=(16,16))
for i, (sino,title) in enumerate(zip(sinos,titles)):
    for a in range(4):
        plt.subplot(4,4,i*4+a+1)
        plt.imshow(sino[a*2,:,:].T, cmap='gray', aspect='auto', origin='lower')
        
        title2 = " Angle "+ str(int(fractions[a]*360))
        plt.title(titles[i]+title2)
plt.show()


titles = ["DD Conebeam","SD Conebeam","JO Conebeam","PD Conebeam"]
fractions = [1/8, 3/8, 5/8, 7/8]
plt.figure(figsize=(16,16))
for i, (sino,title) in enumerate(zip(sinos,titles)):
    for a in range(4):
        plt.subplot(4,4,i*4+a+1)
        plt.imshow(sino[a*2+1,:,:].T, cmap='gray', aspect='auto', origin='lower')
        
        title2 = " Angle "+ str(int(fractions[a]*360))
        plt.title(titles[i]+title2)
plt.show()


plt.figure(figsize=(16,8))
fractions = [0, 1/4, 2/4, 3/4]
for a, fraction in enumerate(fractions):
    plt.subplot(2,len(fractions),a+1)
    plt.plot(sino1c[a*2,:,12], label='DD')
    plt.plot(sino2c[a*2,:,12], label='AW')
    plt.plot(sino3c[a*2,:,12], label='JO')
    plt.plot(sino4c[a*2,:,12], label='PD')
    plt.legend()
    plt.title("Angle "+ str(int(fraction*360))+" profile")

fractions = [1/8, 3/8, 5/8, 7/8]
for a, fraction in enumerate(fractions):
    plt.subplot(2,len(fractions),a+5)
    plt.plot(sino1c[a*2+1,:,12], label='DD')
    plt.plot(sino2c[a*2+1,:,12], label='AW')
    plt.plot(sino3c[a*2+1,:,12], label='JO')
    plt.plot(sino4c[a*2+1,:,12], label='PD')
    plt.legend()
    plt.title("Angle "+ str(int(fraction*360))+" profile")
plt.show()


"""