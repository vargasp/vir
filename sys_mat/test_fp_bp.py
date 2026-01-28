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
nz = 32
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
nv = 32
du = 1
dv = 1
su = 0.0
sv = 0.0

ang_arr = np.linspace(0, np.pi*2, na, endpoint=False)#, dtype=np.float32)
u_arr = du*(np.arange(nu) - nu/2.0 + 0.5 + su)

#Test image
img3d = np.zeros((nx, ny, nz), dtype=np.float32)
img3d[11:13, 13:15,15:17] = 1.0  # center impulse
#img3d[10:22, 10:22, 10:22] = 1.0  # center impulse
#img3d[3:6, 3:6,3:6] = 1.0  # center impulse
#img3d[:] = 1.0  # center impulse

#img[30:, 10:22] = 1.0  # center impulse
#img[10:12, 15:17] = 1.0  # center impulse
#img[31:33, 31:33] = 1.0  # center impulse

img2d = img3d[:,:,int(nz/2)]

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




sino1c = dd.dd_fp_cone_3d(img3d, ang_arr, nu, nv,DSO, DSD, du=1.0, dv=1.0,su=0.0, sv=0.0, d_pix=1.0)

sino2c = rd.aw_fp_cone_3d(img3d, ang_arr, nu, nv, DSO, DSD, du=du, d_pix=d_pix).transpose(0,2,1)
sino3c = rd.aw_fp_cone_3d(img3d, ang_arr, nu, ny, DSO, DSD, du=du, d_pix=d_pix,joseph=True).transpose(0,2,1)
sino4c = pd.pd_fp_cone_3d(img3d, ang_arr, nu, nv, DSO, DSD, du=1.0, dv=1.0,su=0.0, sv=0.0,d_pix=1.0).transpose(0,2,1)




sino1p = dd.dd_fp_par_2d(img2d, ang_arr, nu, du=du, su=su, d_pix=d_pix)
sino1f = dd.dd_fp_fan_2d(img2d, ang_arr, nu, DSO, DSD, du=du, su=su, d_pix=d_pix)
sino2p = rd.aw_fp_par_2d(img2d, ang_arr, nu, du=du, su=su, d_pix=d_pix)
sino2f = rd.aw_fp_fan_2d(img2d, ang_arr, nu, DSO, DSD, du=du, su=su, d_pix=d_pix)
sino3p = rd.aw_fp_par_2d(img2d, ang_arr, nu, du=du, su=su, d_pix=d_pix,joseph=True)
sino3f = rd.aw_fp_fan_2d(img2d, ang_arr, nu, DSO, DSD, du=du, su=su, d_pix=d_pix,joseph=True)
sino4p = pd.pd_fp_par_2d(img2d, ang_arr, nu, du=du, su=su, d_pix=d_pix)
sino4f = pd.pd_fp_fan_2d(img2d, ang_arr, nu, DSO, DSD, du=du, su=su, d_pix=d_pix)







sino1c = dd.dd_fp_cone_3d(img3d, ang_arr, nu, nv,DSO, DSD, du=1.0, dv=1.0,su=0.0, sv=0.0, d_pix=1.0)


sinos = [sino1p,sino2p,sino3p,sino4p,
         sino1f,sino2f,sino3f,sino4f,
         sino1c[:,:,int(nz/2)],sino2c[:,:,int(nz/2)],sino3c[:,:,int(nz/2)],sino4c[:,:,int(nz/2)]]

    
    
titles = ["DD Parallel","SD Parallel","JO Parallel","PD Parallel",
          "DD Fanbeam","SD Fanbeam","JO Fanbeam","PD Fanbeam",
          "DD Conebeam","SD Conebeam","JO Conebeam","PD Conebeam"]
plt.figure(figsize=(16,12))
for i, (sino,title) in enumerate(zip(sinos,titles)):
    plt.subplot(3,4,i+1)
    plt.imshow(sino, cmap='gray', aspect='auto', origin='lower')
    plt.title(title)
    if i % 4 ==0: 
        plt.ylabel("Angle")
    if i > 7:
        plt.xlabel("Detector Bin")
plt.show()



fractions = [0, 1/8, 1/4, 3/8,1/2]
#fractions = [0, 1/16, 1/8, 3/16,1/4]
plt.figure(figsize=(20,8))
for i, fraction in enumerate(fractions):
    plt.subplot(2,len(fractions),i+1)
    plt.plot(sino1p[int(fraction*na),:], label='DD')
    plt.plot(sino2p[int(fraction*na),:], label='AW')
    plt.plot(sino3p[int(fraction*na),:], label='JO')
    plt.plot(sino4p[int(fraction*na),:], label='PD')
    plt.legend()
    plt.title("Angle "+ str(int(fraction*360))+" profile")
    if i == 0: plt.ylabel("Intensity")
    
for i, fraction in enumerate(fractions):
    plt.subplot(2,len(fractions),i+1+5)
    plt.plot(sino1f[int(fraction*na),:], label='DD')
    plt.plot(sino2f[int(fraction*na),:], label='AW')
    plt.plot(sino3f[int(fraction*na),:], label='JO')
    plt.plot(sino4f[int(fraction*na),:], label='PD')
    plt.xlabel("Detector Bin")
    plt.legend()
    if i == 0: plt.ylabel("Intensity")
plt.show()


fractions = [0, 1/8, 1/4, 3/8,1/2]
#fractions = [0, 1/16, 1/8, 3/16,1/4]
plt.figure(figsize=(20,4))
for i, fraction in enumerate(fractions):
    plt.subplot(1,len(fractions),i+1)
    plt.plot(sino1f[int(fraction*na),:], label='F DD')
    plt.plot(sino2f[int(fraction*na),:], label='F AW')
    plt.plot(sino3f[int(fraction*na),:], label='F JO')
    plt.plot(sino1c[int(fraction*na),:,int(nz/2)], label='C DD')
    plt.plot(sino2c[int(fraction*na),:,int(nz/2)], label='C AW')
    plt.plot(sino3c[int(fraction*na),:,int(nz/2)], label='C JO')
    plt.xlabel("Detector Bin")
    plt.legend()
    if i == 0: plt.ylabel("Intensity")
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



fractions = [0, 1/8, 1/4, 3/8,1/2]
#fractions = [0, 1/16, 1/8, 3/16,1/4]
plt.figure(figsize=(20,8))
for i, fraction in enumerate(fractions):
    plt.subplot(2,len(fractions),i+1)
    plt.plot(sino1p[int(fraction*na),:], label='DD')
    plt.plot(sino2p[int(fraction*na),:], label='AW')
    plt.plot(sino3p[int(fraction*na),:], label='JO')
    plt.plot(sino4p[int(fraction*na),:], label='PD')
    plt.legend()
    plt.title("Angle "+ str(int(fraction*360))+" profile")
    if i == 0: plt.ylabel("Intensity")
    
for i, fraction in enumerate(fractions):
    plt.subplot(2,len(fractions),i+1+5)
    plt.plot(sino1f[int(fraction*na),:], label='DD')
    plt.plot(sino2f[int(fraction*na),:], label='AW')
    plt.plot(sino3f[int(fraction*na),:], label='JO')
    plt.plot(sino4f[int(fraction*na),:], label='PD')
    plt.xlabel("Detector Bin")
    plt.legend()
    if i == 0: plt.ylabel("Intensity")
plt.show()



recsp = [rec1p,rec2p,rec3p,rec4p]
recsf = [rec1f,rec2f,rec3f,rec4f]

labels = ["DD","SD","JO","PD"]

titles = ["X Center", "Y Center", "XY Center", "YX Center"]
plt.figure(figsize=(20,8))
plt.subplot(2,4,i+1)
 
for j, rec in enumerate(recsp):
    plt.title(title)
    plt.plot(rec[:,int(ny/2-1):int(ny/2+1)].mean(axis=1), label=labels[j])
    plt.legend()

plt.show()



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