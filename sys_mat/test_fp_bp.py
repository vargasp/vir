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
import vir.sys_mat.analytic_sino as asino


#Image params - Pixels
nx, ny, nz = 32, 32, 32
d_pix = 1

#Fan Beam Geometry - Parallel
DSO = 1e8
DSD = 1e8 + max(nx,ny)/2

#Fan Beam Geometry - Parallel
#DSO = max(nx,ny)*np.sqrt(2)/2 
#DSD = DSO*2

#Sino params 
na = 128
nu, nv = 64, 64
du, dv = 1., 1.
su, sv = 0., 0.
na_lets, nu_lets, nv_lets = 5, 5, 5


ang_arr = np.linspace(0, np.pi*2, na, endpoint=False)#, dtype=np.float32)
ang_arr_lets = np.linspace(0, np.pi*2, na*na_lets, endpoint=False).reshape(na,na_lets)#, dtype=np.float32)
ang_arr_lets - ang_arr_lets[0,2]

u_arr = du*(np.arange(nu) - nu/2.0 + 0.5 + su)
v_arr = dv*(np.arange(nv) - nv/2.0 + 0.5 + sv)

u_arr_lets = du*(np.arange(nu*nu_lets) - nu/2.0*nu_lets + 0.5 + su).reshape(nu,nu_lets)/nu_lets
v_arr_lets = dv*(np.arange(nv*nv_lets) - nv/2.0*nv_lets + 0.5 + sv).reshape(nv,nv_lets)/nv_lets


#Phantom Paramters Sino
r = 2
x0 = 5
y0 = 0
z0 = 0

#Create analytic models
img3d = asino.phantom((x0,y0,z0,r),nx,ny,nz,upsample=5)
img2d = img3d[:,:,int(nz/2)]

sinoPi = asino.analytic_circle_sino_par_2d((x0,y0,r,1),ang_arr,u_arr)
sinoP = asino.analytic_circle_sino_par_2d((x0,y0,r,1),ang_arr_lets,u_arr_lets).mean(3).mean(1)
sinoFi = asino.analytic_circle_sino_fan_2d((x0,y0,r,1), ang_arr, u_arr, DSO, DSD)
sinoF = asino.analytic_circle_sino_fan_2d((x0,y0,r,1), ang_arr_lets, u_arr_lets, DSO, DSD).mean(3).mean(1)
sinoC = asino.analytic_sphere_sino_cone_3d((x0,y0,z0,r,1), ang_arr, u_arr, v_arr,DSO, DSD)


"""
plt.figure(figsize=(4,4))
plt.subplot(1,1,1)
plt.imshow((img3d.transpose([1,0,2]))[:,:,int(nz/2)], cmap='gray', aspect='auto', origin='lower')
#plt.imshow((img3d.transpose([1,0,2]))[:,:,21], cmap='gray', aspect='auto', origin='lower')
plt.title("Image Phantom")
plt.xlabel("X Pixels")
plt.ylabel("Y Pixels")
plt.show()
"""


sino1p = dd.dd_fp_par_2d(img2d, ang_arr, nu, du=du, su=su, d_pix=d_pix)
sino2p = rd.aw_fp_par_2d(img2d, ang_arr, nu, du=du, su=su, d_pix=d_pix)
sino3p = rd.aw_fp_par_2d(img2d, ang_arr, nu, du=du, su=su, d_pix=d_pix,joseph=True)
sino4p = pd.pd_fp_par_2d(img2d, ang_arr, nu, du=du, su=su, d_pix=d_pix)

sino1f = dd.dd_fp_fan_2d(img2d, ang_arr, nu, DSO, DSD, du=du, su=su, d_pix=d_pix)
sino2f = rd.aw_fp_fan_2d(img2d, ang_arr, nu, DSO, DSD, du=du, su=su, d_pix=d_pix)
sino3f = rd.aw_fp_fan_2d(img2d, ang_arr, nu, DSO, DSD, du=du, su=su, d_pix=d_pix,joseph=True)
sino4f = pd.pd_fp_fan_2d(img2d, ang_arr, nu, DSO, DSD, du=du, su=su, d_pix=d_pix)

sino1c = dd.dd_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=1.0)
sino2c = rd.aw_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,d_pix=d_pix)
sino3c = rd.aw_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,d_pix=d_pix,joseph=True)
sino4c = pd.pd_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=1.0)



sinos = [sino1p,sino2p,sino3p,sino4p,
         sino1f,sino2f,sino3f,sino4f,
         sino1c[:,:,int(nv/2)],sino2c[:,:,int(nv/2)],sino3c[:,:,int(nv/2)],sino4c[:,:,int(nv/2)]]

    
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


clip_percent=0
fractions = [0, 1/8, 1/4, 3/8,1/2]
#fractions = [0, 1/16, 1/8, 3/16,1/4]
plt.figure(figsize=(20,8))
for i, fraction in enumerate(fractions):
    plt.subplot(2,len(fractions),i+1)
    plt.plot(sino1p[int(fraction*na),:].clip(2*r*clip_percent), label='DD')
    plt.plot(sino2p[int(fraction*na),:].clip(2*r*clip_percent), label='AW')
    plt.plot(sino3p[int(fraction*na),:].clip(2*r*clip_percent), label='JO')
    plt.plot(sino4p[int(fraction*na),:].clip(2*r*clip_percent), label='PD')
    plt.legend()
    plt.title("Angle "+ str(int(fraction*360))+" profile - Parallel")
    if i == 0: plt.ylabel("Intensity")
    
for i, fraction in enumerate(fractions):
    plt.subplot(2,len(fractions),i+1+5)
    plt.plot(sino1f[int(fraction*na),:].clip(2*r*clip_percent), label='DD')
    plt.plot(sino2f[int(fraction*na),:].clip(2*r*clip_percent), label='AW')
    plt.plot(sino3f[int(fraction*na),:].clip(2*r*clip_percent), label='JO')
    plt.plot(sino4f[int(fraction*na),:].clip(2*r*clip_percent), label='PD')
    plt.xlabel("Detector Bin")
    plt.legend()
    plt.title("Angle "+ str(int(fraction*360))+" profile - Fanbeam")
    if i == 0: plt.ylabel("Intensity")
plt.show()



fractions = [0, 1/8, 1/4, 3/8,1/2]
#fractions = [0, 1/16, 1/8, 3/16,1/4]
plt.figure(figsize=(20,8))
for i, fraction in enumerate(fractions):
    plt.subplot(1,len(fractions),i+1)
    plt.plot(sino1c[int(fraction*na),:,int(nv/2+z0)], label='C DD')
    plt.plot(sino2c[int(fraction*na),:,int(nv/2+z0)], label='C AW')
    plt.plot(sino3c[int(fraction*na),:,int(nv/2+z0)], label='C JO')
    plt.xlabel("Detector Bin")
    plt.legend()
    if i == 0: plt.ylabel("Intensity")

for i, fraction in enumerate(fractions):
    plt.subplot(1,len(fractions),i+1)
    plt.plot(sino1c[int(fraction*na),int(nv/2+z0),:], label='C DD')
    plt.plot(sino2c[int(fraction*na),int(nv/2+z0),:], label='C AW')
    plt.plot(sino3c[int(fraction*na),int(nv/2+z0),:], label='C JO')
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
    plt.plot(sino1c[int(fraction*na),:,int(nv/2)], label='C DD')
    plt.plot(sino2c[int(fraction*na),:,int(nv/2)], label='C AW')
    plt.plot(sino3c[int(fraction*na),:,int(nv/2)], label='C JO')
    plt.xlabel("Detector Bin")
    plt.legend()
    if i == 0: plt.ylabel("Intensity")
plt.show()





"""



#print("Siddons/AW Diff:", (sino1-sino4).max())
#print("Siddons/Joe Diff:", (sino1-sino2).max())
#print("Siddons/DD Diff:", (sino1-sino3).max())






#sino = sino[8:9,:]
#ang_arr = [ang_arr[8]]


rec1p = dd.dd_bp_par_2d(sinoP, ang_arr, (nx,ny), d_pix=d_pix, du=du,su=su)
rec1f = dd.dd_bp_fan_2d(sinoF, ang_arr, (nx,ny), DSO, DSD, d_pix=d_pix, du=du,su=su)
rec2p = rd.aw_bp_par_2d(sinoP, ang_arr, (nx,ny), d_pix=d_pix, du=du,su=su)
rec2f = rd.aw_bp_fan_2d(sinoF, ang_arr, (nx,ny), DSO, DSD, d_pix=d_pix, du=du,su=su)
rec3p = rd.aw_bp_par_2d(sinoP, ang_arr, (nx,ny), d_pix=d_pix, du=du,su=su,joseph=True)
rec3f = rd.aw_bp_fan_2d(sinoF, ang_arr, (nx,ny), DSO, DSD, d_pix=d_pix, du=du,su=su, joseph=True)
rec4p = dd.dd_bp_par_2d(sinoP, ang_arr, (nx,ny), d_pix=d_pix, du=du,su=su)
rec4f = dd.dd_bp_fan_2d(sinoF, ang_arr, (nx,ny), DSO, DSD, d_pix=d_pix, du=du,su=su)

rec4p[:,:] = 0.0
rec4f[:,:] = 0.0



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


recsp = [rec1p,rec2p,rec3p,rec4p]
recsf = [rec1f,rec2f,rec3f,rec4f]
labels = ["DD","SD","JO","PD"]
titles = ["P - X Center", "P - Y Center", "P - XY Center", "P  -YX Center"]
plt.figure(figsize=(16,8))
plt.subplot(2,4,1)
for j, rec in enumerate(recsp):
    plt.plot(rec[:,int(ny/2-1):int(ny/2+1)].mean(axis=1), label=labels[j])
plt.title(titles[0])
plt.legend()

plt.subplot(2,4,2)
for j, rec in enumerate(recsp):
    plt.plot(rec[int(nx/2-1):int(nx/2+1),:].mean(axis=0), label=labels[j])
plt.title(titles[1])
plt.legend()

plt.subplot(2,4,3)
for j, rec in enumerate(recsp):
    plt.plot(rec[np.arange(32),np.arange(32)], label=labels[j])
plt.title(titles[2])
plt.legend()

plt.subplot(2,4,4)
for j, rec in enumerate(recsp):
    plt.plot(rec[np.arange(32), np.arange(32)[::-1]], label=labels[j])
plt.title(titles[2])
plt.legend()


titles = ["F - X Center", "F - Y Center", "F - XY Center", "F - YX Center"]
plt.subplot(2,4,5)
for j, rec in enumerate(recsf):
    plt.plot(rec[:,int(ny/2-1):int(ny/2+1)].mean(axis=1), label=labels[j])
plt.title(titles[0])
plt.legend()

plt.subplot(2,4,6)
for j, rec in enumerate(recsf):
    plt.plot(rec[int(nx/2-1):int(nx/2+1),:].mean(axis=0), label=labels[j])
plt.title(titles[1])
plt.legend()

plt.subplot(2,4,7)
for j, rec in enumerate(recsf):
    plt.plot(rec[np.arange(32),np.arange(32)], label=labels[j])
plt.title(titles[2])
plt.legend()

plt.subplot(2,4,8)
for j, rec in enumerate(recsf):
    plt.plot(rec[np.arange(32), np.arange(32)[::-1]], label=labels[j])
plt.title(titles[2])
plt.legend()
plt.show()


"""

