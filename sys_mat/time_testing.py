#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 20:07:24 2026

@author: pvargas21
"""


from sys import path

path.append('/Users/pvargas21/Codebase/Libraries')


import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 600
import numpy as np
import time

import vir.sys_mat.dd as dd
import vir.sys_mat.rd as rd
import vir.sys_mat.pd as pd
import vir.sys_mat.analytic_sino as asino


#Image params - Pixels
nx, ny, nz = 64, 64, 64
d_pix = 1

#Fan Beam Geometry - Parallel
DSO = 1e8
DSD = 1e8 + max(nx,ny)/2

#Fan Beam Geometry - Parallel
#DSO = max(nx,ny)*np.sqrt(2)/2 
#DSD = DSO*2

#Sino params 
na = 64
nu, nv = 64, 128
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
r = 20
x0 = 0
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



sino1c = dd.dd_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=1.0)

n_runs = 10
t0 = time.perf_counter()
for _ in range(n_runs):
    sino1c = dd.dd_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=1.0)
t1 = time.perf_counter()

print(f"avg time: {(t1 - t0)/n_runs*1e3:.3f} ms")

n_runs = 10
t0 = time.perf_counter()
for _ in range(n_runs):
    sino1c = dd.dd_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=1.0)
t1 = time.perf_counter()

print(f"avg time: {(t1 - t0)/n_runs*1e3:.3f} ms")

