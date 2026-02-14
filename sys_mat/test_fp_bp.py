#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 19 12:57:48 2025

@author: vargasp
"""

from sys import path

path.append('/Users/pvargas21/Codebase/Libraries')

import numpy as np
import vir.sys_mat.analytic_sino as asino
from vir.sys_mat.test_orientation import p_images, p_run, p_run_single
from vir.sys_mat.time_testing import p_time, p_time_single

#Image params - Pixels
nx, ny, nz = 32, 32, 32
d_pix = 1.0

#Fan Beam Geometry - Parallel
DSO = 1e4
DSD = 1e4 + max(nx,ny)/2

#Fan Beam Geometry - Fanbeam
#DSO = max(nx,ny)*np.sqrt(2)/2 
#DSD = DSO*2

#Sino 32 
na = 32
nu, nv = 32,32
du, dv = 1., 1
su, sv = .25, 0.
na_lets, nu_lets, nv_lets = 5, 5, 5

#
ang_arr = np.linspace(0, np.pi*2, na, endpoint=False)#, dtype=np.float32)
ang_arr_lets = np.linspace(0, np.pi*2, na*na_lets, endpoint=False).reshape(na,na_lets)#, dtype=np.float32)
ang_arr_lets -= ang_arr_lets[0,2]

u_arr = du*(np.arange(nu) - nu/2.0 + 0.5 + su)
v_arr = dv*(np.arange(nv) - nv/2.0 + 0.5 + sv)

u_arr_lets = du*(np.arange(nu*nu_lets) - nu/2.0*nu_lets + 0.5 + su).reshape(nu,nu_lets)/nu_lets
v_arr_lets = dv*(np.arange(nv*nv_lets) - nv/2.0*nv_lets + 0.5 + sv).reshape(nv,nv_lets)/nv_lets


#Phantom Paramters Sino
r = 15
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
sinoCi = asino.analytic_sphere_sino_cone_3d((x0,y0,z0,r,1), ang_arr, u_arr, v_arr,DSO, DSD)
sinoC = asino.analytic_sphere_sino_cone_3d((x0,y0,z0,r,1), ang_arr_lets, u_arr_lets, v_arr_lets,DSO, DSD).mean(5).mean(3).mean(1)




"""
ang_arr = np.array([ang_arr[0]])
sinoP = sinoP[:1,...]
sinoF = sinoF[:1,...]
sinoC = sinoC[:1,...]
"""

#test = p_run_single(img3d,sinoP,sinoF,sinoC,ang_arr,DSO,DSD,du,dv,su,sv,d_pix,x0,y0,z0,r,
#                              fp=True,bp=True)


p_images(img3d,sinoP,sinoF,sinoC,ang_arr,DSO,DSD,du,dv,su,sv,d_pix,x0,y0,z0,r,
             ph=False,fp=False,bp=True)


#p_time(img3d,sinoP,sinoF,sinoC,ang_arr,DSO,DSD,du,dv,su,sv,d_pix,x0,y0,z0,r,
#              fp=True,bp=False,n_runs=10)

#p_time_single(img3d,sinoP,sinoF,sinoC,ang_arr,DSO,DSD,du,dv,su,sv,d_pix,x0,y0,z0,r,
#              fp=True,bp=False,n_runs=10)

#p_run(img3d,sinoP,sinoF,sinoC,ang_arr,DSO,DSD,du,dv,su,sv,d_pix,x0,y0,z0,r,
#           fp=True,bp=True)






