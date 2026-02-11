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



def p_time(img3d,sinoP,sinoF,sinoC,ang_arr,DSO,DSD,du,dv,su,sv,d_pix,x0,y0,z0,r,
           fp=True,bp=True,n_runs=10):

    nx, ny, nz = img3d.shape
    na, nu, nv = sinoC.shape

    img2d = img3d[:,:,int(nz/2)]


    if fp:
        sino1p = dd.dd_fp_par_2d(img2d,ang_arr,nu,du=du,su=su,d_pix=d_pix)
        sino2p = rd.aw_fp_par_2d(img2d,ang_arr,nu,du=du,su=su,d_pix=d_pix)
        sino3p = rd.aw_fp_par_2d(img2d,ang_arr,nu,du=du,su=su,d_pix=d_pix,joseph=True)
        sino4p = pd.pd_fp_par_2d(img2d,ang_arr,nu,du=du,su=su,d_pix=d_pix)
        
        sino1f = dd.dd_fp_fan_2d(img2d,ang_arr,nu,DSO,DSD,du=du,su=su,d_pix=d_pix)
        sino2f = rd.aw_fp_fan_2d(img2d,ang_arr,nu,DSO,DSD,du=du,su=su,d_pix=d_pix)
        sino3f = rd.aw_fp_fan_2d(img2d,ang_arr,nu,DSO,DSD,du=du,su=su,d_pix=d_pix,joseph=True)
        sino4f = pd.pd_fp_fan_2d(img2d,ang_arr,nu,DSO,DSD,du=du,su=su,d_pix=d_pix)
        
        sino1c = dd.dd_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
        sino2c = rd.aw_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
        sino3c = rd.aw_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix,joseph=True)
        sino4c = pd.pd_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
    
    
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sino1p = dd.dd_fp_par_2d(img2d,ang_arr,nu,du=du,su=su,d_pix=d_pix)
    t1 = time.perf_counter()
    print(f"FP Parallel - DD: avg time: {(t1 - t0)/n_runs*1e3:.3f} ms")
    
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sino2p = rd.aw_fp_par_2d(img2d,ang_arr,nu,du=du,su=su,d_pix=d_pix)
    t1 = time.perf_counter()
    print(f"FP Parallel - SD: avg time: {(t1 - t0)/n_runs*1e3:.3f} ms")

    t0 = time.perf_counter()
    for _ in range(n_runs):
        sino3p = rd.aw_fp_par_2d(img2d,ang_arr,nu,du=du,su=su,d_pix=d_pix,joseph=True)
    t1 = time.perf_counter()
    print(f"FP Parallel - JO: avg time: {(t1 - t0)/n_runs*1e3:.3f} ms")

    t0 = time.perf_counter()
    for _ in range(n_runs):
        sino4p = pd.pd_fp_par_2d(img2d,ang_arr,nu,du=du,su=su,d_pix=d_pix)
    t1 = time.perf_counter()
    print(f"FP Parallel - PD: avg time: {(t1 - t0)/n_runs*1e3:.3f} ms")
    
    
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sino1f = dd.dd_fp_fan_2d(img2d,ang_arr,nu,DSO,DSD,du=du,su=su,d_pix=d_pix)
    t1 = time.perf_counter()
    print(f"FP Fanbeam - DD: avg time: {(t1 - t0)/n_runs*1e3:.3f} ms")
    
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sino2f = rd.aw_fp_fan_2d(img2d,ang_arr,nu,DSO,DSD,du=du,su=su,d_pix=d_pix)
    t1 = time.perf_counter()
    print(f"FP Fanbeam - SD: avg time: {(t1 - t0)/n_runs*1e3:.3f} ms")

    t0 = time.perf_counter()
    for _ in range(n_runs):
        sino3f = rd.aw_fp_fan_2d(img2d,ang_arr,nu,DSO,DSD,du=du,su=su,d_pix=d_pix,joseph=True)
    t1 = time.perf_counter()
    print(f"FP Fanbeam - JO: avg time: {(t1 - t0)/n_runs*1e3:.3f} ms")

    t0 = time.perf_counter()
    for _ in range(n_runs):
        sino4f = pd.pd_fp_fan_2d(img2d,ang_arr,nu,DSO,DSD,du=du,su=su,d_pix=d_pix)
    t1 = time.perf_counter()
    print(f"FP Fanbeam - PD: avg time: {(t1 - t0)/n_runs*1e3:.3f} ms")
    
    
    t0 = time.perf_counter()
    for _ in range(n_runs):
       sino1c = dd.dd_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
    t1 = time.perf_counter()
    print(f"FP Conebeam - DD: avg time: {(t1 - t0)/n_runs*1e3:.3f} ms")
    
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sino2c = rd.aw_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
    t1 = time.perf_counter()
    print(f"FP Conebeam - SD: avg time: {(t1 - t0)/n_runs*1e3:.3f} ms")

    t0 = time.perf_counter()
    for _ in range(n_runs):
        sino3c = rd.aw_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix,joseph=True)
    t1 = time.perf_counter()
    print(f"FP Conebeam - JO: avg time: {(t1 - t0)/n_runs*1e3:.3f} ms")

    t0 = time.perf_counter()
    for _ in range(n_runs):
        sino4c = pd.pd_fp_cone_3d(img3d,ang_arr,nu,nv,DSO,DSD,du=du,dv=dv,su=su,sv=sv,d_pix=d_pix)
    t1 = time.perf_counter()
    print(f"FP Conebeam - PD: avg time: {(t1 - t0)/n_runs*1e3:.3f} ms")
    
    
    if bp:
        #sino = sino[8:9,:]
        #ang_arr = [ang_arr[8]]
        
        
        rec1p = dd.dd_bp_par_2d(sinoP,ang_arr,(nx,ny),du=du,su=su,d_pix=d_pix)
        rec2p = rd.aw_bp_par_2d(sinoP,ang_arr,(nx,ny),du=du,su=su,d_pix=d_pix)
        rec3p = rd.aw_bp_par_2d(sinoP,ang_arr,(nx,ny),du=du,su=su,d_pix=d_pix,joseph=True)
        rec4p = pd.pd_bp_par_2d(sinoP,ang_arr,(nx,ny),du=du,su=su,d_pix=d_pix)
        
        rec1f = dd.dd_bp_fan_2d(sinoF,ang_arr,(nx,ny),DSO,DSD,du=du,su=su,d_pix=d_pix)
        rec2f = rd.aw_bp_fan_2d(sinoF,ang_arr,(nx,ny),DSO,DSD,du=du,su=su,d_pix=d_pix)
        rec3f = rd.aw_bp_fan_2d(sinoF,ang_arr,(nx,ny),DSO,DSD,du=du,su=su,d_pix=d_pix,joseph=True)
        rec4f = pd.pd_bp_fan_2d(sinoF,ang_arr,(nx,ny),DSO,DSD,du=du,su=su,d_pix=d_pix)
        
        rec1c = dd.dd_bp_cone_3d(sinoC,ang_arr,(nx,ny,nz),DSO,DSD,du=du,dv=dv,d_pix=d_pix)
        rec2c = rd.aw_bp_cone_3d(sinoC,ang_arr,(nx,ny,nz),DSO,DSD,du=du,dv=dv,d_pix=d_pix)
        rec3c = rd.aw_bp_cone_3d(sinoC,ang_arr,(nx,ny,nz),DSO,DSD,du=du,dv=dv,d_pix=d_pix,joseph=True)
        rec4c = pd.pd_bp_cone_3d(sinoC,ang_arr,(nx,ny,nz),DSO,DSD,du=du,dv=dv,d_pix=d_pix)
        
        
        
            

