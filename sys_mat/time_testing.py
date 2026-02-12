#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 20:07:24 2026

@author: pvargas21
"""


from sys import path

path.append('/Users/pvargas21/Codebase/Libraries')


import numpy as np
import time

import vir.sys_mat.dd as dd
import vir.sys_mat.rd as rd
import vir.sys_mat.pd as pd

def benchmark(func, *args, n_runs=10, warmup=2, reset=None, **kwargs):
    """
    Benchmark a function.

    Parameters
    ----------
    func : callable
        Function to benchmark.
    *args :
        Positional arguments for func.
    n_runs : int
        Number of measured runs.
    warmup : int
        Number of warmup runs (not timed).
    reset : callable or None
        Function called before each run (e.g., to zero output buffer).
    **kwargs :
        Keyword arguments for func.
    """

    # Warmup (JIT compilation + cache warm)
    for _ in range(warmup):
        if reset:
            reset()
        func(*args, **kwargs)

    times = []
    medians = []
    for _ in range(5):  # repeat measurements to compute median
        for _ in range(n_runs):
            if reset:
                reset()

            t0 = time.perf_counter()
            func(*args, **kwargs)
            t1 = time.perf_counter()
            times.append((t1 - t0))

        medians.append(np.median(times))
    print(f"{func.__name__}: ", end="")
    for m in medians[:-1]:
        print(f"{m*1e3:.3f} ms, ", end="")

    print(f"{medians[-1]*1e3:.3f} ms")




def p_time_single(img3d,sinoP,sinoF,sinoC,ang_arr,DSO,DSD,du,dv,su,sv,d_pix,x0,y0,z0,r,
           fp=True,bp=True,n_runs=10):
    nx, ny, nz = img3d.shape
    na, nu, nv = sinoC.shape

    img2d = img3d[:,:,int(nz/2)]


    if fp:
        benchmark(dd.dd_fp_cone_3d, img3d,ang_arr,nu,nv,DSO,DSD,du,dv,su,sv,d_pix)

    if bp:
        benchmark(pd.pd_bp_cone_3d, sinoC,ang_arr,(nx,ny,nz),DSO,DSD,du,dv,d_pix)


def p_time(img3d,sinoP,sinoF,sinoC,ang_arr,DSO,DSD,du,dv,su,sv,d_pix,x0,y0,z0,r,
           fp=True,bp=True,n_runs=10):

    nx, ny, nz = img3d.shape
    na, nu, nv = sinoC.shape

    img2d = img3d[:,:,int(nz/2)]


    if fp:
        benchmark(dd.dd_fp_par_2d, img2d,ang_arr,nu,du,su,d_pix)
        benchmark(rd.aw_fp_par_2d, img2d,ang_arr,nu,du,su,d_pix)
        benchmark(rd.aw_fp_par_2d, img2d,ang_arr,nu,du,su,d_pix,True)
        benchmark(pd.pd_fp_par_2d, img2d,ang_arr,nu,du,su,d_pix)
        
        benchmark(dd.dd_fp_fan_2d, img2d,ang_arr,nu,DSO,DSD,du,su,d_pix)
        benchmark(rd.aw_fp_fan_2d, img2d,ang_arr,nu,DSO,DSD,du,su,d_pix)
        benchmark(rd.aw_fp_fan_2d, img2d,ang_arr,nu,DSO,DSD,du,su,d_pix,True)
        benchmark(pd.pd_fp_fan_2d, img2d,ang_arr,nu,DSO,DSD,du,su,d_pix)
        
        benchmark(dd.dd_fp_cone_3d, img3d,ang_arr,nu,nv,DSO,DSD,du,dv,su,sv,d_pix)
        benchmark(rd.aw_fp_cone_3d, img3d,ang_arr,nu,nv,DSO,DSD,du,dv,su,sv,d_pix)
        benchmark(rd.aw_fp_cone_3d, img3d,ang_arr,nu,nv,DSO,DSD,du,dv,su,sv,d_pix,True)
        benchmark(pd.pd_fp_cone_3d, img3d,ang_arr,nu,nv,DSO,DSD,du,dv,su,sv,d_pix)
    
    if bp:
        
        benchmark(dd.dd_bp_par_2d, sinoP,ang_arr,(nx,ny),du,su,d_pix)
        benchmark(rd.aw_bp_par_2d, sinoP,ang_arr,(nx,ny),du,su,d_pix)
        benchmark(rd.aw_bp_par_2d, sinoP,ang_arr,(nx,ny),du,su,d_pix,True)
        benchmark(pd.pd_bp_par_2d, sinoP,ang_arr,(nx,ny),du,su,d_pix)
        
        benchmark(dd.dd_bp_fan_2d, sinoF,ang_arr,(nx,ny),DSO,DSD,du,su,d_pix)
        benchmark(rd.aw_bp_fan_2d, sinoF,ang_arr,(nx,ny),DSO,DSD,du,su,d_pix)
        benchmark(rd.aw_bp_fan_2d, sinoF,ang_arr,(nx,ny),DSO,DSD,du,su,d_pix,True)
        benchmark(pd.pd_bp_fan_2d, sinoF,ang_arr,(nx,ny),DSO,DSD,du,su,d_pix)
        
        benchmark(dd.dd_bp_cone_3d, sinoC,ang_arr,(nx,ny,nz),DSO,DSD,du,dv,d_pix)
        benchmark(rd.aw_bp_cone_3d, sinoC,ang_arr,(nx,ny,nz),DSO,DSD,du,dv,d_pix)
        benchmark(rd.aw_bp_cone_3d, sinoC,ang_arr,(nx,ny,nz),DSO,DSD,du,dv,d_pix,True)
        benchmark(pd.pd_bp_cone_3d, sinoC,ang_arr,(nx,ny,nz),DSO,DSD,du,dv,d_pix)
        
        
        
            

