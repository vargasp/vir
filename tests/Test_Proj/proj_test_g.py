# -*- coding: utf-8 -*-
"""
Created on Mon Dec 15 15:58:25 2025

@author: varga
"""

import os
from sys import path
from functools import partial as pfunc
import timeit
#path.append('C:\\Users\\varga\\Codebase\\Libraries')


import numpy as np
import vir
import vir.sys_mat.siddon as sd
import vir.projection.proj as proj
import vir.projection.proj_ctype as proj_ctype
import vir.projection.proj_numba as proj_numba
#import vir.projection.proj_cuda as proj_cuda

import vir.proj_geom as pg
import vir.mpct as mpct

#Sheep Lgan Cicular
nPix = 32
nPixels = (nPix,nPix,nPix)
dPix = 1.0
nDets = 64
dDet = 1.0
nTheta = 128
det_lets = 1
src_lets = 1


phantom = np.zeros(nPixels, dtype=np.float32)
phantom[4:12,4:12,4:12] = 1.0


try:
    script_path = os.path.realpath(__file__)
    infile_root = os.path.dirname(script_path) + os.sep    
except:
    infile_root = os.getcwd() + os.sep    


try:
    sdlist_r = np.load(infile_root+"sdlist_test_r.npy", allow_pickle=True)
    sdlist_u = np.load(infile_root+"sdlist_test_u.npy", allow_pickle=True)
    sdlist_f = np.load(infile_root+"sdlist_test_f.npy", allow_pickle=True)
except:
    d = vir.Detector1d(nDets=nDets,dDet=dDet, det_lets=det_lets)
    Thetas = np.linspace(0,2*np.pi,nTheta, endpoint=False)
    srcs, trgs = pg.geom_circular(d.Dets, Thetas, geom="cone", DetsZ=d.Dets,\
                                  det_iso=16, src_iso=32)
    
    sdlist_r = sd.siddons(srcs,trgs,nPixels, dPix, ravel=True, flat=False)
    sdlist_u = sd.siddons(srcs,trgs,nPixels, dPix, ravel=False, flat=False)
    sdlist_f = sd.siddons(srcs,trgs,nPixels, dPix, ravel=True, flat=True)
    np.save("sdlist_test_r",sdlist_r)
    np.save("sdlist_test_u",sdlist_u)
    np.save('sdlist_test_f.npy', np.array(sdlist_f, dtype=object), allow_pickle=True)

sdlist_r_c = mpct.list_ctypes_object(sdlist_r, ravel=True, flat=True)
sdlist_u_c = mpct.list_ctypes_object(sdlist_u, ravel=False, flat=True)

sino1r = proj.sd_f_proj(phantom, sdlist_r, ravel=True, flat=False)
sino1u = proj.sd_f_proj(phantom, sdlist_u, ravel=False, flat=False)
sino1f = proj.sd_f_proj(phantom, sdlist_f, ravel=True, flat=True,sino_shape=sdlist_r.shape)

sino2r = np.zeros(sdlist_r.shape, dtype=np.float32)
sino2u = np.zeros(sdlist_u.shape, dtype=np.float32) 
sino3r = np.zeros(sdlist_r.shape, dtype=np.float32) 
sino3u = np.zeros(sdlist_u.shape, dtype=np.float32) 
sino3r, sino3r_c = mpct.ctypes_vars(sino3r)
sino3u, sino3u_c = mpct.ctypes_vars(sino3u)
phantom, phantom_c = mpct.ctypes_vars(phantom)

proj_ctype.sd_f_proj_c(phantom, sino2u, sdlist_u_c, ravel=False, C=False)
proj_ctype.sd_f_proj_c(phantom, sino2r, sdlist_r_c, ravel=True, C=False)
proj_ctype.sd_f_proj_c(phantom_c, sino3u_c, sdlist_u_c, ravel=False, C=True,dims=phantom.shape,nRays=sino3u.size)
proj_ctype.sd_f_proj_c(phantom_c, sino3r_c, sdlist_r_c, ravel=True, C=True,dims=phantom.shape,nRays=sino3r.size)


sino4s = proj_numba.sd_f_proj_numba(phantom, sdlist_f,sdlist_r.shape)
sino4p = proj_numba.sd_f_proj_numba_p(phantom, sdlist_f,sdlist_r.shape)

#sino5 = proj_cuda.sd_f_proj_numba_g(phantom, sdlist_f,sdlist_r.shape)


diff1u = np.max(np.abs(sino1r - sino1u))
diff1f = np.max(np.abs(sino1f - sino1u))
diff2u = np.max(np.abs(sino2u - sino1u))
diff2r = np.max(np.abs(sino2r - sino1u))
diff3u = np.max(np.abs(sino3u - sino1u))
diff3r = np.max(np.abs(sino3r - sino1u))
diff4s = np.max(np.abs(sino4s - sino1u))
diff4p = np.max(np.abs(sino4p - sino1u))
#diff5 = np.max(np.abs(sino5 - sino1u))

iters = 50

pf = pfunc(proj.sd_f_proj, phantom,sdlist_u,True,False)
a = timeit.timeit(pf, number=iters)/iters
print(f"{'FP S-Core Python Shp-Unrav':<27} Time: {a:.5f}s")

pf = pfunc(proj.sd_f_proj, phantom,sdlist_r,True,False)
a = timeit.timeit(pf, number=iters)/iters
print(f"{'FP S-Core Python Shp-Rav':<27} Time: {a:.5f}s, Diff: {diff1u:.2e}")

pf = pfunc(proj.sd_f_proj, phantom,sdlist_f,True,True,sdlist_r.shape)
a = timeit.timeit(pf, number=iters)/iters
print(f"{'FP S-Core Python Flt-Rav':<27} Time: {a:.5f}s, Diff: {diff1f:.2e}")


pf = pfunc(proj_ctype.sd_f_proj_c, phantom,sino2u,sdlist_u_c,False,False)
a = timeit.timeit(pf, number=iters)/iters
print(f"{'FP S-Core Ctype Shp-Unrav':<27} Time: {a:.5f}s, Diff: {diff3u:.2e}")    

pf = pfunc(proj_ctype.sd_f_proj_c, phantom,sino2r,sdlist_r_c,True,False)
a = timeit.timeit(pf, number=iters)/iters
print(f"{'FP S-Core Ctype Shp-Rav':<27} Time: {a:.5f}s, Diff: {diff3r:.2e}")    

pf = pfunc(proj_ctype.sd_f_proj_c, \
           phantom_c,sino3u_c,sdlist_u_c,False,True,phantom.shape,sino3u.size)
a = timeit.timeit(pf, number=iters)/iters
print(f"{'FP S-Core Ctype C-Shp-Unrav':<27} Time: {a:.5f}s, Diff: {diff3u:.2e}")    

pf = pfunc(proj_ctype.sd_f_proj_c, \
           phantom_c,sino3r_c,sdlist_r_c,True,True,phantom.shape,sino3r.size)
a = timeit.timeit(pf, number=iters)/iters
print(f"{'FP S-Core Ctype C-Shp-Rav':<27} Time: {a:.5f}s, Diff: {diff3r:.2e}")    


pf = pfunc(proj_numba.sd_f_proj_numba, phantom, sdlist_f,sdlist_r.shape)
a = timeit.timeit(pf, number=iters)/iters
print(f"{'FP S-Core Numba Flt-Rav':<27} Time: {a:.5f}s, Diff: {diff4s:.2e}")

pf = pfunc(proj_numba.sd_f_proj_numba_p, phantom, sdlist_f,sdlist_r.shape)
a = timeit.timeit(pf, number=iters)/iters
print(f"{'FP M-Core Numba Flt-Rav':<27} Time: {a:.5f}s, Diff: {diff4p:.2e}")

"""
pf = pfunc(proj_cuda.sd_f_proj_numba_g, phantom, sdlist_f,sdlist_r.shape)
a = timeit.timeit(pf, number=iters)/iters
print(f"{'FP GPU Numba Flt-Rav':<27} Time: {a:.5f}s, Diff: {diff5:.2e}")
"""




"""

iters=1000
pf = pfunc(proj_numba.sd_f_proj_numba, phantom, sdlist_f,sdlist_r.shape)
a = timeit.timeit(pf, number=iters)/iters
print(f"{'FP S-Core Numba Flt-Rav':<27} Time: {a:.5f}s, Diff: {diff2u:.2e}")

"""




"""
partial_function = functools.partial(proj_mpct.mp_sd_f_proj, phantom, sdlist_u)
a = timeit.timeit(partial_function, number=iters)/iters

test = proj_mpct.mp_sd_f_proj(phantom, sdlist_u)

"""









