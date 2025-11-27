#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 22:46:20 2022

@author: vargasp
"""


import os
filename = os.environ.get('PYTHONSTARTUP')
if filename and os.path.isfile(filename):
    with open(filename) as fobj:
        startup_file = fobj.read()
    exec(startup_file)
    

import numpy as np

import vir.siddon as sd
import vir.projection as proj
import vir.proj_geom as pg
import vir
import time
import matplotlib.pyplot as plt
import vir.phantoms as phat
from skimage.transform import resize


#Sheep Lgan Cicular
nPix = 32
nPixels = (nPix,nPix,nPix)
dPix = 1.0
nDets = 64
dDet = 1.0
nTheta = 128
det_lets = 1
src_lets = 1

"""
phantom = phat.discrete_sphere(nPixels=np.array(nPixels)*5, radius=nPixels[0]*5/2)*1.0
phantom = resize(phantom, nPixels)

phantom[:] = 0.0
"""

phantom = np.zeros(nPixels, dtype=np.float32)
phantom[4:12,4:12,4:12] = 1.0
    
"""
d = vir.Detector1d(nDets=nDets,dDet=dDet, det_lets=det_lets)

Thetas = np.linspace(0,2*np.pi,nTheta, endpoint=False)
#srcs, trgs = sd.circular_geom_st(d.Dets, Thetas, geom="cone", DetsZ=d.Dets,\
#                                 det_iso=16, src_iso=32)
    

srcs, trgs = pg.geom_circular(d.Dets, Thetas, geom="cone", DetsZ=d.Dets,\
                              det_iso=16, src_iso=32)
    

    
sdlist_r = sd.siddons(srcs,trgs,nPixels, dPix, ravel=True, flat=False)
sdlist_u = sd.siddons(srcs,trgs,nPixels, dPix, ravel=False, flat=False)
np.save("sdlist_test_r",sdlist_r)
np.save("sdlist_test_u",sdlist_u)

"""
sdlist_r = np.load("sdlist_test_r.npy", allow_pickle=True)
sdlist_u = np.load("sdlist_test_u.npy", allow_pickle=True)

sdlist_r_c = sd.list_ctypes_object(sdlist_r, ravel=True, flat=True)
sdlist_u_c = sd.list_ctypes_object(sdlist_u, ravel=False, flat=True)


import functools
import timeit
import vir.mpct as mpct
iters = 20


sino1r = proj.sd_f_proj(phantom, sdlist_r, ravel=True)
sino1u = proj.sd_f_proj(phantom, sdlist_u, ravel=False)

sino2r = np.zeros(sdlist_r.shape, dtype=np.float32)
sino2u = np.zeros(sdlist_u.shape, dtype=np.float32) 
sino3r = np.zeros(sdlist_r.shape, dtype=np.float32) 
sino3u = np.zeros(sdlist_u.shape, dtype=np.float32) 
sino3r, sino3r_c = mpct.ctypes_vars(sino3r)
sino3u, sino3u_c = mpct.ctypes_vars(sino3u)
phantom, phantom_c = mpct.ctypes_vars(phantom)

proj.sd_f_proj_c(phantom, sino2u, sdlist_u_c, ravel=False, C=False)
proj.sd_f_proj_c(phantom, sino2r, sdlist_r_c, ravel=True, C=False)
proj.sd_f_proj_c(phantom_c, sino3u_c, sdlist_u_c, ravel=False, C=True,dims=phantom.shape,nRays=sino3u.size)
proj.sd_f_proj_c(phantom_c, sino3r_c, sdlist_r_c, ravel=True, C=True,dims=phantom.shape,nRays=sino3r.size)


diff1ru = np.max(np.abs(sino1r - sino1u))
diff2ru = np.max(np.abs(sino2r - sino2u))
diff3ru = np.max(np.abs(sino3r - sino3u))
diff21 = np.max(np.abs(sino2r - sino1r))
diff32 = np.max(np.abs(sino3r - sino2r))


"""
partial_function = functools.partial(proj.sd_f_proj, phantom, sdlist_u,False)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'FP Single Core Python Unraveled':<25} Time: {a:.5f}s")

partial_function = functools.partial(proj.sd_f_proj, phantom, sdlist_r,True)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'FP Single Core Python Raveled':<25} Time: {a:.5f}s")
print(f"Difference Ravel: {diff1ru:.2e}\n")

      
partial_function = functools.partial(proj.sd_f_proj_c, phantom,sino2u,sdlist_u_c,False,False)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'FP Single Core C Unraveled':<25} Time: {a:.5f}s")    

partial_function = functools.partial(proj.sd_f_proj_c, phantom,sino2r,sdlist_r_c,True,False)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'FP Single Core C Raveled':<25} Time: {a:.5f}s")    
print(f"Difference Ravel: {diff2ru:.2e}")
print(f"Difference Python/C: {diff21:.2e}\n")


partial_function = functools.partial(proj.sd_f_proj_c, phantom_c,sino3u_c,sdlist_u_c,False,True,phantom.shape,sino3r.size)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'FP Single Core C Unraveled':<25} Time: {a:.5f}s")    

partial_function = functools.partial(proj.sd_f_proj_c, phantom_c,sino3r_c,sdlist_r_c,True,True,phantom.shape,sino3u.size)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'FP Single Core C Raveled':<25} Time: {a:.5f}s")    
print(f"Difference Ravel: {diff3ru:.2e}")
print(f"Difference Python/C: {diff32:.2e}\n")


"""





bp1r = proj.sd_b_proj(sino1r, sdlist_r, nPixels, ravel=True)
bp1u = proj.sd_b_proj(sino1u, sdlist_u, nPixels, ravel=False)

bp2r = np.zeros(nPixels, dtype=np.float32 )
bp2u = np.zeros(nPixels, dtype=np.float32 )
bp3r = np.zeros(nPixels, dtype=np.float32 )
bp3u = np.zeros(nPixels, dtype=np.float32 )
bp3r, bp3r_c = mpct.ctypes_vars(bp3r)
bp3u, bp3u_c = mpct.ctypes_vars(bp3u)
phantom, phantom_c = mpct.ctypes_vars(phantom)

proj.sd_b_proj_c(bp2r, sino2r, sdlist_r_c, ravel=True)
proj.sd_b_proj_c(bp2u, sino2u, sdlist_u_c, ravel=False)
proj.sd_b_proj_c(bp3r_c, sino3r_c, sdlist_r_c, ravel=True, C=True,dims=phantom.shape,nRays=sino3u.size)
proj.sd_b_proj_c(bp3u_c, sino3u_c, sdlist_u_c, ravel=False, C=True,dims=phantom.shape,nRays=sino3r.size)


diff1ru = np.max(np.abs(bp1r - bp1u))
diff2ru = np.max(np.abs(bp2r - bp2u))
diff3ru = np.max(np.abs(bp3r - bp3u))
diff21 = np.max(np.abs(bp2r - bp1r))
diff32 = np.max(np.abs(bp3r - bp2r))


partial_function = functools.partial(proj.sd_b_proj, sino1u, sdlist_r, nPixels)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'BP Single Core Python Unraveled':<30} Time: {a:.5f}s")

partial_function = functools.partial(proj.sd_b_proj, sino1r, sdlist_u, nPixels)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'BP Single Core Python Raveled':<30} Time: {a:.5f}s")
print(f"Difference Ravel: {diff1ru:.2e}\n")


partial_function = functools.partial(proj.sd_b_proj_c,bp2u,sino2u,sdlist_u_c,False,False)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'BP Single Core C Unraveled':<30} Time: {a:.5f}s")

partial_function = functools.partial(proj.sd_b_proj_c,bp2r,sino2r,sdlist_r_c,True,False)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'BP Single Core C Raveled':<30} Time: {a:.5f}s") 
print(f"Difference Ravel: {diff2ru:.2e}")
print(f"Difference Python/C: {diff21:.2e}\n")


partial_function = functools.partial(proj.sd_b_proj_c,bp3u_c,sino3u_c,sdlist_u_c,False,True,phantom.shape,sino3r.size)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'BP Single Core C Unraveled':<30} Time: {a:.5f}s")

partial_function = functools.partial(proj.sd_b_proj_c,bp3r_c,sino3r_c,sdlist_r_c,True,True,phantom.shape,sino3r.size)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'BP Single Core C Raveled':<30} Time: {a:.5f}s") 
print(f"Difference Ravel: {diff3ru:.2e}")
print(f"Difference C/C: {diff32:.2e}\n")








"""
binYSft = (-8,8)
binZSft = (-8,8)

nBinsY = binYSft[1]-binYSft[0]
nBinsZ = binZSft[1]-binZSft[0]


st1 = np.zeros(sdlist.shape + (nBinsY, nBinsZ), dtype=np.float32)
a = time.time()
proj.sd_f_proj_t_c(phantom, st1, sdlist_c, binYSft, binZSft, flat=False)
a = (time.time() - a)
print("{name:<30} Time: {a:.5f}s, Diff: {diff:.2e}"\
      .format(name="FP Single Core C Trans", a=a,diff=np.max(np.abs(st1[...,8,8] - sino2))))

st2 = np.zeros(sdlist.shape + (nBinsY, nBinsZ), dtype=np.float32)
a = time.time()
proj.sd_f_proj_t_c(phantom,st2,sdlist_c_flat,binYSft, binZSft, flat=True) 
a = (time.time() - a)
print("{name:<30} Time: {a:.5f}s, Diff: {diff:.2e}"\
      .format(name="FP Single Core C Trans Flat", a=a,diff=np.max(np.abs(st2 - st1))))

bt1 = np.zeros(nPixels, dtype=np.float32)
a = time.time()
proj.sd_b_proj_t_c(bt1, st1, sdlist_c, binYSft, binZSft, flat=False)
a = (time.time() - a)
print("{name:<30} Time: {a:.5f}s"\
      .format(name="BP Single Core C Trans", a=a))

bt2 = np.zeros(nPixels, dtype=np.float32)
a = time.time()
proj.sd_b_proj_t_c(bt2,st1,sdlist_c_flat,binYSft, binZSft, flat=True) 
a = (time.time() - a)
print("{name:<30} Time: {a:.5f}s, Diff: {diff:.2e}"\
      .format(name="BP Single Core C Trans Flat", a=a,diff=np.max(np.abs(bt2 - bt1))))

    

  
ss1 = np.zeros(sdlist.shape + (nBinsY, nBinsZ,2), dtype=np.float32)
a = time.time()
proj.sd_f_proj_s_c(phantom, ss1, sdlist_c, binYSft, binZSft, flat=False)
a = (time.time() - a)
print("{name:<30} Time: {a:.5f}s, Diff: {diff:.2e}"\
      .format(name="FP Single Core C Sym", a=a,diff=np.max(np.abs(ss1[...,0] - st1))))

ss2 = np.zeros(sdlist.shape + (nBinsY, nBinsZ, 2), dtype=np.float32)
a = time.time()
proj.sd_f_proj_s_c(phantom,ss2,sdlist_c_flat,binYSft, binZSft, flat=True) 
a = (time.time() - a)
print("{name:<30} Time: {a:.5f}s, Diff: {diff:.2e}"\
      .format(name="FP Single Core C Sym Flat", a=a,diff=np.max(np.abs(ss2 - ss1))))


bs1 = np.zeros(nPixels, dtype=np.float32)
a = time.time()
proj.sd_b_proj_s_c(bs1, ss1, sdlist_c, binYSft, binZSft, flat=False)
a = (time.time() - a)
print("{name:<30} Time: {a:.5f}s"\
      .format(name="BP Single Core C Sym", a=a,))

bs2 = np.zeros(nPixels, dtype=np.float32)
a = time.time()
proj.sd_b_proj_s_c(bs2,ss1,sdlist_c_flat,binYSft, binZSft, flat=True) 
a = (time.time() - a)
print("{name:<30} Time: {a:.5f}s, Diff: {diff:.2e}"\
      .format(name="BP Single Core C Sym Flat", a=a,diff=np.max(np.abs(bs2 - bs1))))    
        
    
iters = 10
times = np.zeros([2, iters])
for i in range(iters):    
        
    ss2 = np.zeros(sdlist.shape + (nBinsY, nBinsZ, 2), dtype=np.float32)
    a = time.time()
    proj.sd_f_proj_s_c(phantom,ss2,sdlist_c_flat,binYSft, binZSft, flat=True) 
    a = (time.time() - a)
    times[0,i] = a
    print("{name:<30} Time: {a:.5f}s, Diff: {diff:.2e}"\
          .format(name="FP Single Core C Sym Flat", a=a,diff=np.max(np.abs(ss2 - ss1))))
        
    bs2 = np.zeros(nPixels, dtype=np.float32)
    a = time.time()
    proj.sd_b_proj_s_c(bs2,ss1,sdlist_c_flat,binYSft, binZSft, flat=True) 
    a = (time.time() - a)
    times[1,i] = a
    print("{name:<30} Time: {a:.5f}s, Diff: {diff:.2e}"\
          .format(name="BP Single Core C Sym Flat", a=a,diff=np.max(np.abs(bs2 - bs1))))    
        
        

print("{name:<30} Time Average: {mu:.5f}s, S.D.: {std:.5f}, Min: {m:.5F}"\
      .format(name="FP Single Core C Sym Flat", mu=times[0,:].mean(),\
              std=times[0,:].std(), m=times[0,:].min()))
        
print("{name:<30} Time Average: {mu:.5f}s, S.D.: {std:.5f}, Min: {m:.5F}"\
      .format(name="BP Single Core C Sym Flat", mu=times[1,:].mean(),\
              std=times[1,:].std(), m=times[1,:].min()))

    
"""
    