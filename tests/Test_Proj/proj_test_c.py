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
srcs, trgs = sd.circular_geom_st(d.Dets, Thetas, geom="cone", DetsZ=d.Dets,\
                                 det_iso=16, src_iso=32)
    

srcs, trgs = pg.geom_circular(d.Dets, Thetas, geom="cone", DetsZ=d.Dets,\
                              det_iso=16, src_iso=32)
    

    
sdlist = sd.siddons(srcs,trgs,nPixels, dPix)
sdlist_flat = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
np.save("sdlist_test",sdlist)
"""
sdlist = np.load("sdlist_test.npy", allow_pickle=True)


sdlist_c = sd.list_ctypes_object(sdlist, flat=False)
sdlist_c_flat = sd.list_ctypes_object(sdlist, flat=True)


import functools
import timeit
iters = 20


sino1 = proj.sd_f_proj(phantom, sdlist)
sino2 = np.zeros(sdlist.shape, dtype=np.float32) 
sino3 = np.zeros(sdlist.shape, dtype=np.float32) 
proj.sd_f_proj_c(phantom, sino2, sdlist_c, flat=False)
proj.sd_f_proj_c(phantom, sino3, sdlist_c_flat, flat=True)

diff21 = np.max(np.abs(sino2 - sino1))
diff31 = np.max(np.abs(sino3 - sino1))
diff32 = np.max(np.abs(sino3 - sino2))

partial_function = functools.partial(proj.sd_f_proj, phantom, sdlist)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'FP Single Core Python':<25} Time: {a:.5f}s")

partial_function = functools.partial(proj.sd_f_proj_c, phantom,sino2,sdlist_c,False)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'FP Single Core C':<25} Time: {a:.5f}s, Diff: {diff21:.2e}")    
    
partial_function = functools.partial(proj.sd_f_proj_c, phantom,sino2,sdlist_c_flat,True)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'FP Single Core C Flat':<25} Time: {a:.5f}s, Diff: {diff31:.2e}")
print(f"C Diff: {diff32:.5f}\n")


bp1 = proj.sd_b_proj(sino1, sdlist, nPixels)
bp2 = np.zeros(nPixels, dtype=np.float32 )
bp3 = np.zeros(nPixels, dtype=np.float32 )
proj.sd_b_proj_c(bp2, sino1, sdlist_c, flat=False)
proj.sd_b_proj_c(bp3, sino1, sdlist_c_flat, flat=True)

diff21 = np.max(np.abs(bp2 - bp1))
diff31 = np.max(np.abs(bp3 - bp1))
diff32 = np.max(np.abs(bp3 - bp2))

partial_function = functools.partial(proj.sd_b_proj, sino1, sdlist, nPixels)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'BP Single Core Python':<25} Time: {a:.5f}s")

partial_function = functools.partial(proj.sd_b_proj_c, bp2, sino1,sdlist_c,False)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'BP Single Core C':<25} Time: {a:.5f}s, Diff: {diff21:.2e}")
 
partial_function = functools.partial(proj.sd_b_proj_c, bp3, sino1,sdlist_c_flat,True)
a = timeit.timeit(partial_function, number=iters)/iters
print(f"{'BP Single Core C Flat':<25} Time: {a:.5f}s, Diff: {diff31:.2e}")
print(f"C Diff: {diff32:.5f}\n")



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

    
    
    