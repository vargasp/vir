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
import vir
import time
import matplotlib.pyplot as plt
import vir.phantoms as phat
from skimage.transform import resize


if __name__ == '__main__':
    #Sheep Lgan Cicular
    nPix = 16
    nPixels = (nPix,nPix,nPix)
    dPix = 1.0
    nDets = 32
    dDet = 1.0
    nTheta = 1
    det_lets = 1
    src_lets = 1
    
    """
    phantom = phat.discrete_sphere(nPixels=np.array(nPixels)*5, radius=nPixels[0]*5/2)*1.0
    phantom = resize(phantom, nPixels)
    
    phantom[:] = 0.0
    """
    
    phantom = np.zeros(nPixels, dtype=np.float32)
    phantom[4:8,4:8,4:8] = 1.0
        
    
    d = vir.Detector1d(nDets=nDets,dDet=dDet, det_lets=det_lets)
    
    Thetas = np.linspace(0,2*np.pi,nTheta, endpoint=False)
    srcs, trgs = sd.circular_geom_st(d.Dets, Thetas, geom="cone", DetsZ=d.Dets,\
                                     det_iso=16, src_iso=32)
        
    sdlist = sd.siddons(srcs,trgs,nPixels, dPix)
    sdlist_c = sd.list_ctypes_object(sdlist, flat = False)
    sdlist_c_flat = sd.list_ctypes_object(sdlist, flat = True)
    
    
    a = time.time()
    sino1 = proj.sd_f_proj(phantom, sdlist)
    a = (time.time() - a)
    print("{name:<25} Time: {a:.5f}s"\
          .format(name="FP Single Core Python", a=a))
    
    sino2 = np.zeros(sdlist.shape, dtype=np.float32) 
    a = time.time()
    proj.sd_f_proj_c(phantom, sino2, sdlist_c, flat=False)
    a = (time.time() - a)
    print("{name:<25} Time: {a:.5f}s, Diff: {diff:.2e}"\
          .format(name="FP Single Core C", a=a,diff=np.max(np.abs(sino2 - sino1))))
    
    sino3 = np.zeros(sdlist.shape, dtype=np.float32) 
    a = time.time()
    proj.sd_f_proj_c(phantom, sino3, sdlist_c_flat, flat=True)
    a = (time.time() - a)
    print("{name:<25} Time: {a:.5f}s, Diff: {diff:.2e}"\
          .format(name="FP Single Core C Flat", a=a,diff=np.max(np.abs(sino3 - sino1))))
    print("C Diff: {diff:.5f}".format(diff = np.max(np.abs(sino3 - sino2))))
    
    print("\n")
    
    a = time.time()
    bp1 = proj.sd_b_proj(sino1, sdlist, nPixels)
    a = (time.time() - a)
    print("{name:<25} Time: {a:.5f}s".format(name="BP Single Core Python", a=a))
    
    bp2 = np.zeros(nPixels, dtype=np.float32 )
    a = time.time()
    proj.sd_b_proj_c(bp2, sino1, sdlist_c, flat=False)
    a = (time.time() - a)
    print("{name:<25} Time: {a:.5f}s, Diff: {diff:.2e}"\
          .format(name="BP Single Core C", a=a,diff=np.max(np.abs(bp2 - bp1))))
     
    bp3 = np.zeros(nPixels, dtype=np.float32 )
    a = time.time()
    proj.sd_b_proj_c(bp3, sino1, sdlist_c_flat, flat=True)
    a = (time.time() - a)
    print("{name:<25} Time: {a:.5f}s, Diff: {diff:.2e}"\
          .format(name="BP Single Core C Flat", a=a,diff=np.max(np.abs(bp3 - bp1))))
    print("C Diff: {diff:.5f}".format(diff = np.max(np.abs(bp3 - bp2))))
    
    
    print("\n")
      

    a = time.time()
    sm1 = proj.mp_sd_f_proj(phantom, sdlist, cpus=5)
    a = (time.time() - a)
    print("{name:<25} Time: {a:.5f}, Diff: {diff:.2e}"\
          .format(name="FP Multi Core Python", a=a,diff=np.max(np.abs(sm1 - sino1))))

    a = time.time()
    bm1 = proj.mp_sd_b_proj(sino1, sdlist, nPixels, cpus=5)
    a = (time.time() - a)
    print("{name:<25} Time: {a:.5f}, Diff: {diff:.2e}"\
          .format(name="BP Multi Core Python", a=a,diff=np.max(np.abs(bm1 - bp1))))
   
    a = time.time()
    sm2 = proj.mpc_sd_f_proj(phantom, sdlist, cpus=5)
    a = (time.time() - a)
    print("{name:<25} Time: {a:.5f}, Diff: {diff:.2e}"\
          .format(name="FP Multi Core C", a=a,diff=np.max(np.abs(sm2 - sino1))))
    
    a = time.time()
    bm2 = proj.mpc_sd_b_proj(sino1, sdlist, nPixels, cpus=5)
    a = (time.time() - a)
    print("{name:<25} Time: {a:.5f}, Diff: {diff:.2e}"\
          .format(name="BP Multi Core C", a=a,diff=np.max(np.abs(bm2 - bp1))))
    
  
    print("\n")

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
    st2 = proj.mpc_sd_f_proj_t(phantom,sdlist,binYSft, binZSft) 
    a = (time.time() - a)
    print("{name:<30} Time: {a:.5f}s, Diff: {diff:.2e}"\
          .format(name="FP Multi Core C Trans", a=a,diff=np.max(np.abs(st2 - st1))))

    bt1 = np.zeros(nPixels, dtype=np.float32)
    a = time.time()
    proj.sd_b_proj_t_c(bt1, st1, sdlist_c, binYSft, binZSft, flat=False)
    a = (time.time() - a)
    print("{name:<30} Time: {a:.5f}s"\
          .format(name="BP Single Core C Trans", a=a))
        
    a = time.time()
    bt2 = proj.mpc_sd_b_proj_t(st1,sdlist,nPixels,binYSft, binZSft, cpus=5)
    a = (time.time() - a)
    print("{name:<30} Time: {a:.5f}s, Diff: {diff:.2e}"\
          .format(name="BP Multi Core C Trans", a=a,diff=np.max(np.abs(bt2 - bt1))))   
            
    
    
    ss1 = np.zeros(sdlist.shape + (nBinsY, nBinsZ,2), dtype=np.float32)
    a = time.time()
    proj.sd_f_proj_s_c(phantom, ss1, sdlist_c, binYSft, binZSft, flat=False)
    a = (time.time() - a)
    print("{name:<30} Time: {a:.5f}s, Diff: {diff:.2e}"\
          .format(name="FP Single Core C Sym", a=a,diff=np.max(np.abs(ss1[...,0] - st1))))
    
    a = time.time()
    ss2 = proj.mpc_sd_f_proj_s(phantom,sdlist,binYSft, binZSft, cpus=5) 
    a = (time.time() - a)
    print("{name:<30} Time: {a:.5f}s, Diff: {diff:.2e}"\
          .format(name="FP Multi Core C Sym", a=a,diff=np.max(np.abs(ss2 - ss1))))
        
             
    bs1 = np.zeros(nPixels, dtype=np.float32)
    a = time.time()
    proj.sd_b_proj_s_c(bs1, ss1, sdlist_c, binYSft, binZSft, flat=False)
    a = (time.time() - a)
    print("{name:<30} Time: {a:.5f}s"\
          .format(name="BP Single Core C Sym", a=a,))
    
    a = time.time()
    bs2 = proj.mpc_sd_b_proj_s(ss1,sdlist,nPixels, binYSft, binZSft, cpus=5) 
    a = (time.time() - a)
    print("{name:<30} Time: {a:.5f}s, Diff: {diff:.2e}"\
          .format(name="BP Multi Core C Sym", a=a,diff=np.max(np.abs(bs2 - bs1))))    
            
            
            
            