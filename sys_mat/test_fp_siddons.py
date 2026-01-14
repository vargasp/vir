#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 07:38:55 2026

@author: pvargas21
"""



import matplotlib.pyplot as plt
import numpy as np

import vir.sys_mat.siddon as sd
import vir.sys_mat.dd as dd
import vir.sys_mat.aw as aw
import vir.sys_mat.fp_2d_par as proj_2d
import vir.projection.proj as proj


nPix = 4
nPixels = (nPix,nPix,1)
d_pix = 

n_dets = 8
d_det = 1

phantom = np.ones((nPix,nPix))

src0 = -1e6
trg0 = 2


dets =  d_det * (np.arange(n_dets) - n_dets/2 + 0.5)

#Cases where src and trg at and near edge in both directions
srcs = np.empty((1,n_dets,3))
srcs[:,:,0] = src0
srcs[:,:,1] = 0.0
srcs[:,:,2] = 0.0

trgs = np.empty((1,n_dets,3))
trgs[:,:,0] = trg0
trgs[:,:,1] = dets
trgs[:,:,2] = 0.0


a = sd.siddons(srcs,trgs,nPixels, d_pix, flat=False,ravel=True)
s1 = proj.sd_f_proj(phantom, a, flat=False,ravel=True)


DSO = np.abs(src0)
DSD = np.abs(src0) + np.abs(trg0)
angles = np.array([0])


s2 = aw.aw_fp_2d_fan_flat(phantom, angles, n_dets, DSO, DSD, d_det=d_det, d_pix=d_pix)


print(s1-s2)