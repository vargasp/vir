#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 22:16:09 2021

@author: vargasp
"""


import numpy as np
import matplotlib.pyplot as plt
import time
import importlib

import vt
import vir
import vir.siddon as sd
import vir.proj_geom as pg
import time
    


nPix = 4
nPixels = (nPix,nPix,1)
dPix = 1.0


#Cases where src and trg at and near edge in both directions
trgs = np.array([2.0,.25,.25])
srcs = np.array([-2.1,.25,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
srcs = np.array([-2.0,.25,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
srcs = np.array([-1.9,.25,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)

srcs = np.array([-2.0,.25,.25])
trgs = np.array([1.9,.25,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
trgs = np.array([2.0,.25,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
trgs = np.array([2.1,.25,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)

trgs = np.array([-2.0,.25,.25])
srcs = np.array([2.1,.25,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
srcs = np.array([2.0,.25,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
srcs = np.array([1.9,.25,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)

srcs = np.array([2.0,.25,.25])
trgs = np.array([-1.9,.25,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
trgs = np.array([-2.0,.25,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
trgs = np.array([-2.1,.25,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)


#Cases where the line is outside the grid
srcs = np.array([.25,4.0,0.25])
trgs = np.array([.25,4.0,0.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
srcs = np.array([-4.0,.25,0.25])
trgs = np.array([-4.0,.25,0.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)


#Cases where the line is along an intersection
srcs = np.array([2.0,-2.0,0.0])
trgs = np.array([-2.0,-2.0,0.0])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
srcs = np.array([2.0,-1.0,0.0])
trgs = np.array([-2.0,-1.0,0.0])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
srcs = np.array([2.0,0.0,0.0])
trgs = np.array([-2.0,0.0,0.0])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
srcs = np.array([2.0,1.0,0.0])
trgs = np.array([-2.0,1.0,0.0])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
srcs = np.array([2.0,2.0,0.0])
trgs = np.array([-2.0,2.0,0.0])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)


#Cases where the line intersects at a grid intersection
srcs = np.array([-2.1,-2.1,.25])
trgs = np.array([2.1,2.1,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
srcs = np.array([-2.0,-2.0,.25])
trgs = np.array([2.0,2.0,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
srcs = np.array([-1.9,-1.9,.25])
trgs = np.array([1.9,1.9,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)

srcs = np.array([-4.0,0.0,.25])
trgs = np.array([0.0,4.0,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
srcs = np.array([0.0,4.0,.25])
trgs = np.array([4.0,0.0,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
srcs = np.array([4.0,0.0,.25])
trgs = np.array([0.0,4.0,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
srcs = np.array([4.0,0.0,.25])
trgs = np.array([0.0,-4.0,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)


#Small change test
nPix = 128
nPixels = (nPix,nPix,1)
dPix = 1.0

srcs = np.array([-64.1,0.0,.25])
trgs = np.array([64.1,0.0,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
srcs = np.array([-64.1,-0.0001,.25])
trgs = np.array([64.1,0.0001,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
srcs = np.array([-64.1,-2,.25])
trgs = np.array([64.1,0.0001,.25])
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)






#Parallel Beam Circular Trajectory
nPix = 128
nPixels = (nPix,nPix,1)
dPix = 1.0
nDets = nPix*4
dDet = .1
nTheta = 1
det_lets = 1
src_lets = 1

d = vir.Detector1d(nDets=nDets,dDet=dDet, det_lets=det_lets)
Thetas = np.linspace(0,np.pi,nTheta,endpoint=False) + np.pi/4

srcs, trgs = pg.geom_circular(d.Dets, Thetas,geom="par", src_iso=np.ceil(np.sqrt(2*(nPix*dPix/2.)**2)), det_iso=np.ceil(np.sqrt(2*(nPix*dPix/2.)**2)))
srcs[:,:,2] = .25
trgs[:,:,2] = .25

a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True, epsilon=1e-5)
b = sd.siddons(srcs,trgs,nPixels, dPix, ravel=True, epsilon=1e-5)
c = sd.siddons(srcs,trgs,nPixels, dPix, ravel=False, epsilon=1e-5)
h = sd.list2array(a, nPixels, flat=True)
i = sd.list2array(b, nPixels, ravel=True)
j = sd.list2array(c, nPixels, ravel=False)

vt.CreateImage(h[:,:,0])
vt.CreateImage(i[:,:,0])
vt.CreateImage(j[:,:,0])

print((h - i).max())
print((h - j).max())


#List Utilities
d = sd.list_flatten(b,nPixels, ravel=True)
e = sd.list_flatten(c,nPixels, ravel=False)
h = sd.list2array(a, nPixels, flat=True)
i = sd.list2array(d, nPixels, flat=True)
j = sd.list2array(e, nPixels, flat=True)

print((h - i)[:,:,0])
print((h - j)[:,:,0])


d = sd.list_unravel(b,nPixels)
e = sd.list_ravel(c,nPixels)
h = sd.list2array(a, nPixels, flat=True)
i = sd.list2array(d, nPixels, ravel=False)
j = sd.list2array(e, nPixels, ravel=True)

print((h - i)[:,:,0])
print((h - j)[:,:,0])

a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
sd.list_ave(a, flat=True, ravel=True, nPixels=nPixels, nRays=32)




























