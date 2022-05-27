#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 21:46:15 2021

@author: vargasp
"""


import importlib
import numpy as np
import matplotlib.pyplot as plt

import vir
import visualization_tools as vt
import phantoms as pt
import proj as pj

dDet = .25
radius = 10
nDets = int(radius*2/dDet)
nViews = 1
det_lets = 5
pd = pt.Phantom(spheres = [[0,0,0,radius,1]])
g = vir.Geom(nViews=nViews)
s = vir.Source2d()
d = vir.Detector2d(nDets=nDets,dDet=dDet,offset=0.0,det_lets=det_lets)


importlib.reload(pj)
sino = pj.createSino(g,d,s,pd)
sino = -np.log(sino[0,:,:])

vt.CreateImage(-np.log(sino[0,:,:]))


sino_i = pj.createParSinoClassInt(g,d,s,pd)/dDet/dDet


vt.CreateImage(sino_i[0,:,:] - sino[:,:])


s1 = -np.log(pj.createSino(g,d,s,pd)[0,:,:])
s2 = -np.log(pj.createSinoLets(g,d,s,pd)[0,:,:])

importlib.reload(pj)
det_lets = 1
d = vir.Detector2d(nDets=nDets,dDet=dDet,offset=0.0,det_lets=det_lets)
s1 = -np.log(pj.createSinoLets(g,d,s,pd)[0,:,:])

det_lets = 2
d = vir.Detector2d(nDets=nDets,dDet=dDet,offset=0.0,det_lets=det_lets)
s2 = -np.log(pj.createSinoLets(g,d,s,pd)[0,:,:])

det_lets = 5
d = vir.Detector2d(nDets=nDets,dDet=dDet,offset=0.0,det_lets=det_lets)
s3 = -np.log(pj.createSinoLets(g,d,s,pd)[0,:,:])

det_lets = 10
d = vir.Detector2d(nDets=nDets,dDet=dDet,offset=0.0,det_lets=det_lets)
s4 = -np.log(pj.createSinoLets(g,d,s,pd)[0,:,:])

det_lets = 15
d = vir.Detector2d(nDets=nDets,dDet=dDet,offset=0.0,det_lets=det_lets)
s5 = -np.log(pj.createSinoLets(g,d,s,pd)[0,:,:])

det_lets = 20
d = vir.Detector2d(nDets=nDets,dDet=dDet,offset=0.0,det_lets=det_lets)
s6 = -np.log(pj.createSinoLets(g,d,s,pd)[0,:,:])

import intersection as inter
inter.SpherePlaneIntersection(P,S)
    

import proj as pj
det_lets = 2
d = vir.Detector2d(nDets=nDets,dDet=dDet,offset=0.0,det_lets=det_lets)
s2 = pj.createSinoLets(g,d,s,pd)


vt.CreateImage(-np.log(s1)[0,:,:])





