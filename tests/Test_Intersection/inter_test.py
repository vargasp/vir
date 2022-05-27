#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 10:10:15 2021

@author: vargasp
"""

import importlib
import numpy as np
import matplotlib.pyplot as plt

import vir as vir
import intersection as inter








src_iso = 60
#Sphere parameters (x0,y0,r,val)
spheres =[2,0,10,1]


ellipses = [0,0,10,10,0,1.0]

#The projection angle(s) in radians (nViews)
views = [0,.1,.2]
       
#The detector column postion(s) (nDets)
Cols = [0,1,2,3,4]
Rows = [0.,1,.2]
Bins = np.array(Cols)/src_iso


importlib.reload(inter)
print(inter.AnalyticSinoParSphere(spheres,views,Cols))
print(inter.AnalyticSinoFanSphere(spheres,views,Bins,src_iso,))




#print(inter.AnalyticSinoParSphere2(spheres,views,Rows,Cols))

print(inter.AnalyticSino(ellipses,np.array(views),np.array(Cols),src_iso))
    
    
    
    

print(inter.AnalyticSinoParEllipse([0,0,10,10,0,1],view,x))







nDets = 5
dDet = 1
det_lets = 3
d = vir.Detector2d(nDets=nDets,dDet=dDet,det_lets=det_lets,offset=0.0)
k = np.reshape(d.H_lets, [d.nH,d.nH_lets])
print(k.shape)
print(k)

for row_idx, row in enumerate(k):
    print(row)
    
    
    
nViews = 1
pd = pt.Phantom(spheres = [[0,0,0,10,1]])
g = vir.Geom(nViews=nViews)
s = vir.Source2d()


"""
SpherePlaneIntersection(P,S):
"""
P = ((0,0,1,2),(0,0,1,8))
S = (0,0,0,4)


P = np.array(P)
S = np.array(S)

if P.ndim == 1:
    P = P[np.newaxis]
    
Pt = np.sum(P[:,:3]*S[:3],axis=1)+P[:,3]
N = np.sum(P[:,:3]**2,axis=1)

d = np.abs(Pt) / np.sqrt(N)

with np.errstate(invalid='ignore'):
    r = np.sqrt(S[3]**2 - d**2)

r = np.nan_to_num(r) 
    
print(np.hstack((S[:3] -P[:,:3]*(Pt/N)[:,np.newaxis], r[np.newaxis].T)))




P = ((0,0,1,-5),(0,0,1,-4),(0,0,1,-3),(0,0,1,-2),(0,0,1,-1),(0,0,1,-0),\
     (0,0,1,1),(0,0,1,2),(0,0,1,3),(0,0,1,4),(0,0,1,5))
Sphere = (0,0,0,4)

S = inter.SpherePlaneIntersection(P,Sphere)


xi = np.linspace(0,10,5) - 5
view =0.0
u = np.add.outer(np.linalg.norm(S[:,:2],axis=1)*np.cos(np.arctan2(S[:,0], S[:,1]) - view),xi)
2.*np.sqrt((S[:,3][np.newaxis].T**2 - u**2).clip(0))




xi = np.linspace(0,10,5) - 5
view = np.array([0.0,1.0,3.0])

#view = 1.
#xi = 2.

xi = np.array(xi)
u = np.add.outer(np.linalg.norm(S[:,:2],axis=1)*np.cos(np.subtract.outer(np.arctan2(S[:,0], S[:,1]),view).T),xi)

print(u.shape)

if xi.ndim == 1:
    u2 = 2.*np.sqrt((S[:,3][np.newaxis].T**2 - u**2).clip(0))
else:
    u2 = 2.*np.sqrt((S[:,3]**2 - u**2).clip(0))




Sphere = (0,0,1,4)
Views = np.array([0.0,np.pi/2.0,np.pi])
Rows = np.linspace(0,10,5) - 5
Cols = np.linspace(0,10,6) - 5

Rows = 0
Rows = np.array(Rows)
P = np.hstack([np.tile([0,0,1],(Rows.size,1)), Rows[np.newaxis].T])


importlib.reload(inter)
sino = inter.AnalyticSinoParSphere2(Sphere,Views,Rows,Cols)
print(sino.shape)
print(sino)



