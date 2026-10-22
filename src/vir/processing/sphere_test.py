#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 22:11:09 2021

@author: vargasp
"""

import importlib
import numpy as np
import matplotlib.pyplot as plt

import vir
import sphere_int as si
import visualization_tools as vt
import phantoms as pt
import filtration as ft
import proj as pj



nViews = 1
pd = pt.Phantom(spheres = [[0,0,0,10,1]])
g = vir.Geom(nViews=nViews)
s = vir.Source2d()


nDets_list = np.arange(2,51)

im_i = []
for i, nDets in enumerate(nDets_list):
    print(i)
    X = np.linspace(-10,10,nDets+1)
    Y = np.linspace(-10,10,nDets+1)

    V = si.sphere_int_proj2d(X,Y,x0=0,y0=0,r=10)
    im_i.append(V)
    



#Calculates min number of detectorlets
det_lets_list = np.arange(20)+1

RMSE = np.zeros([nDets_list.size,det_lets_list.size])
im_p =[] 
for i, nDets in enumerate(nDets_list):
    dDet = 20.0/nDets
    
    for j, det_lets in enumerate(det_lets_list):
        d = vir.Detector2d(nDets=nDets,dDet=dDet,offset=0.0,det_lets=det_lets)
        X = np.linspace(-10,10,int(20/d.dW)+1)
        Y = np.linspace(-10,10,int(20/d.dH)+1)
        P = pj.createParSinoClass(g,d,s,pd).squeeze() # [distance * linear attenuation]
        im_p.append(P)
        
        RMSE[i,j] = np.sqrt(np.mean((im_i[i]/(dDet**2) - P)**2))



for i, nDets in enumerate(nDets_list):    
    title = "Integration:  nDet = "+str(nDets)
    f_title = "Integrat_nDet"+str(nDets)
    vt.CreateImageCart(im_i[i],title=title,outfile='Integrarion')
    
    for j, det_lets in enumerate(det_lets_list):
        title = "Intersection:  nDets = "+str(nDets)+" Detlets = "+str(det_lets**2)
        title = "Intersec_nDets"+str(nDets)+"_lets"+str(det_lets**2)
        vt.CreateImageCart(im_p[j*i+j],title=title)


    
vt.CreatePlot(RMSE.T)



dMinSphere = 0.4 #[Micronns]
SphereScale = dMinSphere/0.8 #[Unitless]
nViews = 1
derenzo_spheres = pt.DerenzoPhantomSpheres(scale=SphereScale, z=1)
pd = pt.Phantom(spheres = derenzo_spheres)
pd.S =pd.S[:,[2,1,0,3,4]]

g = vir.Geom(nViews=nViews)
s = vir.Source2d()

dDet_list = np.array([.125,.25,.5,1.0 ])#[Micronns]


im_id = []
for i, dDet in enumerate(dDet_list):
    print(i)
    nDets = (32/dDet)

    d = vir.Detector2d(nDets=nDets,dDet=dDet,offset=0.0)
    V = pj.createParSinoClassInt(g,d,s,pd)
    im_id.append(V)
    

#Calculates min number of detectorlets
det_lets_list = np.arange(20)+1

RMSE2 = np.zeros([dDet_list.size,det_lets_list.size])
im_pd =[] 
for i, dDet in enumerate(dDet_list):
    nDets = (32/dDet)
    
    for j, det_lets in enumerate(det_lets_list):
        d = vir.Detector2d(nDets=nDets,dDet=dDet,offset=0.0,det_lets=det_lets)
        P = pj.createParSinoClass(g,d,s,pd).squeeze() # [distance * linear attenuation]
        im_pd.append(P)
        
        RMSE2[i,j] = np.sqrt(np.mean((im_id[i]/(dDet**2) - P)**2))




vt.CreateImageCart(im_pd[0][::-1,::-1].T,title = "Intersection dDet: 0.125, lets: 1",\
    outfile='Intersect_dDet0.125_let1')
vt.CreateImageCart(im_pd[20][::-1,::-1].T,title = "Intersection dDet: 0.25, lets: 1",\
    outfile='Intersect_dDet0.25_let1')
vt.CreateImageCart(im_pd[40][::-1,::-1].T,title = "Intersection dDet: 0.5, lets: 1",\
    outfile='Intersect_dDet0.5_let1')
vt.CreateImageCart(im_pd[60][::-1,::-1].T,title = "Intersection dDet: 1.0, lets: 1",\
    outfile='Intersect_dDet1.0_let1')



vt.CreateImageCart(im_pd[9][::-1,::-1].T,title = "Intersection dDet: 0.125, lets: 100",\
    outfile='Intersect_dDet0.125_let100')
vt.CreateImageCart(im_pd[29][::-1,::-1].T,title = "Intersection dDet: 0.25, lets: 100",\
    outfile='Intersect_dDet0.25_let100')
vt.CreateImageCart(im_pd[49][::-1,::-1].T,title = "Intersection dDet: 0.5, lets: 100",\
    outfile='Intersect_dDet0.5_let100')
vt.CreateImageCart(im_pd[69][::-1,::-1].T,title = "Intersection dDet: 1.0, lets: 100",\
    outfile='Intersect_dDet1.0_let100')





vt.CreateImageCart(im_id[0][0,:,:],title = "Integration dDet: 0.125",\
    outfile='Integrate_dDet0.125')
vt.CreateImageCart(im_id[1][0,:,:],title = "Integration dDet: 0.25",\
    outfile='Integrate_dDet0.25')
vt.CreateImageCart(im_id[2][0,:,:],title = "Integration dDet: 0.5",\
    outfile='Integrate_dDet0.5')
vt.CreateImageCart(im_id[3][0,:,:],title = "Integration dDet: 1.0",\
    outfile='Integrate_dDet1.0')

import blur
from scipy import ndimage
lb050 = blur.LorentzianPSF(0.5,dPixel=.5,dims=2)
lb075 = blur.LorentzianPSF(0.75,dPixel=.5,dims=2)
lb100 = blur.LorentzianPSF(1.0,dPixel=.5,dims=2)


b1 = ndimage.convolve(im_id[2][0,:,:],lb050)
b2 = ndimage.convolve(im_id[2][0,:,:],lb075)
b3 = ndimage.convolve(im_id[2][0,:,:],lb100)
        

vt.CreateImageCart(b1,title = "Projected Image Blur = 0.5 dDet: 0.5",\
    outfile='blur5')
vt.CreateImageCart(b2,title = "Projected Image Blur = 0.75 dDet: 0.5",\
    outfile='blur75')  
vt.CreateImageCart(b3,title = "Projected Image Blur = 1.0 dDet: 0.5",\
    outfile='blur1')
    
    
    