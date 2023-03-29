#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 21:46:57 2023

@author: vargasp
"""

import numpy as np
import matplotlib.pyplot as plt


import vir
import vir.siddon as sd
import vt


def plot_params(nPix, dPix, src, trg):
    x0 = -int(nPix[0]/2)
    xN = int(nPix[0]/2)
    y0 = -int(nPix[1]/2)
    yN = int(nPix[1]/2)

    src = np.array(src)/np.array(dPix) #[[1,0,2]]
    trg = np.array(trg)/np.array(dPix) #[[1,0,2]]

    xS = (src[0], trg[0])
    yS = (src[1], trg[1])

    return x0,xN,y0,yN,xS,yS



#Plots line and sdarray
nPix = (12,10,1)
dPix = (.8, 1,1)

src = np.array((-2.5, .25,0))
trg = np.array((2,1.25, 0))


sdlist = sd.siddons(trg, src, nPixels=nPix, dPixels=dPix)
sdarray = sd.list2array(sdlist, nPix)
print(sdlist)

x0,xN,y0,yN,xS,yS = plot_params(nPix, dPix, src, trg)
plt.imshow(sdarray[:,:,0].T, origin='lower', extent=(x0,xN,y0,yN))
plt.plot(xS,yS, color="red", linewidth=1)
plt.xticks(np.arange(x0-1, xN+2))
plt.yticks(np.arange(y0-1, yN+2))
plt.grid()
plt.show()



#Plots backprojection
nPix = (100,100,1)
dPix = (.5, .5,2)

DetsY = vir.censpace(400,d=.1)
Views = np.linspace(0,2*np.pi, 400, endpoint=False)
src, trg = sd.circular_geom_st(DetsY, Views, geom="par", src_iso=None, det_iso=None, DetsZ=None)

sdlist = sd.siddons(trg, src, nPixels=nPix, dPixels=dPix)
sdarray = sd.list2array(sdlist, nPix, ave=False)
a = sdarray

x0,xN,y0,yN,xS,yS = plot_params(nPix, dPix, src, trg)
plt.imshow(sdarray[:,:,0].T, extent=(x0,xN,y0,yN), origin='lower')
plt.show()


#Plots backprojection
nPix = (100,100,1)
dPix = (.5, .5,2)

DetsY = vir.censpace(400,d=.1)
Views = np.linspace(0,np.pi, 400, endpoint=False)
src, trg = sd.circular_geom_st(DetsY, Views, geom="par", src_iso=None, det_iso=None, DetsZ=None)

sdlist = sd.siddons(trg, src, nPixels=nPix, dPixels=dPix)
sdarray = sd.list2array(sdlist, nPix, ave=False)
b = sdarray

x0,xN,y0,yN,xS,yS = plot_params(nPix, dPix, src, trg)
plt.imshow(sdarray[:,:,0].T, extent=(x0,xN,y0,yN), origin='lower')
plt.show()



