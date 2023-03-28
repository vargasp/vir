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





nPix = (10,10,1)
dPix = (1.2, .8,.6)


src = (-5, -5, -1)
trg = (5.5,5.5, 1)




sdlist = sd.siddons(trg, src, nPixels=nPix, dPixels=dPix)
sdarray = sd.list2array(sdlist, nPix)
print(sdlist)


plt.imshow(sdarray[:,:,0].T, extent=(-5,5,-5,5), origin='lower')

src = np.array(src)*np.array(dPix)[[1,0,2]]
trg = np.array(trg)*np.array(dPix)[[1,0,2]]

plt.plot((src[0], trg[0]) , (src[1], trg[1]), color="red", linewidth=1)

plt.yticks(np.arange(-6, 7))
plt.xticks(np.arange(-6, 7))
plt.grid()
plt.show()







#(x,y)
nPix = (20,10,1)
dPix = (1., .1,2)


DetsY = vir.censpace(6,d=1)
Views = np.linspace(0,2*np.pi, 2, endpoint=False)


src, trg = sd.circular_geom_st(DetsY, Views, geom="par", src_iso=None, det_iso=None, DetsZ=None)

src = src[1,2,:]
trg = trg[1,2,:]
src[2]= 0.5
trg[2]= 0.5

sdlist = sd.siddons(trg, src, nPixels=nPix, dPixels=dPix)
sdarray = sd.list2array(sdlist, nPix, ave=False)
#print(sdlist)

x0 = -int(nPix[0]/2)
xN = int(nPix[0]/2)
y0 = -int(nPix[1]/2)
yN = int(nPix[1]/2)

plt.imshow(sdarray[:,:,0].T, extent=(x0,xN,y0,yN), origin='lower')

src = np.array(src)*np.array(dPix)[[1,0,2]]
trg = np.array(trg)*np.array(dPix)[[1,0,2]]

#plt.plot((src[0], trg[0]) , (src[1], trg[1]), color="red", linewidth=1)

plt.xticks(np.arange(x0-1, xN+2))
plt.yticks(np.arange(y0-1, yN+2))
plt.grid()
plt.show()



