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


def print_sdlist(sdlist):
    arrs = ['x:','y:','z:']
 
    if sdlist[0] != None:    
        for i, arr in enumerate(arrs):
            print(arr, end='')


            for j in sdlist[0][i][:-1]:
                print(f'{j:6},', end = '')
            
            print(f'{sdlist[0][i][-1]:6}')

        print('d:', end='')
    
    
        for j in sdlist[0][3][:-1]:
            print(f' {j:1.3f},', end = '')
        
        print(f' {sdlist[0][3][-1]:2.3f}')



def plot_params(nPix, dPix, src, trg):
    x0 = -int(nPix[0]/2)*dPix[0]
    xN = int(nPix[0]/2)*dPix[0]
    y0 = -int(nPix[1]/2)*dPix[1]
    yN = int(nPix[1]/2)*dPix[1]

    src = np.array(src) #[[1,0,2]]
    trg = np.array(trg) #[[1,0,2]]

    xS = (src[0], trg[0])
    yS = (src[1], trg[1])

    return x0,xN,y0,yN,xS,yS



#Plots line and sdarray
nPix = (12,10,1)
dPix = (1.0, 1,1)

src = np.array((-0.5, -6.0,0))
trg = np.array((5.5,5.0, 0))


sdlist = sd.siddons(trg, src, nPixels=nPix, dPixels=dPix)
sdarray = sd.list2array(sdlist, nPix)
print_sdlist(sdlist)

x0,xN,y0,yN,xS,yS = plot_params(nPix, dPix, src, trg)
plt.imshow(sdarray[:,:,0].T, origin='lower', extent=(x0,xN,y0,yN))
plt.plot(xS,yS, color="red", linewidth=1)
plt.xticks(np.linspace(x0-dPix[0],xN+dPix[0], int(np.round((xN - x0)/dPix[0])+3)))
plt.yticks(np.linspace(y0-dPix[1],yN+dPix[1], int(np.round((yN - y0)/dPix[1])+3)))
plt.grid()
plt.xlabel('X')
plt.ylabel('Y')
plt.show()










"""

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




src = np.array([-141.38,0.49, 0.49])
trg = np.array([120., 176.66012, 176.66012])
sdlist = sd.siddons(trg, src, nPixels=(250,500,160), dPixels=.96, origin=(0, 0, 76.8),flat = True, ravel= True)



sd.siddons(Srcs, Trgs, nPixels=nPixelsG, dPixels=dPixel, origin=origin,\
            flat = True, ravel= True)


"""




