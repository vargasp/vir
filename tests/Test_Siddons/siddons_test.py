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

import vir
import vir.siddon as sd
import time







nPix = (4,4,1)
dPix = 1.0

src = (-2, .5, 0.0)
trg = (2, .5, 0.0)
print(sd.siddons(trg, src, nPixels=nPix, dPixels=dPix))


src = (-2, -.5, 0.0)
trg = (2, -.5, 0.0)
print(sd.siddons(trg, src, nPixels=nPix, dPixels=dPix))

src = (-200, 2., 0.0)
trg = (20, 2., 0.0)
print(sd.siddons(trg, src, nPixels=nPix, dPixels=dPix))


src = (-2, 2, 0.0)
trg = (2, -2, 0.0)
print(sd.siddons(trg, src, nPixels=nPix, dPixels=dPix))






print(sd.siddons(trg, src, nPixels=nPix, dPixels=dPix, origin=0.0,\
            ravel=False, flat=False))
    
    
    
    
    
src = (10, 1.5, 0.0)
trg = (-10, 1.5, 0.0)
b = sd.siddons(trg, src, nPixels=nPix, dPixels=dPix, origin=0.0,\
            ravel=False, flat=False)

    

print(b)
        
    


nPix = (4,4,1)
dPix = 1.0

src = (10.5, 1.5, 0.0)
trg = (-10.5, 1.5, 0.0)
a = sd.siddons(trg, src, nPixels=nPix, dPixels=dPix, origin=0.0,\
            ravel=False, flat=False)
    
    
    
    
    
    
    
    
#Sheep Lgan Cicular
nPix = 4
nPixels = (nPix,nPix,1)
dPix = 1.0
nDets = nPix
dDet = 1.0
nTheta = 8
det_lets = 1
src_lets = 1

d = vir.Detector1d(nDets=nDets,dDet=dDet, det_lets=det_lets)

Thetas = np.linspace(0,2*np.pi,nTheta)
srcs, trgs = sd.circular_geom_st(d.Dets, Thetas, 10)
a = sd.siddons(srcs,trgs,nPixels, dPix)


a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
b = sd.siddons(srcs,trgs,nPixels, dPix, ravel=True)
c = sd.siddons(srcs,trgs,nPixels, dPix, ravel=False)
d = sd.list_flatten(b,nPixels, ravel=True)
e = sd.list_flatten(c,nPixels, ravel=False)
f = sd.list_ravel(c,nPixels)
g = sd.list_unravel(b,nPixels)

h = sd.list2array(a, nPixels, flat=True, nRays=nTheta*nDets)
i = sd.list2array(b, nPixels, ravel=True, nRays=nTheta*nDets)
j = sd.list2array(c, nPixels, ravel=False, nRays=nTheta*nDets)

h = sd.list_ave(a, flat=True, nPixels=nPixels, nRays=nTheta*nDets)
i = sd.list_ave(b, ravel=True, nPixels=nPixels, nRays=nTheta*nDets)
j = sd.list_ave(c, ravel=False, nPixels=nPixels, nRays=nTheta*nDets)



test = sd.list_flip(c.copy(),x=2,y=None,z=None,ravel=False,nPixels=nPixels)
test2 = sd.list_flip(b.copy(),x=2,y=None,z=None,ravel=True,nPixels=nPixels)




X = -256
Y = 0
Z = 0
src = sd.broadcast_pts(X,Y,Z)

X = 256
Y = vir.censpace(4,)
Z = vir.censpace(4,d=.5)
trg = sd.broadcast_pts(X,Y,Z)


nPixels = (16,4,1)
a =sd.siddons(src,trg,nPixels, 1.0, ravel = False)
sd.list_ave(a, ravel=False, nPixels=nPixels, axis=None)




X = -256
Y = 0
Z = 0
src = sd.broadcast_pts(X,Y,Z)

D = vir.Detector2d(nDets=4,dDet=128,det_lets=5)
X = 256
Y = D.W_lets
Z = 0
trg = sd.broadcast_pts(X,Y,Z)

nPixels = (512,512,1)
a =sd.siddons(src,trg,nPixels, 1.0, ravel = False)
sd.list_ave(a, ravel=False, nPixels=nPixels, axis=None)


b = sd.list2array(a, nPixels, ravel=False, ave=True)







sd.siddons3(srcs,trgs,nPixels, dPix, ravel=False)


inter_pix1,inter_len1 = sd.siddons_list2arrays(a)
inter_pix1A,inter_len1A = sd.average_inter(a)













#Sheep Lgan Cicular
nPix = 64
nPixels = (nPix,nPix,1)
dPix = 1.0
nDets = nPix
dDet = 1.0
nTheta = nPix*2


det_lets = 1
src_lets = 1
d = vir.Detector1d(nDets=nDets,dDet=dDet, det_lets=det_lets)
Thetas = np.linspace(0,np.pi,nTheta*src_lets)
Thetas = np.reshape(Thetas,(nTheta,src_lets))

srcs, trgs = sd.circular_geom_st(d.Det2_lets, Thetas, 128)
a = sd.siddons2(srcs,trgs,nPixels, dPix)
inter_pix1,inter_len1 = sd.siddons_list2arrays(a)
inter_pix1A,inter_len1A = sd.average_inter(a)


det_lets = 5
src_lets = 5
d = vir.Detector1d(nDets=nDets,dDet=dDet, det_lets=det_lets)
Thetas = np.linspace(0,np.pi,nTheta*src_lets)
Thetas = np.reshape(Thetas,(nTheta,src_lets))

srcs, trgs = sd.circular_geom_st(d.Det2_lets, Thetas, 128)
a = sd.siddons2(srcs,trgs,nPixels, dPix)
inter_pix2,inter_len2 = sd.siddons_list2arrays(a)
inter_pix2A,inter_len2A = sd.average_inter(a,ave_axes=(3,1))








#Siddons 2 Test


nPix = (4,4,1)
dPix = 1.0
nDets = (6)
dDet = 1.0
nTheta = 2

d = vir.Detector2d(nDets=(nDets,1),dDet=dDet)
src = np.stack([np.repeat(500,nDets), d.W, np.repeat(0,nDets)]).T
trg = np.stack([np.repeat(-500,nDets), d.W, np.repeat(0,nDets)]).T

thetas = np.linspace(0,np.pi,nTheta)

srcs = np.empty(thetas.shape + src.shape)
trgs = np.empty(thetas.shape + trg.shape)

for i, theta in enumerate(thetas):
    r_mat = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
    srcs[i,...] = src
    trgs[i,...] = trg
    srcs[i,...,:2] = np.matmul(np.array([src[...,0],src[...,1]]).T,r_mat)
    trgs[i,...,:2] = np.matmul(np.array([trg[...,0],trg[...,1]]).T,r_mat)


importlib.reload(sd)
a = sd.siddons2(srcs,trgs,nPix, dPix)
b,c = sd.siddons_list2arrays(a)
d,e = sd.siddons_list2arrays(a,mask=True)




C = np.zeros(np.prod(nPix))
for p_idx, pix in np.ndenumerate(b):
    #pix_u = np.unravel_index(pix, nPix)
    C[pix] += c[p_idx]
    

D = np.reshape(C,nPix)
E = np.sum(c,axis = 2)












theta = .4
x = np.repeat(5,11)
y = np.linspace(-5,5,11)
v = np.array([x,y])
r_mat = np.array([[np.cos(theta),np.sin(theta)],[-np.sin(theta), np.cos(theta)]])
np.matmul(r_mat, v)

a = sd.siddons(trg,src,nPix, dPix, return_format='unravel_idx')










nPix = 256
dPix = 1.0
pixels = vir.censpace(nPix,d=dPix)

trg_x = np.repeat(nPix/2,nPix**2)
trg_y = np.tile(pixels,nPix)
trg_z = np.repeat(pixels,nPix)

trg = list(zip(trg_x,trg_y,trg_z))

src = [-nPix/2,0,0]

importlib.reload(sd)
start = time.time()
k = sd.siddons(src, trg, nPix, dPix, return_array=True)
end = time.time()
print(end - start)
plt.imshow(k[:,int(nPix/2),:])







importlib.reload(sd)
nPix = (4,4,1)
dPix = 1.0

src = (10.5, 0.5, 0.0)
trg = (-10.5, 0.5, 0.0)
sd.siddons2(trg,src,nPix, dPix)
sd.siddons2(src,trg,nPix, dPix)




src = (0.5, 1.5, 0.5)
trg = (0.5, -1.5, 0.5)
a = sd.siddons(trg,src,nPix, dPix, return_format='unravel_idx')

src = (0.5, 0.5, 1.5)
trg = (0.5, 0.5, -1.5)
a = sd.siddons(trg,src,nPix, dPix, return_format='unravel_idx')




src = (2.5, 0.0, 0.5)
trg = (-2.5, 0.0, 0.5)
a = sd.siddons(trg,src,nPix, dPix, return_format='unravel_idx')

src = (2.5, 1.0, 0.5)
trg = (-2.5, 1.0, 0.5)
a = sd.siddons(trg,src,nPix, dPix, return_format='unravel_idx')




src = (2.5, -2.0, 0.5)
trg = (-2.5, -2.0, 0.5)
a= sd.siddons(trg,src,nPix, dPix, return_format='unravel_idx')

src = (2.5, 2.0, 0.5)
trg = (-2.5, 2.0, 0.5)
a= sd.siddons(trg,src,nPix, dPix, return_format='unravel_idx')


src = (2.5, -2.1, 0.5)
trg = (-2.5, -2.1, 0.5)
a = sd.siddons(trg,src,nPix, dPix, return_format='unravel_idx')

src = (2.5, 2.1, 0.5)
trg = (-2.5, 2.1, 0.5)
a = sd.siddons(trg,src,nPix, dPix, return_format='unravel_idx')




src = (-4.0, -4.0, -4.0)
trg = (8.0, 4.0, 4.0)
a = sd.siddons(trg,src,nPix, dPix, return_format='ravel_idx')


xs = np.linspace(-4,4,2000) 
ys = np.linspace(-4,4,2000) 
zs = np.linspace(-4,4,2000) 
for i, x in enumerate(xs):
    for j, y in enumerate(ys):
        for k, z in enumerate(zs):
            src = (x,y,z)
            trg = (4.0, 4.0, 4.0)
            print(i,j,k)
            a = sd.siddons(trg,src,nPix, dPix, return_format='ravel_idx')




importlib.reload(sd)
src = (xs[0],ys[1],zs[667])
trg = (4.0, 4.0, 4.0)
a = sd.siddons(trg,src,nPix, dPix, return_format='ravel_idx')




#Phantom Paramters



#Phantom Paramters
nPixels = (250,250,160)
dPixel = 0.96

src = [500, 311.8166274539262, 235.9766274539262]
trg = [-147.0, -0.72, -76.55999999999999]
a = sd.siddons(trg,src,nPixels, dPixel, return_format='ravel_idx')



