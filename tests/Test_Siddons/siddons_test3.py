#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 22:16:09 2021

@author: vargasp
"""

import os
filename = os.environ.get('PYTHONSTARTUP')
if filename and os.path.isfile(filename):
    with open(filename) as fobj:
        startup_file = fobj.read()
    exec(startup_file)
    
    
    
import numpy as np
import matplotlib.pyplot as plt
import time
import importlib

#import vt
#import vir
import vir.siddon as sd
import vir.proj_geom as pg
import time


nPix = 4
nPixels = (nPix,nPix,1)
dPix = 1.0

importlib.reload(sd)

trg = np.array([3.0,.25,.25])
src = np.array([-3.0,.25,.25])

importlib.reload(sd)
#sd.calc_grid_lines(5,4,4,1.0,1,1.5)
    
sd.siddon_c(src,trg,nPixels,dPix)

a = sd.siddons(src,trg,nPixels,dPix)
print(a)

"""

nPix = 4
nPixels = (nPix,nPix,1)
dPix = 1.0

importlib.reload(sd)

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








    
#Sheep Lgan Cicular
nPix = 128
nPixels = (nPix,nPix,1)
dPix = 1.0
nDets = nPix
dDet = .5
nTheta = 16
det_lets = 1
src_lets = 1

d = vir.Detector1d(nDets=nDets,dDet=dDet, det_lets=det_lets)

Thetas = np.linspace(0,np.pi,nTheta,endpoint=False)
srcs, trgs = pg.geom_circular(d.Dets, Thetas,geom="par")
srcs[:,:,0] = -64
trgs[:,:,0] = 64

importlib.reload(sd)
a = sd.siddons(srcs,trgs,nPixels, dPix, flat=True)
b = sd.siddons(srcs,trgs,nPixels, dPix, ravel=True)
c = sd.siddons(srcs,trgs,nPixels, dPix, ravel=False)
h = sd.list2array(a, nPixels, flat=True)
i = sd.list2array(b, nPixels, ravel=True)
j = sd.list2array(c, nPixels, ravel=False)


vt.CreateImage(h[:,:,0])



print((h - i)[:,:,0])
print((h - j)[:,:,0])


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
"""


#test = sd.list_flip(c.copy(),x=2,y=None,z=None,ravel=False,nPixels=nPixels)
#test2 = sd.list_flip(b.copy(),x=2,y=None,z=None,ravel=True,nPixels=nPixels)


"""

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

"""









"""


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

srcs, trgs = pg.geom_circular(d.Det2_lets, Thetas,geom="par")
a = sd.siddons(srcs,trgs,nPixels, dPix)
inter_pix1,inter_len1 = sd.siddons_list2arrays(a)
inter_pix1A,inter_len1A = sd.average_inter(a)


det_lets = 5
src_lets = 5
d = vir.Detector1d(nDets=nDets,dDet=dDet, det_lets=det_lets)
Thetas = np.linspace(0,np.pi,nTheta*src_lets)
Thetas = np.reshape(Thetas,(nTheta,src_lets))

srcs, trgs = pg.geom_circulart(d.Det2_lets, Thetas,geom="par")
a = sd.siddons(srcs,trgs,nPixels, dPix)
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

"""

