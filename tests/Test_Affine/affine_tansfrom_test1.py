#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:39:39 2024

@author: pvargas21
"""

import numpy as np
import matplotlib.pyplot as plt

import vir.affine_transforms as af




nX = 128
nY = 128
nZ = 64
coords = af.coords_array((nX,nY), ones=True)


phantom2d = np.zeros([nX, nY])
#phantom2d[60:68,32:96] = 1
phantom2d[32:96,32:96] = 1


plt.imshow(phantom2d)
plt.show()

#1 , 32
#.5 , 16
#.25 , 24
#.125 , 28
#.0625 , 30


s = .25
c = 63.5

# (c - c*s)

S = af.scaleMat((s,s),center=( c,c))
S = np.linalg.inv(S)
SC = (S @ coords)
test = af.coords_transform(phantom2d, SC)

plt.imshow(test)
plt.show()



#plt.plot(test[56:72,64])
#plt.show()









coords = af.coords_array((nX,nY), ones=True)


angles = np.linspace(0,np.pi/2,1000)
angles_m = np.zeros(angles.size)


for i, angle in enumerate(angles):
    R = af.rotateMat((0,angle/100,0), center=(120,0))

    R = np.linalg.inv(R)
    RC = (R @ coords)
    test = af.coords_transform(phantom2d, RC)
    angles_m[i] = np.arccos(np.sum(test, axis=0).max()/2/32)


plt.plot(angles)
plt.plot(angles_m)
plt.show()






R = af.rotateMat((0.,0.0,0.), center=(64,0))
print(R)
RC = (R @ coords)
test = af.coords_transform(phantom2d, RC)
plt.imshow(test,origin='lower')
plt.show()


R = af.rotateMat((0.01,0,0.), center=(64,0))
print(R)
R = af.rotateMat((-0.01,0.), center=(64,0))
print(R)




R = af.rotateMat((0.,0.01,0.), center=(0,64))
print(R)
R = af.rotateMat((0.,-0.01,0.), center=(0,64))
print(R)




R = af.rotateMat((0.1,0.,0), center=(0,64))
print(R)
RC = (R @ coords)
test = af.coords_transform(phantom2d, RC)
plt.imshow(test,origin='lower')


samples = 240
xs = np.arange(-128,129,32)
ys = np.arange(-128,129,32)
tests1 = np.zeros([samples, nX*9, nY*9])
tests2 = np.zeros([samples, nX*9, nY*9])
angles = np.linspace(-.1, .1, samples)
for idx_x, x in enumerate(xs):
    for idx_y, y in enumerate(ys):
        for idx_a, angle in enumerate(angles):
            R = af.rotateMat((angle,0.,0.), center=(x,y))
            RC = (R @ coords)
            tests1[idx_a,(idx_x*nX):((idx_x+1)*nX),(idx_y*nY):((idx_y+1)*nY)] = af.coords_transform(phantom2d, RC)
        
            R = af.rotateMat((0.,angle,0.), center=(x,y))
            RC = (R @ coords)
            tests2[idx_a,(idx_x*nX):((idx_x+1)*nX),(idx_y*nY):((idx_y+1)*nY)] = af.coords_transform(phantom2d, RC)
        
b1 = np.arange(0,1152,128)
b2 = np.arange(127,1152,128)
tests1[:,:,b1] = 1.0
tests1[:,:,b2] = 1.0
tests1[:,b1,:] = 1.0
tests1[:,b2,:] = 1.0
tests2[:,:,b1] = 1.0
tests2[:,:,b2] = 1.0
tests2[:,b1,:] = 1.0
tests2[:,b2,:] = 1.0

vt.animated_gif(tests1, "outfile1", fps=24)
vt.animated_gif(tests2, "outfile2", fps=24)
        





samples = 960
arr = np.zeros([samples])
angles = np.linspace(-.1, .1, samples)
tests =np.zeros([samples, nX,nY])
for idx, angle in enumerate(angles):


vt.animated_gif(tests, "outfile1", fps=24)
        
        
        
        
        
        
plt.plot(angles,arr)