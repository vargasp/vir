# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:14:44 2023

@author: vargasp
"""

import numpy as np
import matplotlib.pyplot as plt

import vir.affine_transforms as af
import vir.sinogram as sg
import vt

from scipy.ndimage import affine_transform, shift, rotate
from skimage.transform import rotate as rotate2


def affine3d(arr, mat):
    return affine_transform(arr,mat,order=1,cval=0.0)


nX = 128
nY = 128
nZ = 64

phantom2d = np.zeros([nX, nY])
phantom2d[32:96,32:96] = 1
phantom3d = np.tile(phantom2d, (nZ,1,1))
phantom3d = phantom3d.transpose([1,2,0])
phantom3d *= np.arange(nZ)


"""
2d
"""
coords = af.coords_array((nX,nY), ones=True)
test = af.coords_transform(phantom2d, coords)
plt.imshow(test,origin='lower')

T = af.transMat((16,16))
T = np.linalg.inv(T)
TC = (T @ coords)
test = af.coords_transform(phantom2d, TC)
test2 = shift(phantom2d, (16,16), order=1)
plt.imshow(test,origin='lower')
plt.show()
plt.imshow(test2,origin='lower')
plt.show()


R = af.rotateMat(np.pi/8, center=np.array(phantom2d.shape)/2.0 - 0.5)
R = np.linalg.inv(R)
RC = (R @ coords)
test = af.coords_transform(phantom2d, RC)
test2 = rotate(phantom2d,180/8, order=1,reshape=False)
test3 = rotate2(phantom2d,180/8, order=1)

plt.imshow(test,origin='lower')
plt.show()
plt.imshow(test2,origin='lower')
plt.show()
plt.imshow(test3,origin='lower')
plt.show()


R = af.rotateMat(np.pi/8, center=(32,32))
R = np.linalg.inv(R)
RC = (R @ coords)
test = af.coords_transform(phantom2d, RC)
test3 = rotate2(phantom2d,180/8, center=(32,32), order=1)

plt.imshow(test,origin='lower')
plt.show()
plt.imshow(test3,origin='lower')
plt.show()


R = af.rotateMat(np.pi/8, center=np.array(phantom2d.shape)/2.0 - 0.5)
R = af.rotateMat((0,.01,0), center=(64,0))

R = np.linalg.inv(R)
RC = (R @ coords)
test = af.coords_transform(phantom2d, RC)
plt.imshow(test,origin='lower')
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


"""
2.1d
"""
coords = af.coords_array((nX,nY,1), ones=True)
coords[:,:,2,:] = 32
test = af.coords_transform(phantom, coords)
plt.imshow(test[:,:,0],origin='lower')


T = af.transMat((64,64,0))
TC = (T @ coords)
test = af.coords_transform(phantom, TC)
plt.imshow(test[:,:,0],origin='lower')


R = af.rotateMat((0,45,0), center=np.array(phantom.shape)/2.0)
RC = (R @ coords)
test = af.coords_transform(phantom, RC)
plt.imshow(test[:,:,0],origin='lower')


coords = af.coords_array((nX,nY,1), ones=True)
T = af.transMat((0,0,32),rank=None)
R = af.rotateMat((0,45,0), center=np.array(phantom.shape)/2.0)
RTC = (R @ T @ coords)
test = af.coords_transform(phantom, RTC)
plt.imshow(test[:,:,0],origin='lower')




nX = 128
nY = 128
nZ = 64
f =4

"""
3d
"""
phantom = np.zeros([nX*f, nY*f])
phantom[56*f:72*f,56*f:72*f] = 1
phantom = np.tile(phantom, (nZ,1,1))
phantom = phantom.transpose([1,2,0])
phantom *= np.arange(nZ)

nX, nY, nZ = phantom.shape

nAng = 256*f
angs = np.linspace(0,nAng,nAng,endpoint=False)


coords = af.coords_array((nX,nY,nZ), ones=True)
R = af.rotateMat((0,0,90), center=np.array(phantom.shape)/2.0-.5)
RC = (R @ coords)
test = af.coords_transform(phantom, np.round(RC,6))
(phantom - test).max()


sino0 = sg.forward_project_wobble(phantom, angs, 0, 0, center=(nX/2.-.5,nY/2.-.5,0.5))
sino10 = sg.forward_project_wobble(phantom, angs, 10, 0, center=(nX/2.-.5,nY/2.-.5,0.5))

plt.imshow(sino0[0,:,:], origin='lower')
plt.imshow(sino10[0,:,:], origin='lower')



