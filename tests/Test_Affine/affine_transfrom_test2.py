# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 15:14:44 2023

@author: vargasp
"""

import numpy as np
import matplotlib.pyplot as plt

import vir.affine_transforms as af
import vir.sinogram as sg

from scipy.ndimage import shift, rotate, zoom
from skimage.transform import rotate as rotate2


nX = 128
nY = 128
nZ = 64

phantom2d = np.zeros([nX, nY])
phantom2d[32:96,32:96] = 1
phantom2_1d = phantom2d[...,np.newaxis]
phantom3d = np.tile(phantom2d, (nZ,1,1))
phantom3d = phantom3d.transpose([1,2,0])
phantom3d *= np.arange(nZ)


"""
2d
"""
coords = af.coords_array((nX,nY), ones=True)
test = af.coords_transform(phantom2d, coords)
plt.imshow(test,origin='lower')

T = af.transMat((32,8))
T = np.linalg.inv(T)
TC = (T @ coords)
test = af.coords_transform(phantom2d, TC)
test2 = shift(phantom2d, (32,8), order=1)
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


S = af.scaleMat((1.10,1.10))
S = np.linalg.inv(S)
SC = (S @ coords)
test = af.coords_transform(phantom2d, SC)
test3 = zoom(phantom2d, (1.10,1.10), order=1)[:nX,:nY]

plt.imshow(test,origin='lower')
plt.show()
plt.imshow(test3,origin='lower')
plt.show()
plt.imshow(test3 - test,origin='lower')
plt.show()



"""
2.1d
"""
coords = af.coords_array((nX,nY,1), ones=True)
coords[:,:,2,:] = 0
T = af.transMat((16,16,0))
T = np.linalg.inv(T)
TC = (T @ coords)
test = af.coords_transform(phantom2_1d, TC)
plt.imshow(test,origin='lower')
plt.show()


R = af.rotateMat((0,0,np.pi/8), center=np.array(phantom2_1d.shape)/2.0 - 0.5)
R = np.linalg.inv(R)
RC = (R @ coords)
test = af.coords_transform(phantom2_1d, RC)
plt.imshow(test,origin='lower')
plt.show()


S = af.scaleMat((1.10,1.10,1.1), center=np.array(phantom2_1d.shape)/2.0 - 0.5)
S = np.linalg.inv(S)
SC = (S @ coords)
test = af.coords_transform(phantom2_1d, SC)
plt.imshow(test,origin='lower')
plt.show()



"""
3d
"""
coords = af.coords_array((nX,nY,nZ), ones=True)
T = af.transMat((32,16,8))
T = np.linalg.inv(T)
TC = (T @ coords)
test = af.coords_transform(phantom3d, TC)
plt.imshow(test[:,:,32],origin='lower')
plt.show()


coords = af.coords_array((nX,nY,nZ), ones=True)
R = af.rotateMat((np.pi/8,0,0), center=np.array(phantom3d.shape)/2.0-.5)
R = np.linalg.inv(R)
RC = (R @ coords)
test = af.coords_transform(phantom3d, RC)
plt.imshow(test[:,:,32],origin='lower')
plt.show()


coords = af.coords_array((nX,nY,nZ), ones=True)
R = af.rotateMat((0,np.pi/8,0), center=np.array(phantom3d.shape)/2.0-.5)
R = np.linalg.inv(R)
RC = (R @ coords)
test = af.coords_transform(phantom3d, RC)
plt.imshow(test[:,:,32],origin='lower')
plt.show()


coords = af.coords_array((nX,nY,nZ), ones=True)
R = af.rotateMat((0,0,np.pi/8), center=np.array(phantom3d.shape)/2.0-.5)
R = np.linalg.inv(R)
RC = (R @ coords)
test = af.coords_transform(phantom3d, RC)
plt.imshow(test[:,:,32],origin='lower')
plt.show()


S = af.scaleMat((1.10,1.10,1.10), center=np.array(phantom3d.shape)/2.0 - 0.5)
S = np.linalg.inv(S)
SC = (S @ coords)
test = af.coords_transform(phantom3d, SC)
plt.imshow(test[:,64,:],origin='lower')
plt.show()




phantom = np.zeros([nX*f, nY*f])
phantom[56*f:72*f,56*f:72*f] = 1
phantom = np.tile(phantom, (nZ,1,1))
phantom = phantom.transpose([1,2,0])
phantom *= np.arange(nZ)

nX, nY, nZ = phantom.shape

nAng = 256*f
angs = np.linspace(0,nAng,nAng,endpoint=False)



sino0 = sg.forward_project_wobble(phantom, angs, 0, 0, center=(nX/2.-.5,nY/2.-.5,0.5))
sino10 = sg.forward_project_wobble(phantom, angs, 10, 0, center=(nX/2.-.5,nY/2.-.5,0.5))

plt.imshow(sino0[0,:,:], origin='lower')
plt.imshow(sino10[0,:,:], origin='lower')



