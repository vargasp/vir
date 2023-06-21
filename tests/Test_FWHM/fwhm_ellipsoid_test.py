#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:11:35 2023

@author: pvargas21
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import EllipseModel
from scipy.interpolate import interp1d
import vir.fwhm as fwhm
import vir.psf as psf
import vir

img3d = psf.gaussian3d(mus=(2,10,10), sigmas=(5,10,15), nX=129,nY=129,nZ=129)


cX, cY, cZ = np.unravel_index(np.argmax(img3d), img3d.shape)
img2d = img3d[:,:,cZ]
img2d = img2d/np.max(img2d)
plt.imshow(img2d)

samples=128

#Calculates the x and y coordinates around a circle
x2, y2 = vir.sample_circle(radius=50.0, samples=samples)
r, theta = vir.cart2circ(x2,y2)
plt.plot(x2,y2,'r.')


#Calculates profile starting and ending points for line profiles 
p0 = np.tile([cX,cY],[samples,1])
p1 = np.vstack([x2,y2]).T + np.array([cX, cY])
profiles2 = fwhm.profile_img(img2d, p0, p1)
plt.imshow(profiles2)

#Calcualtes the FWHM at each profile
fwhms = np.zeros(samples)
for i in range(samples):
    fwhms[i] = fwhm.fwhm_psf_side_profile(profiles2[i,:])

#Calcualtes the x and y coordinates of the FWHM at each profile
xf, yf = vir.circ2cart(fwhms, theta)
plt.plot(xf,yf,'b.')

#Calcuates the ellipse parameters
params = fwhm.fit_error_ellipse(xf,yf)

#Calcuates the x and y coordinates of the fitted ellipse
xe, ye, = fwhm.ellipse_params2xy(params)
plt.plot(xf,yf,'g.')





img3d = img3d/np.max(img3d)
cX, cY, cZ = np.unravel_index(np.argmax(img3d), img3d.shape)

plt.imshow(img3d[cX,:,:])
plt.imshow(img3d[:,cY,:])
plt.imshow(img3d[:,:,cZ])


samples=1024

#Calculates the x, y, and z coordinates around a sphere
x3,y3,z3 = vir.sample_sphere(radius=50.0, samples=samples)
r, theta, phi = vir.cart2sph(x3,y3,z3)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x3,y3,z3, c = 'r', marker='.')

#Calculates profile starting and ending points for line profiles 
p0 = np.tile([cX,cY,cZ],[samples,1])
p1 = np.vstack([x3,y3,z3]).T + np.array([cX, cY, cZ])
profiles3 = fwhm.profile_img(img3d, p0, p1)
plt.imshow(profiles3)

#Calcualtes the FWHM at each profile
fwhms = np.zeros(samples)
for i in range(samples):
    fwhms[i] = fwhm.fwhm_psf_side_profile(profiles3[i,:])

#Calcualtes the x,y and z coordinates of the FWHM at each profile
xf, yf, zf = vir.sph2cart(fwhms, theta, phi)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xf,yf,zf, c = 'b', marker='.')

#Calcuates the ellipse parameters
lsvec = psf.ls_ellipsoid(xf, yf, zf)

psf.polyToParams3D(lsvec,True)

from scipy.spatial import Delaunay
tri = Delaunay(np.vstack([xf,yf,zf]).T)



