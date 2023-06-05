# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 08:27:59 2023

@author: vargasp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi, spatial

import vir.fwhm as fwhm
import vir.psf as psf

p0 = np.array([64,64,64])
p1 = np.array([64,64,70.5])

p0 = np.array([[64,64,64],[64,64,64]])
p1 = np.array([[64,64,70.5],[64,64,80.5]])



dPts = np.array(p1) - np.array(p0)
d = np.linalg.norm(dPts,axis=-1)
nPix = int(d.min())

coords = np.outer(dPts/d[...,np.newaxis], np.arange(nPix)) + np.array(p0)[:,np.newaxis]
print(coords)


# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 21:51:21 2023

@author: vargasp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi, spatial

import vir.fwhm as fwhm
import vir.psf as psf


x,y,z = fwhm.sphere_pts(samples=1000)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z, c = 'b', marker='.')

img3d = psf.gaussian3d()

s = np.array([64,64,64])
e = np.array([64,64,70.5])
dist = np.linalg.norm(e-s)
step = np.outer((e-s)/dist,np.arange(dist))+s[:,np.newaxis]


print(profile_line2(img3d,s,e))


def profile_line2(image, start_coords, end_coords):
    delta_coords = np.array(end_coords) - np.array(start_coords)

    dist = np.linalg.norm(delta_coords)

    coords = np.outer(delta_coords/dist, np.arange(dist)) + \
             np.array(start_coords)[:,np.newaxis]
             
    profile = ndi.map_coordinates(image, coords, order=2)
    return profile


def profile_line(image, start_coords, end_coords, *,
                 spacing=1, order=0, endpoint=True):
    coords = []
    n_points = int(np.ceil(spatial.distance.euclidean(start_coords, end_coords)
                           / spacing))
    for s, e in zip(start_coords, end_coords):
        print(s,e)
        coords.append(np.linspace(s, e, n_points, endpoint=endpoint))
    
    print(coords)
    profile = ndi.map_coordinates(image, coords, order=order)
    return profile






def polar2cart(r, theta, phi):
    return [
         r * math.sin(theta) * math.cos(phi),
         r * math.sin(theta) * math.sin(phi),
         r * math.cos(theta)
    ]