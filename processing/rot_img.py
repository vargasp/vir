#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 18:36:43 2022

@author: vargasp
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

import math
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage import transform
from skimage import img_as_float

img = img_as_float(data.chelsea())

def view1(img,x,y,a,b,c):
    Ty,Tx,maps = img.shape
    T = np.identity(3)
    T[0,2] = -Tx/2.
    T[1,2] = -Ty/2.
    
    matrix = np.identity(3)
    
    
    T1 = np.identity(3)
    T1[0,2] = x
    T1[1,2] = y
    
    rX = R.from_euler('x', a, degrees=True).as_matrix()
    rY = R.from_euler('y', b, degrees=True).as_matrix()
    rZ = R.from_euler('z', c, degrees=True).as_matrix()
    r = rX @ rY @ rZ
    matrix = matrix @ np.linalg.inv(T) @ T1 @ r @ T
    
    
    tform = transform.ProjectiveTransform(matrix=matrix)
    tf_img = transform.warp(img, tform.inverse)
    fig, ax = plt.subplots()
    ax.imshow(tf_img)
    ax.set_title('Projective transformation')
    plt.show()




def view2(img,x,y,z,a,b,c):
    Ty,Tx,maps = img.shape
    T = np.eye(4,3)
    T[0,2] = -Tx/2.
    T[1,2] = -Ty/2.
    T[2,2] = 0.0
    T[3,2] = 1.0
    
    Ti = np.eye(3,4)
    Ti[0,2] = Tx/2.
    Ti[1,2] = Ty/2.
    Ti[2,2] = 1.0
    
    rX44 = np.identity(4)
    rX = R.from_euler('x', a, degrees=True).as_matrix()
    rX44[:3,:3] = rX
    
    rY44 = np.identity(4)
    rY = R.from_euler('y', b, degrees=True).as_matrix()
    rY44[:3,:3] = rY
    
    rZ44 = np.identity(4)
    rZ = R.from_euler('z', c, degrees=True).as_matrix()
    rZ44[:3,:3] = rZ
    
    r = rX44 @ rY44 @ rZ44

    T1 = np.identity(4)
    T1[0,3] = x
    T1[1,3] = y
    T1[2,3] = z
    
    matrix =  Ti @ T1 @ r @ T
    
    
    
    tform = transform.ProjectiveTransform(matrix=matrix)
    tf_img = transform.warp(img, tform.inverse)
    fig, ax = plt.subplots()
    ax.imshow(tf_img)
    ax.set_title('Projective transformation')
    
    plt.show()
    


