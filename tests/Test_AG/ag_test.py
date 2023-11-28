#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 17:39:20 2023

@author: pvargas21
"""

import numpy as np

import matplotlib.pyplot as plt

import vir
import vir.analytic_geom as ag


def sa_pyramid(angle1,angle2):
   return 4*np.arcsin(np.sin(angle1/2) * np.sin(angle2/2)) 


def da_pyrmaid(apex, b1, b2, b3, b4):
    
    apex = np.array(apex)
    b1 = np.array(b1)
    b2 = np.array(b2)
    b3 = np.array(b3)
    b4 = np.array(b4)

    a12 = (b1 + b2)/2 - apex
    a23 = (b2 + b3)/2 - apex
    a34 = (b3 + b4)/2 - apex
    a41 = (b4 + b1)/2 - apex
    
    ang1 = np.arccos( np.dot(a12,a34) /np.linalg.norm(a12)/np.linalg.norm(a34))  
    ang2 = np.arccos( np.dot(a23,a41) /np.linalg.norm(a23)/np.linalg.norm(a41))  
    return ang1, ang2
    

def sa_cone(angle):
    """
    Calculates the solid angle of a cone

    Parameters
    ----------
    angle : float
        The angle between the side of the cone and the center ray [rad].

    Returns
    -------
    float
        The solid angle in steridians.

    """
    return 2*np.pi *(1 - np.cos(angle))



def xyx_a(ang1, ang2):
    x1,y1,z1 = vir.sph2cart(1, ang1/2, 0)
    x2,y2,z2 = vir.sph2cart(1, -ang1/2, 0)
    x3,y3,z3 = vir.sph2cart(1, 0, ang1/2)
    x4,y4,z4 = vir.sph2cart(1, 0, -ang1/2)
    
    return [x1,x2,x3,x4],[y1,y2,y3,y4], [z1,z2,z3,z4]
    
def print_rad(angle):
    print(f'{angle/np.pi:.5f}\N{GREEK SMALL LETTER PI}')


print_rad(sa_cone(np.pi))


x  = np.sqrt(3)*np.array([1, 1,-1,-1])
y  = np.sqrt(3)*np.array([1,-1,-1, 1])
z  = np.sqrt(3)*np.array([1,1,1,1])

ang1, ang2 = da_pyrmaid([0,0,0], [1,1,1], [1,-1,1], [-1,-1,1], [-1,1,1])
print_rad(sa_pyramid(ang1,ang2))


sa1 = ag.pyramid_sa(x, y, z)


