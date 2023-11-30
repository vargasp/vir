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


def oblique_pym(pa, pb1, pb2, pb3, pb4):
    
    def base_area(pB):
        #Circulent Base Vectors (b1- b2, b2-b3, ..., bn - b0)
        pB_circ = np.vstack([pB, pB[0,:]])
        
        #Calculates area from the summation of cross-products
        return np.linalg.norm(np.cross(pB_circ[:-1,:], pB_circ[1:,:]).sum(axis=0))/2.
    
    
    def solid_angle(vB_unit):
        vB_unit_circ = np.vstack([vB_unit[-1,:], vB_unit, vB_unit[0,:]])

        SA = 0.0
        for j in range(1,vB_unit_circ.shape[0] -1):
            #Cosines of the spherical triangle formed at the vertices
            a = np.dot(vB_unit_circ[j-1,:],vB_unit_circ[j+1,:])
            b = np.dot(vB_unit_circ[j-1,:],vB_unit_circ[j,:])
            c = np.dot(vB_unit_circ[j,:],vB_unit_circ[j+1,:])
            
            #Volume of the parallelepiped spannned by the vectors
            d =  np.dot(vB_unit_circ[j-1,:],np.cross(vB_unit_circ[j+1,:],vB_unit_circ[j,:]))
            
            SA += np.arctan2(d, b*c - a)

        return 2*np.pi - SA
        
        
    #Vertices
    pa = np.array(pa)
    pB = np.vstack([pb1,pb2,pb3,pb4])

    #Base Vectors and Magnitude
    vB = pB - pa   
    vB_mag = np.linalg.norm(vB, axis=1)
    vB_unit = vB/vB_mag[:,np.newaxis]  

    #Centroid Vector and Magnitude
    vc = pB.sum(axis=0)/pB.shape[0] - pa 
    vc_mag = np.linalg.norm(vc)
    vc_unit = vc/vc_mag

    #Effective height of right pyramid inscribed in pyrmaid 
    #Projection of closest base vertex to apex on Centroid Vector
    vEc = np.dot(vB[np.argmin(vB_mag),:], vc_unit)

    #Effective Base Vectors Magnitude
    vEB_mag = vEc/np.dot(vB_unit, vc_unit)
    vEB = vB_unit*vEB_mag[:,np.newaxis]

    #Calculates the base and effective base areas
    Area = base_area(pB)
    Eff_Area = base_area(vEB + pa)

    #Calculates the solid area
    SA1 = solid_angle(vB_unit)
    SA2 = solid_angle(vEB/vEB_mag[:,np.newaxis])


    return Area, Eff_Area, SA1, SA2



    """
    #Base Midpoint Vertices (b12, b23, b34, b41)
    pBm = (pB + np.roll(pB,-1,0))/2
    
    
    #Base vectors angles
    aB = np.arccos(np.dot(vB/vB_mag[:,np.newaxis], vc/vc_mag))
    """        

    






pa = [0,0,0]
pa = [1,1,0]
pb1 = [1,1,1]
pb2 = [1,-1,1]
pb3 = [-1,-1,1]
pb4 = [-1,1,1]


oblique_pym([1,1,.0], pb1,pb2,pb3,pb4)

print_rad(sa_cone(np.pi))


x  = np.sqrt(3)*np.array([1, 1,-1,-1])
y  = np.sqrt(3)*np.array([1,-1,-1, 1])
z  = np.sqrt(3)*np.array([1,1,1,1])

ang1, ang2 = da_pyrmaid([0,0,0], [1,1,1], [1,-1,1], [-1,-1,1], [-1,1,1])
print_rad(sa_pyramid(ang1,ang2))


sa1 = ag.pyramid_sa(x, y, z)


