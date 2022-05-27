#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 23:04:17 2021

@author: vargasp
"""
import numpy as np


def siddons(source, target, x_plane, y_plane, z_plane, \
            x_size, y_size, z_size, row):
    
    X = x_plane.size - 1
    Y = y_plane.size - 1
    Z = z_plane.size - 1    

    #Calculate the parametric values of the intersections of the line in question
    #with the first and last grid lines, both horizontal and vertical.
    source = np.array(source, dtype=float)
    target = np.array(target, dtype=float)
  
  
    dST  = target - source
    distance = np.linalg.norm(dST)
    print(dST)
    
    alpha1 = np.divide([x_plane[0],y_plane[0],z_plane[0]]-source, dST, out=np.zeros_like(source), where=dST!=0)
    alphaP = np.divide([x_plane[-1],y_plane[-1],z_plane[-1]]-source, dST, out=np.zeros_like(source), where=dST!=0)

    #Calculate alpha_min, which is either the parametric value of the intersection
    #where the line of interest enters the grid, or 0.0 if the source is inside the grid.
    #Calculate alpha_max, which is either the parametric value of the intersection
    #where the line of interest leaves the grid, or 1.0 if the target is inside the grid.
    m_min = np.zeros(4)
    m_max = np.ones(4)
    
    m_min[1:][alpha1 < alphaP] = alpha1[alpha1 < alphaP] 
    m_max[1:][alpha1 < alphaP] = alphaP[alpha1 < alphaP] 

    m_min[1:][alpha1 > alphaP] = alphaP[alpha1 > alphaP] 
    m_max[1:][alpha1 > alphaP] = alpha1[alpha1 > alphaP] 

    alpha_min = np.max(m_min)
    alpha_max = np.min(m_max)

    #If alpha_max <= alpha_min, then the ray doesn't pass through the grid.
    if alpha_max <= alpha_min:
        return 

    #Determine i_min, i_max, j_min, j_max, the indices of the first and last x and y planes
    #crossed by the line after entering the grid. We use ceil for mins and floor for maxs.
    if dST[0] > 0.0:
        i_min = np.floor(X - (x_plane[X] - alpha_min*dST[0] - source[0])/x_size).astype(int)
        i_max = np.ceil((source[0] + alpha_max*dST[0] - x_plane[0])/x_size - 1).astype(int)
    else:
        i_min = np.floor(X - (x_plane[X] - alpha_max*dST[0] - source[0])/x_size).astype(int)
        i_max = np.ceil((source[0] + alpha_min*dST[0] - x_plane[0])/x_size - 1).astype(int)

    if dST[1] > 0.0:
        j_min = np.floor(Y - (y_plane[Y] - alpha_min*dST[1] - source[1])/y_size).astype(int)
        j_max = np.ceil((source[1] + alpha_max*dST[1] - y_plane[0])/y_size - 1).astype(int)
    else:
        j_min = np.floor(Y - (y_plane[Y] - alpha_max*dST[1] - source[1])/y_size).astype(int)
        j_max = np.ceil((source[1] + alpha_min*dST[1] - y_plane[0])/y_size - 1).astype(int)

    if dST[2] > 0.0:
        k_min = np.floor(Z - (z_plane[Z] - alpha_min*dST[2] - source[2])/z_size).astype(int)
        k_max = np.ceil((source[2] + alpha_max*dST[2] - z_plane[0])/z_size - 1).astype(int)
    else:
        k_min = np.floor(Z - (z_plane[Z] - alpha_max*dST[2] - source[2])/z_size).astype(int)
        k_max = np.ceil((source[2] + alpha_min*dST[2] - z_plane[0])/z_size - 1).astype(int)


    print(i_min,i_max)
    print(j_min,j_max)
    print(k_min,k_max)


    #Compute the alpha values of the intersections of the line with all the relevant x planes in the grid.
    if dST[0] != 0.0:
        x_alpha = (x_plane[i_min:(i_max + 1)] - source[0])/dST[0]
    else:
        x_alpha = np.array([])
        
    #Compute the alpha values of the intersections of the line with all the relevant y planes in the grid.
    if dST[1] != 0.0 :
        y_alpha = (y_plane[j_min:(j_max + 1)] - source[1])/dST[1]
    else:
        y_alpha = np.array([])
        
    #Compute the alpha values of the intersections of the line with all the relevant z planes in the grid.
    if dST[2] != 0.0:
        z_alpha = (z_plane[k_min:(k_max + 1)] - source[2])/dST[2]
    else:
        z_alpha = np.array([])
        
    print(x_alpha)
    print(y_alpha)
    print(z_alpha)

    print(alpha_min, alpha_max)
    
    #Merges and sorts the alphas
    all_alpha = np.unique(np.concatenate([[alpha_min], x_alpha, y_alpha, z_alpha, [alpha_max]]))

    x_index = np.empty(all_alpha.size-1, dtype=int)
    y_index = np.empty(all_alpha.size-1, dtype=int)
    z_index = np.empty(all_alpha.size-1, dtype=int)
    length = np.empty(all_alpha.size-1)
    
    #Loops through the alphas and calculates pixel length and pixel index
    for a_idx in range(all_alpha.size-1):
        delta_alpha = all_alpha[a_idx+1]-all_alpha[a_idx]
        print(delta_alpha)
        alpha_sum = 0.5 * (all_alpha[a_idx+1] + all_alpha[a_idx])
        x_index[a_idx] = ((source[0] + alpha_sum*dST[0] - x_plane[0])/x_size).astype(int)
        y_index[a_idx] = ((source[1] + alpha_sum*dST[1] - y_plane[0])/y_size).astype(int)
        z_index[a_idx] = ((source[2] + alpha_sum*dST[2] - z_plane[0])/z_size).astype(int)
        length[a_idx] = distance*delta_alpha

    return x_index, y_index, z_index, length
