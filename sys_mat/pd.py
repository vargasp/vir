# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 15:27:50 2026

@author: varga
"""

import numpy as np

def proj_img2det_fan(px, py, ox, oy, DSO, DSD):
    return DSD * (px + py) / (DSO - (ox + oy))


def proj_object2det_par(px, py):
    return px + py


def accumuate(sino, val, ia, p_min, p_max, u_bnd, du, nu):
    # Overlapping detector bins
    iu0 = np.searchsorted(u_bnd, p_min, side="right") - 1
    iu1 = np.searchsorted(u_bnd, p_max, side="left")

    for iu in range(max(0, iu0), min(nu, iu1)):
        left = max(p_min, u_bnd[iu])
        right = min(p_max, u_bnd[iu + 1])
        if right > left:
            sino[ia, iu] += val * (right - left) / du



def pd_fp_par_2d(img, ang_arr, nu, du=1.0, su=0.0, d_pix=1.0):
    """
    Separable footprints forward projector for 2D parallel-beam CT.

    Implements the separable footprint model where each pixel is projected
    as an axis-aligned rectangle (“footprint”) onto the detector array.

    Parameters:
        img     : ndarray shape (nX, nY)
        Angs  : ndarray of projection angles (radians)
        nDets   : number of detector bins
        d_pix    : pixel width
        d_det    : detector bin width

    Returns:
        sino    : ndarray shape (len(angles), nDets)
    """
    nx, ny = img.shape
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)

    # Pixel boundaries (image centered at origin)
    x_bnd_arr = d_pix*(np.arange(nx+1, dtype=np.float32) - nx/2)
    y_bnd_arr = d_pix*(np.arange(ny+1, dtype=np.float32) - ny/2)

    # Detector bin boundaries (u)
    u_bnd_arr = du*(np.arange(nu + 1, dtype=np.float32) - nu/2 + su)

    # Precompute trig functions for all angles
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)


    for ia, (cos_ang,sin_ang) in enumerate(zip(cos_ang_arr,sin_ang_arr)):

        if abs(cos_ang) > abs(sin_ang):
            ray_len = 1/abs(cos_ang)
        else:
            ray_len =  1/abs(sin_ang)
        
        #Rotates the x,y coordinate system to o,p (orthogonal and parallel to
        #the detector)
        #In parallel beam these coordinates will projection directly on the 
        #detector with p(x,y) = -sin_ang*x + cos_ang*y

        #Precomputes the components
        px_bnd_arr = -sin_ang*x_bnd_arr
        py_bnd_arr =  cos_ang*y_bnd_arr
        
        #Loops through the precomputed x cooridnates  
        px_bnd_l = px_bnd_arr[0]
        for ix, px_bnd_r in enumerate(px_bnd_arr[1:]):
                                    
            # Initalize first boundary
            py_bnd_l = py_bnd_arr[0]    
            
            #Calulates the first two pixel corners projected on detector
            p0 = py_bnd_l + px_bnd_l
            p1 = py_bnd_l + px_bnd_r
            
            #Loops through the precomputed y cooridnates  
            for iy, py_bnd_r in enumerate(py_bnd_arr[1:]):
                #Calulates the second two pixel corners projected on detector
                p2 = py_bnd_r + px_bnd_l
                p3 = py_bnd_r + px_bnd_r


                #If image pixel is 0 advance to next pixel
                val = img[ix, iy]*ray_len
                if img[ix, iy] == 0:
                    py_bnd_l = py_bnd_r
                    p0 = p2
                    p1 = p3
                    continue

                #min and 
                p_min = min([p0, p1, p2, p3])
                p_max = max([p0, p1, p2, p3])

                accumuate(sino, val, ia, p_min, p_max, u_bnd_arr, du, nu)


                py_bnd_l = py_bnd_r
                p0 = p2
                p1 = p3

            px_bnd_l = px_bnd_r

    return sino


def pd_fp_fan_2d(img, ang_arr, nu, DSO, DSD, du=1.0, su=0.0, d_pix=1.0):
    """
    Pixel-driven separable-footprint fan-beam forward projector (2D).
    """
    nx, ny = img.shape
    sino = np.zeros((ang_arr.size, nu), dtype=np.float32)

    # Pixel boundaries (image centered at origin)
    x_bnd_arr = d_pix * (np.arange(nx + 1) - nx / 2)
    y_bnd_arr = d_pix * (np.arange(ny + 1) - ny / 2)

    # Detector bin boundaries (u)
    u_bnd_arr = du * (np.arange(nu + 1) - nu/2 + su)

    # Precompute trig functions for all angles
    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)
    
    for ia, (cos_ang,sin_ang) in enumerate(zip(cos_ang_arr,sin_ang_arr)):
            
        #Rotates the x,y coordinate system to o,p (orthogonal and parallel to
        #the detector)
        #In parallel beam these coordinates will projection directly on the 
        #detector with p(x,y = -sin_ang*x + cos_ang*y

        #Precomputes the components    
        px_bnd_arr = -sin_ang*x_bnd_arr
        py_bnd_arr =  cos_ang*y_bnd_arr

        ox_bnd_arr =  cos_ang*x_bnd_arr
        oy_bnd_arr =  sin_ang*y_bnd_arr
    
        #Loops through the precomputed x cooridnates  
        px_bnd_l = px_bnd_arr[0]
        ox_bnd_l = ox_bnd_arr[0]        
        for ix, (px_bnd_r,ox_bnd_r) in enumerate(zip(px_bnd_arr[1:],ox_bnd_arr[1:])):
    
            # Initalize first boundary
            py_bnd_l = py_bnd_arr[0]
            oy_bnd_l = oy_bnd_arr[0]
            
            #Calulates the first two pixel corners projected on detector
            p0 = proj_img2det_fan(px_bnd_l, py_bnd_l, ox_bnd_l, oy_bnd_l, DSO, DSD)
            p1 = proj_img2det_fan(px_bnd_r, py_bnd_l, ox_bnd_r, oy_bnd_l, DSO, DSD)

            #Loops through the precomputed y cooridnates              
            for iy, (py_bnd_r,oy_bnd_r) in enumerate(zip(py_bnd_arr[1:],oy_bnd_arr[1:])):

                #Calulates the second two pixel corners projected on detector    
                p2 = proj_img2det_fan(px_bnd_l, py_bnd_r, ox_bnd_l, oy_bnd_r, DSO, DSD)
                p3 = proj_img2det_fan(px_bnd_r, py_bnd_r, ox_bnd_r, oy_bnd_r, DSO, DSD)
        
                val = img[ix, iy]
                if val == 0:
                    py_bnd_l = py_bnd_r
                    oy_bnd_l = oy_bnd_r
                    p0 = p2
                    p1 = p3
                    continue

                p_min = min([p0,p1,p2,p3])
                p_max = max([p0,p1,p2,p3])

                accumuate(sino, val, ia, p_min, p_max, u_bnd_arr, du, nu)

                py_bnd_l = py_bnd_r
                oy_bnd_l = oy_bnd_r
                p0 = p2
                p1 = p3

            px_bnd_l = px_bnd_r
            ox_bnd_l = ox_bnd_r

    return sino


