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

        #Normaliz by footprint stretch factor
        pix_scale = 1/ (abs(sin_ang) + abs(cos_ang)) 
        
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
                val = img[ix, iy]*pix_scale
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

            # Calculated midpoint
            px_c = (px_bnd_l + px_bnd_r)/2
            ox_c = (ox_bnd_l + ox_bnd_r)/2

            
            #Calulates the first two pixel corners projected on detector
            p0 = proj_img2det_fan(px_bnd_l, py_bnd_l, ox_bnd_l, oy_bnd_l, DSO, DSD)
            p1 = proj_img2det_fan(px_bnd_r, py_bnd_l, ox_bnd_r, oy_bnd_l, DSO, DSD)

            #Loops through the precomputed y cooridnates              
            for iy, (py_bnd_r,oy_bnd_r) in enumerate(zip(py_bnd_arr[1:],oy_bnd_arr[1:])):


                py_c = (py_bnd_l + py_bnd_r)/2
                oy_c = (oy_bnd_l + oy_bnd_r)/2

                u_c = DSD * (px_c + py_c) / (DSO - (ox_c + oy_c))

                gamma = np.arctan(u_c / DSD)
                
                rx = np.cos(ang_arr[ia] + gamma)
                ry = np.sin(ang_arr[ia] + gamma)
                
                ray_norm3 = 1.0 / (abs(rx) + abs(ry))

                #Calulates the second two pixel corners projected on detector    
                p2 = proj_img2det_fan(px_bnd_l, py_bnd_r, ox_bnd_l, oy_bnd_r, DSO, DSD)
                p3 = proj_img2det_fan(px_bnd_r, py_bnd_r, ox_bnd_r, oy_bnd_r, DSO, DSD)
        
                val = img[ix, iy]*ray_norm3 # * ray_norm2
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



def pd_fp_cone_3d(img, ang_arr,
                  nu, nv,
                  DSO, DSD,
                  du=1.0, dv=1.0,
                  su=0.0, sv=0.0,
                  d_pix=1.0):
    """
    Pixel-driven separable-footprint cone-beam forward projector (3D, circular).

    img : ndarray (nx, ny, nz)
    ang_arr : projection angles (radians)
    nu, nv : detector size
    """

    nx, ny, nz = img.shape
    sino = np.zeros((ang_arr.size, nv, nu), dtype=np.float32)

    # Pixel boundaries
    x_bnd = d_pix * (np.arange(nx + 1) - nx / 2)
    y_bnd = d_pix * (np.arange(ny + 1) - ny / 2)
    z_bnd = d_pix * (np.arange(nz + 1) - nz / 2)

    # Detector bin boundaries
    u_bnd = du * (np.arange(nu + 1) - nu / 2 + su)
    v_bnd = dv * (np.arange(nv + 1) - nv / 2 + sv)

    cos_arr = np.cos(ang_arr)
    sin_arr = np.sin(ang_arr)

    for ia, (c, s) in enumerate(zip(cos_arr, sin_arr)):

        # Rotate image coordinates
        px_bnd = -s * x_bnd
        py_bnd =  c * y_bnd
        ox_bnd =  c * x_bnd
        oy_bnd =  s * y_bnd

        px_l = px_bnd[0]
        ox_l = ox_bnd[0]

        for ix, (px_r, ox_r) in enumerate(zip(px_bnd[1:], ox_bnd[1:])):

            py_l = py_bnd[0]
            oy_l = oy_bnd[0]

            # Midpoints for ray normalization
            px_c = 0.5 * (px_l + px_r)
            ox_c = 0.5 * (ox_l + ox_r)

            for iy, (py_r, oy_r) in enumerate(zip(py_bnd[1:], oy_bnd[1:])):

                py_c = 0.5 * (py_l + py_r)
                oy_c = 0.5 * (oy_l + oy_r)

                # In-plane cone angle
                u_c = DSD * (px_c + py_c) / (DSO - (ox_c + oy_c))
                gamma = np.arctan(u_c / DSD)

                rx = np.cos(ang_arr[ia] + gamma)
                ry = np.sin(ang_arr[ia] + gamma)

                ray_norm_xy = 1.0 / (abs(rx) + abs(ry))

                z_l = z_bnd[0]

                for iz, z_r in enumerate(z_bnd[1:]):

                    val = img[ix, iy, iz]
                    if val == 0:
                        z_l = z_r
                        continue

                    z_c = 0.5 * (z_l + z_r)

                    denom = DSO - (ox_c + oy_c)
                    u0 = DSD * (px_l + py_l) / denom
                    u1 = DSD * (px_r + py_l) / denom
                    u2 = DSD * (px_l + py_r) / denom
                    u3 = DSD * (px_r + py_r) / denom

                    v0 = DSD * z_l / denom
                    v1 = DSD * z_r / denom

                    u_min = min(u0, u1, u2, u3)
                    u_max = max(u0, u1, u2, u3)
                    v_min = min(v0, v1)
                    v_max = max(v0, v1)

                    # Footprint stretch (separable)
                    #ray_norm = ray_norm_xy / (abs(z_c) + denom / DSD)

                    ray_norm_z = denom/np.sqrt(denom**2 +z_c**2)
                    #ray_norm = ray_norm_xy * ray_norm_z

                    pix_scale = 1.0 / (abs(s) + abs(c))
                    p_c = px_c + py_c
                    ray_norm_xy = np.cos(np.arctan(p_c / (DSO - (ox_c + oy_c))))
                    ray_norm = ray_norm_xy * ray_norm_z*pix_scale

                    iu0 = np.searchsorted(u_bnd, u_min, side="right") - 1
                    iu1 = np.searchsorted(u_bnd, u_max, side="left")
                    iv0 = np.searchsorted(v_bnd, v_min, side="right") - 1
                    iv1 = np.searchsorted(v_bnd, v_max, side="left")

                    for iv in range(max(0, iv0), min(nv, iv1)):
                        vl = max(v_min, v_bnd[iv])
                        vr = min(v_max, v_bnd[iv + 1])
                        if vr <= vl:
                            continue

                        for iu in range(max(0, iu0), min(nu, iu1)):
                            ul = max(u_min, u_bnd[iu])
                            ur = min(u_max, u_bnd[iu + 1])
                            if ur > ul:
                                sino[ia, iv, iu] += (
                                    val * ray_norm *
                                    (ur - ul) / du *
                                    (vr - vl) / dv
                                )

                    z_l = z_r

                py_l = py_r
                oy_l = oy_r

            px_l = px_r
            ox_l = ox_r

    return sino