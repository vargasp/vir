# -*- coding: utf-8 -*-
"""
Created on Mon Jan 19 15:27:50 2026

@author: varga
"""

import numpy as np

def proj_img2det_fan(px, py, ox, oy, DSO, DSD):
    return DSD * (px + py) / (DSO - (ox + oy))


def proj_img2det_fan_mag(px, py, ox, oy, DSO, DSD):
    return DSD * (px + py) / (DSO - (ox + oy))


def proj_object2det_par(px, py):
    return px + py


def _accumuate_forward(sino, val, ia, p_min, p_max, u_bnd, du, nu, su):
    # Overlapping detector bins
    iu0 = max(0,  int(np.floor(p_min / du + nu/2 - su)))
    iu1 = min(nu, int(np.ceil (p_max / du + nu/2 - su)))

    for iu in range(iu0, iu1):
        left = max(p_min, u_bnd[iu])
        right = min(p_max, u_bnd[iu + 1])
        if right > left:
            sino[ia, iu] += val * (right - left) / du


def _accumuate_forward_3d(sino,val,ia, u_bnd,v_bnd,u_min,v_min,u_max,v_max, 
                          du,dv, nu, nv,su):
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
                sino[ia,iu,iv] += val*(ur - ul)/du*(vr - vl)/dv
                

def _accumuate_back_3d(img,sino,ix,iy,iz,ia,u_bnd,v_bnd,u_min,v_min,u_max,v_max, 
                          du,dv, nu, nv,su,scale):
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
                img[ix,iy,iz] += sino[ia,iu,iv]*(ur - ul)/du*(vr - vl)/dv*scale


def _accumuate_back(img, sino, ix, iy, ia, p_min, p_max, u_bnd, du, nu, su, scale):
    # Overlapping detector bins (same logic!)
    iu0 = max(0,  int(np.floor(p_min / du + nu/2 - su)))
    iu1 = min(nu, int(np.ceil (p_max / du + nu/2 - su)))

    for iu in range(iu0, iu1):
        left = max(p_min, u_bnd[iu])
        right = min(p_max, u_bnd[iu + 1])
        if right > left:
            img[ix, iy] += sino[ia, iu] * (right - left) / du *scale


def pd_fp_par_2d(img, ang_arr, nu, du=1.0, su=0.0, d_pix=1.0):
   nx, ny = img.shape
   sino = np.zeros((ang_arr.size, nu), dtype=np.float32)
   
   return pd_p_par_2d(img,sino,ang_arr,nx,ny,nu, 
               du=du,su=su,d_pix=d_pix, bp=False)

def pd_bp_par_2d(sino, ang_arr, img_shape, du=1.0, su=0.0, d_pix=1.0):
    nx, ny = img_shape
    na, nu = sino.shape
    img = np.zeros((nx, ny), dtype=np.float32)

    return pd_p_par_2d(img,sino,ang_arr,nx,ny,nu, 
                       du=du,su=su,d_pix=d_pix, bp=True)


def pd_p_par_2d(img,sino,ang_arr,nx,ny,nu, 
                du,su,d_pix,bp):
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

                p_min = min([p0, p1, p2, p3])
                p_max = max([p0, p1, p2, p3])

                if bp:
                    _accumuate_back(
                    img, sino, ix, iy, ia,
                    p_min, p_max, u_bnd_arr, du, nu, su,pix_scale)
                else:
                    if img[ix, iy] == 0:
                        py_bnd_l = py_bnd_r
                        p0 = p2
                        p1 = p3
                        continue
            
                    _accumuate_forward(sino, img[ix, iy]*pix_scale, ia, p_min, p_max, u_bnd_arr, du, nu, su)
        

                py_bnd_l = py_bnd_r
                p0 = p2
                p1 = p3

            px_bnd_l = px_bnd_r
    
    if bp:
        return img/ang_arr.size
    else:
        return sino


def pd_fp_fan_2d(img, ang_arr, nu, DSO, DSD, du=1.0, su=0.0, d_pix=1.0):
   nx, ny = img.shape
   sino = np.zeros((ang_arr.size, nu), dtype=np.float32)
   
   return pd_p_fan_2d(img,sino,ang_arr,nx,ny,nu, DSO, DSD, 
               du=du,su=su,d_pix=d_pix, bp=False)

def pd_bp_fan_2d(sino, ang_arr, img_shape, DSO, DSD, du=1.0, su=0.0, d_pix=1.0):
    nx, ny = img_shape
    na, nu = sino.shape
    img = np.zeros((nx, ny), dtype=np.float32)

    return pd_p_fan_2d(img,sino,ang_arr,nx,ny,nu, DSO, DSD, 
                       du=du,su=su,d_pix=d_pix, bp=True)



def pd_p_fan_2d(img,sino,ang_arr,nx,ny,nu,DSO,DSD,du,su,d_pix,bp):
    """
    Pixel-driven separable-footprint fan-beam forward projector (2D).
    """
 
    # Pixel boundaries (image centered at origin)
    x_bnd_arr = d_pix*(np.arange(nx + 1) - nx/2)
    y_bnd_arr = d_pix*(np.arange(ny + 1) - ny/2)

    # Detector bin boundaries (u)
    u_bnd_arr = du*(np.arange(nu + 1) - nu/2 + su)

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
        
                p_min = min([p0,p1,p2,p3])
                p_max = max([p0,p1,p2,p3])

                if bp:
                    _accumuate_back(
                    img, sino, ix, iy, ia,
                    p_min, p_max, u_bnd_arr, du, nu, su, ray_norm3)
                else:
                    if img[ix, iy] == 0:
                        py_bnd_l = py_bnd_r
                        p0 = p2
                        p1 = p3
                        continue
            
                    _accumuate_forward(sino, img[ix, iy]*ray_norm3, ia, p_min, p_max, u_bnd_arr, du, nu, su)
        

                py_bnd_l = py_bnd_r
                oy_bnd_l = oy_bnd_r
                p0 = p2
                p1 = p3

            px_bnd_l = px_bnd_r
            ox_bnd_l = ox_bnd_r

    if bp:
        return img/ang_arr.size
    else:
        return sino


def pd_fp_cone_3d(img,ang_arr,nu,nv,DSO,DSD,du=1.0,dv=1.0,su=0.0,sv=0.0,d_pix=1.0):
   nx, ny, nz  = img.shape
   sino = np.zeros((ang_arr.size, nu, nv), dtype=np.float32)
   
   return pd_p_cone_3d(img,sino,ang_arr,nx,ny,nz,nu,nv,DSO,DSD,
                        du=du,dv=dv,su=su,sv=sv,d_pix=d_pix,bp=False)


def pd_bp_cone_3d(sino,ang_arr,img_shape,DSO,DSD,du=1.0,dv=1.0,su=0.0,sv=0.0,d_pix=1.0):
    nx, ny, nz = img_shape
    na, nu, nv = sino.shape
    img = np.zeros((nx, ny, nz), dtype=np.float32)

    return pd_p_cone_3d(img,sino,ang_arr,nx,ny,nz,nu,nv,DSO,DSD,
                        du=du,dv=dv,su=su,sv=sv,d_pix=d_pix,bp=True)

def pd_p_cone_3d(img,sino,ang_arr,nx,ny,nz,nu,nv,DSO,DSD,
                 du,dv,su,sv,d_pix,bp):
    """
    Pixel-driven separable-footprint cone-beam forward projector (3D, circular).

    img : ndarray (nx, ny, nz)
    ang_arr : projection angles (radians)
    nu, nv : detector size
    """


    # Pixel boundaries
    x_bnd_arr = d_pix*(np.arange(nx + 1) - nx/2)
    y_bnd_arr = d_pix*(np.arange(ny + 1) - ny/2)
    z_bnd_arr = d_pix*(np.arange(nz + 1) - nz/2)

    # Detector bin boundaries
    u_bnd_arr = du*(np.arange(nu + 1) - nu/2 + su)
    v_bnd_arr = dv*(np.arange(nv + 1) - nv/2 + sv)

    cos_ang_arr = np.cos(ang_arr)
    sin_ang_arr = np.sin(ang_arr)

    for ia, (cos_ang, sin_ang) in enumerate(zip(cos_ang_arr, sin_ang_arr)):

        pix_scale = 1.0 / (abs(sin_ang) + abs(cos_ang))

        # Rotate image coordinates
        px_bnd_arr = -sin_ang * x_bnd_arr
        py_bnd_arr =  cos_ang * y_bnd_arr
        
        ox_bnd_arr =  cos_ang * x_bnd_arr
        oy_bnd_arr =  sin_ang * y_bnd_arr

        px_bnd_l = px_bnd_arr[0]
        ox_bnd_l = ox_bnd_arr[0]
        for ix, (px_bnd_r, ox_bnd_r) in enumerate(zip(px_bnd_arr[1:], ox_bnd_arr[1:])):

            py_bnd_l = py_bnd_arr[0]
            oy_bnd_l = oy_bnd_arr[0]

            # Calculated midpoint
            px_c = (px_bnd_l + px_bnd_r)/2
            ox_c = (ox_bnd_l + ox_bnd_r)/2


            for iy, (py_bnd_r, oy_bnd_r) in enumerate(zip(py_bnd_arr[1:], oy_bnd_arr[1:])):

                py_c = (py_bnd_l + py_bnd_r)/2
                oy_c = (oy_bnd_l + oy_bnd_r)/2


                # In-plane cone angle
                u_c = DSD * (px_c + py_c) / (DSO - (ox_c + oy_c))
                gamma = np.arctan(u_c / DSD)


                fan_magification = DSD/(DSO - (ox_c + oy_c))
                cone_magification = fan_magification/np.cos(gamma)
                u0 = fan_magification*(px_bnd_l + py_bnd_l)
                u1 = fan_magification*(px_bnd_r + py_bnd_l)
                u2 = fan_magification*(px_bnd_l + py_bnd_r)
                u3 = fan_magification*(px_bnd_r + py_bnd_r)


                #rx = np.cos(ang_arr[ia] + gamma)
                #ry = np.sin(ang_arr[ia] + gamma)

                z_bnd_l = z_bnd_arr[0]
                for iz, z_bnd_r in enumerate(z_bnd_arr[1:]):



                    #z_c = (z_bnd_l + z_bnd_r)/2
                    #rz = z_c / np.sqrt((DSO - (ox_c + oy_c))**2 + z_c**2)
                    #pix_scale = 1.0 / (abs(rx) + abs(ry) + abs(rz))


                    v0 = cone_magification*z_bnd_l 
                    v1 = cone_magification*z_bnd_r 

                    u_min = min(u0, u1, u2, u3)
                    u_max = max(u0, u1, u2, u3)
                    v_min = min(v0, v1)
                    v_max = max(v0, v1)

                    # Footprint stretch (separable)
                    ray_norm = pix_scale 


                    if bp:
                        _accumuate_back_3d(img,sino,ix,iy,iz,ia,u_bnd_arr,v_bnd_arr,
                                           u_min,v_min,u_max,v_max,
                                           du,dv, nu, nv,su,ray_norm)
                    else:
                        val = img[ix, iy, iz]
                        if val == 0:
                            z_bnd_l = z_bnd_r
                            continue
                        _accumuate_forward_3d(sino,val*ray_norm,ia, u_bnd_arr,v_bnd_arr,u_min,v_min,u_max,v_max, 
                                              du,dv, nu, nv,su)


                    z_bnd_l = z_bnd_r

                py_bnd_l = py_bnd_r
                oy_bnd_l = oy_bnd_r

            px_bnd_l = px_bnd_r
            ox_bnd_l = ox_bnd_r
    if bp:
        return img/ang_arr.size
    else:
        return sino

