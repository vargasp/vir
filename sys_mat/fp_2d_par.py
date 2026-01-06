# -*- coding: utf-8 -*-
"""
Created on Thu Jan  1 08:57:56 2026

@author: varga
"""

import matplotlib.pyplot as plt
import numpy as np


def dd_fp_2d_sweep(sino,img,drive_axis,offset,Dets_bnd,nP,nP2,inv,iAng):
    
    for iP2 in range(nP2):
        #Projected boundaries at the detector plane
        P_bnd = drive_axis + offset[iP2]
        img_vec = img[:,iP2]
        
    
        iP = iDet = 0
        while iP < nP and iDet < nDets:
            left  = max(P_bnd[iP],   Dets_bnd[iDet])
            right = min(P_bnd[iP+1], Dets_bnd[iDet+1])
            if right > left:
                sino[iAng,iDet] += img_vec[iP] * (right-left) / inv
            
            # Advance to next pixel or detector
            if P_bnd[iP+1] < Dets_bnd[iDet+1]:
                iP += 1
            else:
                iDet += 1
        

def distance_driven_fp_2d(img, Angs, nDets, dPix=1.0, dDet=1.0):
    """
    Distance-driven forward projection for 2D parallel-beam CT.

    Parameters:
        img     : ndarray shape (nX, nY)
        Angs    : ndarray of projection Angs (radians)
        nDets   : number of detector bins
        dPix    : pixel width
        dDet    : detector bin width
    Returns:
        sino    : ndarray shape (len(angles), nDets) -- the sinogram
    """
    nX, nY = img.shape
    sino = np.zeros((Angs.size, nDets), dtype=np.float32)
    
    #Pixel boundaries coordinates
    X_bnd = dPix * (np.arange(nX+1) - nX/2.0)
    Y_bnd = dPix * (np.arange(nY+1) - nY/2.0)

    #Pixel center coordinates
    X_cnt = 0.5 * (X_bnd[:-1] + X_bnd[1:])
    Y_cnt = 0.5 * (Y_bnd[:-1] + Y_bnd[1:])

    #Detector coordinates
    Dets_bnd = dDet * (np.arange(nDets+1) - nDets/2.0)

    for iAng, angle in enumerate(Angs):
        ang_cos, ang_sin = np.cos(angle), np.sin(angle)
        
        # X-driven: projection axis is closer to X
        if abs(ang_cos) >= abs(ang_sin):
            nP1 = nX
            nP2 = nY
            inv = dDet*abs(ang_cos)
            
            #Rotate the driving axis
            drive_axis = ang_cos * X_bnd
            offset = ang_sin * -Y_cnt
            
            # If X_cos is decreasing, create reverse views of relevant axes
            if drive_axis[1] < drive_axis[0]:
                drive_axis = drive_axis[::-1]
                imgP = img[::-1, :]
            else:
                imgP = img
                        
        # Y-driven: projection axis is closer to Y
        else:
            nP1 = nY
            nP2 = nX
            inv = dDet*abs(ang_sin)
            drive_axis = ang_sin * -Y_bnd
            offset = ang_cos * X_cnt
            
            # If drive_axis is decreasing, create reverse views of relevant axes
            if drive_axis[1] < drive_axis[0]:
                drive_axis = drive_axis[::-1]
                imgP = img[:,::-1].T
            else:
                imgP = img.T
            
        dd_fp_2d_sweep(sino,imgP,drive_axis,offset,Dets_bnd,nP1,nP2,inv,iAng)

    return sino


def distance_driven_bp_2d(sino, Angs, nX, nY, dPix=1.0, dDet=1.0):
    """
    Adjoint distance-driven backprojection for 2D parallel-beam CT.

    Parameters:
        sino    : ndarray shape (len(Angs), nDets)
        Angs    : ndarray of projection angles (radians)
        nX, nY  : image size
        dPix    : pixel width
        dDet    : detector bin width

    Returns:
        img     : ndarray shape (nX, nY)
    """
    nAngs, nDets = sino.shape
    img = np.zeros((nX, nY), dtype=np.float32)

    # Pixel boundaries
    X_bnd = dPix * (np.arange(nX + 1) - nX / 2.0)
    Y_bnd = dPix * (np.arange(nY + 1) - nY / 2.0)

    # Pixel centers
    X_cnt = 0.5 * (X_bnd[:-1] + X_bnd[1:])
    Y_cnt = 0.5 * (Y_bnd[:-1] + Y_bnd[1:])

    # Detector boundaries
    Dets_bnd = dDet * (np.arange(nDets + 1) - nDets / 2.0)

    for iAng, angle in enumerate(Angs):
        ang_cos, ang_sin = np.cos(angle), np.sin(angle)
        proj = sino[iAng]

        # X-driven
        if abs(ang_cos) >= abs(ang_sin):
            X_cos = ang_cos * X_bnd
            Y_sin = ang_sin * -Y_cnt

            if X_cos[1] < X_cos[0]:
                X_cos = X_cos[::-1]
                flip_x = True
            else:
                flip_x = False

            for iY in range(nY):
                P_bnd = X_cos + Y_sin[iY]

                iP = iDet = 0
                while iP < nX and iDet < nDets:
                    left  = max(P_bnd[iP],   Dets_bnd[iDet])
                    right = min(P_bnd[iP+1], Dets_bnd[iDet+1])

                    if right > left:
                        val = proj[iDet] * (right - left) / (dDet * abs(ang_cos))
                        if flip_x:
                            img[nX - 1 - iP, iY] += val
                        else:
                            img[iP, iY] += val

                    if P_bnd[iP+1] < Dets_bnd[iDet+1]:
                        iP += 1
                    else:
                        iDet += 1

        # Y-driven
        else:
            Y_sin = ang_sin * -Y_bnd
            X_cos = ang_cos * X_cnt

            if Y_sin[1] < Y_sin[0]:
                Y_sin = Y_sin[::-1]
                flip_y = True
            else:
                flip_y = False

            for iX in range(nX):
                P_bnd = Y_sin + X_cos[iX]

                iP = iDet = 0
                while iP < nY and iDet < nDets:
                    left  = max(P_bnd[iP],   Dets_bnd[iDet])
                    right = min(P_bnd[iP+1], Dets_bnd[iDet+1])

                    if right > left:
                        val = proj[iDet] * (right - left) / (dDet * abs(ang_sin))
                        if flip_y:
                            img[iX, nY - 1 - iP] += val
                        else:
                            img[iX, iP] += val

                    if P_bnd[iP+1] < Dets_bnd[iDet+1]:
                        iP += 1
                    else:
                        iDet += 1

    return img




def siddons_fp_2d(img, Angs, nDets, dPix=1.0, dDet=1.0):
    nX, nY = img.shape
    sino = np.zeros((Angs.size, nDets), dtype=np.float32)

    # Image edges (centered)
    X_bnd = dPix * (np.arange(nX+1) - nX/2.0)
    Y_bnd = dPix * (np.arange(nY+1) - nY/2.0)
    
    # Detector bins (centered)
    Dets_cnt = dDet * (np.arange(nDets) - nDets/2.0 + 0.5)
    
    for iAng, ang in enumerate(Angs):
        dx, dy = np.cos(ang), np.sin(ang)
        
        for iDet, det_cnt in enumerate(Dets_cnt):
            # Setup ray origin *ON detector array, perpendicular to ray direction*
            # For parallel beam: select t so that for t = 0, detector axis at image center
            # ray passes through (x0, y0) = (det_center_offset along normal)
            x0 = det_cnt * -dy
            y0 = det_cnt *  dx

            # Calculate entry and exit parameters
            txs, tys = np.array([-np.inf, np.inf]), np.array([-np.inf, np.inf])
            if dx != 0:
                txs = (X_bnd - x0) / dx
            if dy != 0:
                tys = (Y_bnd - y0) / dy
            
            t0 = max(min(txs[0], txs[-1]), min(tys[0], tys[-1]))
            t1 = min(max(txs[0], txs[-1]), max(tys[0], tys[-1]))
            
            if t1 <= t0 or np.isinf(t0) or np.isinf(t1):
                continue
            # Find all boundary crossings
            cross_list = []
            if dx != 0:
                cross_list += list((X_bnd - x0) / dx)
            if dy != 0:
                cross_list += list((Y_bnd - y0) / dy)
            crosses = np.array(sorted([t for t in cross_list if t0<=t<=t1]))
            if crosses.size < 2:
                continue
            for seg in range(crosses.size-1):
                tc0, tc1 = crosses[seg], crosses[seg+1]
                if tc1<=tc0:
                    continue
                xc = x0 + 0.5*(tc0+tc1)*dx
                yc = y0 + 0.5*(tc0+tc1)*dy
                ix = int(np.floor((xc - X_bnd[0]) / dPix))
                iy = int(np.floor((yc - Y_bnd[0]) / dPix))
                if 0<=ix<nX and 0<=iy<nY:
                    sino[iAng,iDet] += img[ix,iy] * abs(tc1-tc0)
    return sino


def joseph_fp_2d(img, Angs, nDets, dPix=1.0, dDet=1.0):
    """
    Joseph's ray-interpolation forward projector for 2D parallel-beam CT.

    Parameters:
        img     : ndarray, shape (nX, nY)
        angles  : ndarray of projection angles [radians]
        nDets   : number of detector bins
        dPix    : pixel size (width)
        dDet    : detector bin width

    Returns:
        sino    : ndarray (len(angles), nDets)
    """
    img = np.ascontiguousarray(img, dtype=np.float32)
    Angs = np.asarray(Angs)
    nX, nY = img.shape
    sino = np.zeros((Angs.size, nDets), dtype=np.float32)
    
    # Grid: centered at zero, units in physical space
    x0 = -dPix*(nX + 1)/2.0  # First pixel center (x)
    y0 = -dPix*(nY + 1)/2.0  # First pixel center (y)

    Dets_cnt = dDet*(np.arange(nDets) - nDets / 2.0 + 0.5)



    # Project image grid boundaries onto the ray
    # Ray length: covers diagonal of image for safety
    L = dPix * max(nX, nY) * 2
 
    # Find t range so that we cover the whole image
    t0 = -L / 2
    t1 =  L / 2


    for iAng, angle in enumerate(Angs):
        ang_cos, ang_sin = np.cos(angle), np.sin(angle)
        
        # Ray direction is [cos_a, sin_a], detector axis is [-sin_a, cos_a]
        for iDet, det_cnt in enumerate(Dets_cnt):
            # Ray passes through (x_s, y_s)

            x_s = -ang_sin * det_cnt
            y_s =  ang_cos * det_cnt
           
            # Step size along ray (Joseph typically steps in 1-pixel increments)
            step = dPix / max(abs(ang_cos), abs(ang_sin))
            
            t = t0
            while t <= t1:
                # Current position along ray
                x = x_s + ang_cos * t
                y = y_s + ang_sin * t
                
                # Convert to pixel index
                ix = (x - x0) / dPix
                iy = (y - y0) / dPix
                
                if 0 <= ix < nX-1 and 0 <= iy < nY-1:
                    # Bilinear interpolation
                    ix0 = int(np.floor(ix))
                    iy0 = int(np.floor(iy))
                    dx = ix - ix0
                    dy = iy - iy0
                    
                    v00 = img[ix0  , iy0  ]
                    v01 = img[ix0  , iy0+1]
                    v10 = img[ix0+1, iy0  ]
                    v11 = img[ix0+1, iy0+1]
                    
                    val = (v00 * (1-dx)*(1-dy) +
                           v10 *    dx *(1-dy) +
                           v01 * (1-dx)*   dy  +
                           v11 *    dx *   dy)
                    sino[iAng, iDet] += val * step
                t += step
    return sino


def separable_footprint_fp_2d(img, Angs, nDets, dPix=1.0, dDet=1.0):
    """
    Separable footprints forward projector for 2D parallel-beam CT.

    Implements the separable footprint model where each pixel is projected
    as an axis-aligned rectangle (“footprint”) onto the detector array.

    Parameters:
        img     : ndarray shape (nX, nY)
        Angs  : ndarray of projection angles (radians)
        nDets   : number of detector bins
        dPix    : pixel width
        dDet    : detector bin width

    Returns:
        sino    : ndarray shape (len(angles), nDets)
    """
    nX, nY = img.shape
    sino = np.zeros((Angs.size, nDets), dtype=np.float32)

    # Detector edges:
    Dets_bnd = dDet * (np.arange(nDets+1) - nDets/2.0)

    # Image edges (centered)
    X_bnd = dPix * (np.arange(nX+1) - nX/2.0)
    Y_bnd = dPix * (np.arange(nY+1) - nY/2.0)

    for iAng, angle in enumerate(Angs):
        Angs_cos, Angs_sin = np.cos(angle), np.sin(angle)

        for iX, (xmin,xmax) in enumerate(zip(X_bnd[:-1],X_bnd[1:])):
            for iY, (ymin,ymax) in enumerate(zip(Y_bnd[:-1],Y_bnd[1:])):

                if img[iX, iY] == 0: continue
                
                # Project pixel corners to detector axis
                # These are the min/max of axis_u . (corner position)
                corners = [Angs_cos*ymin - Angs_sin*xmin, Angs_cos*ymax - Angs_sin*xmin,
                           Angs_cos*ymin - Angs_sin*xmax, Angs_cos*ymax - Angs_sin*xmax]

                P_min = min(corners)
                P_max = max(corners)

                # Footprint: find detector bins that overlap with rectangle projection

                # Bin k: between det_edges[k], det_edges[k+1]
                iDet0 = np.searchsorted(Dets_bnd, P_min, side='right') - 1
                iDetN = np.searchsorted(Dets_bnd, P_max, side='left')

                # For each overlapped bin, calculate geometric overlap
                for iDet in range(max(0, iDet0), min(nDets, iDetN)):
                    left = max(P_min, Dets_bnd[iDet])
                    right = min(P_max, Dets_bnd[iDet+1])
                    if right > left:
                        # Normalize by bin width
                        sino[iAng, iDet] += img[iX, iY] * (right - left) / dDet

    return sino







img = np.zeros((32, 32))
img[4:8, 4:8] = 1.0  # center impulse

angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
#angles = np.array([np.pi/4])
nDets = 64
dDet = .5

sino1 = siddons_fp_2d(img, angles, nDets=nDets,dDet=dDet)
sino2 = joseph_fp_2d(img, angles, nDets=nDets,dDet=dDet)
sino3 = distance_driven_fp_2d(img, angles, nDets=nDets,dDet=dDet)
sino4 = separable_footprint_fp_2d(img, angles, nDets,dDet=dDet)
print(sino1)
print(sino2)
print(sino3)
print(sino4)

plt.figure(figsize=(6,6))
plt.subplot(2,2,1)
plt.imshow(sino1, cmap='gray', aspect='auto')
plt.title("Siddons")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(2,2,2)
plt.imshow(sino2, cmap='gray', aspect='auto')
plt.title("Joseph")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(2,2,3)
plt.imshow(sino3, cmap='gray', aspect='auto')
plt.title("Distance-Driven")
plt.xlabel("Detector bin")
plt.ylabel("Angle")

plt.subplot(2,2,4)
plt.imshow(sino4, cmap='gray', aspect='auto')
plt.title("Seperable Footprint")
plt.xlabel("Detector bin")
plt.ylabel("Angle")
plt.tight_layout()
plt.show()

"""

dDet = .5
nDets = 64
nAngs = 32
r = 12.5

Dets  = dDet*(np.arange(nDets) - nDets / 2.0 + 0.5)


proj = 2*np.sqrt((r**2  - Dets**2).clip(0))
sino = np.tile(proj,[nAngs,1])

Angs = np.linspace(0,np.pi*2,32,endpoint=False)
nX = 32
nY = 32

rec = distance_driven_bp_2d(sino, Angs, nX, nY, dPix=1.0, dDet=dDet)

#plt.imshow(rec)

plt.plot(rec[:,16])
plt.plot(rec[16,:])
plt.show()

plt.plot(rec[:,16])
plt.plot(rec[:,15])
plt.show()


"""