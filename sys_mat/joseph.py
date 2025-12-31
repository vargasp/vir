# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 22:13:19 2025

@author: varga
"""
import numpy as np


def precompute_joseph(srcs, trgs, img_shape, origin, spacing, mode='ragged'):
    """
    Precompute Joseph ray geometry.

    Parameters
    ----------
    srcs : ndarray, shape [nBins, nAngles, 3]
    trgs : ndarray, shape [nBins, nAngles, 3]
    img_shape : tuple (nx,ny,nz)
    origin : tuple (x0,y0,z0)
    spacing : tuple (dx,dy,dz)
    mode : str, 'ragged' or 'flat'

    Returns
    -------
    If mode='ragged':
        rays : ndarray [nBins, nAngles], dtype=object
            Each element is dict {'base': (N,3), 'w': (N,8)}
    If mode='flat':
        base_flat : (total_samples,3)
        w_flat    : (total_samples,8)
        offsets   : (nRays+1,) start index per ray
        counts    : (nRays,) number of samples per ray
    """
    
    #Converts and assigns paramters to approprate data types
    trgs = np.array(trgs, dtype=np.float32)
    if trgs.ndim == 1:
        trgs = trgs[np.newaxis,:]

    srcs = np.array(srcs, dtype=np.float32)
    if srcs.ndim == 1:
        srcs = src[np.newaxis,:]
    
    
    nx, ny, nz = img_shape
    dx, dy, dz = spacing
    dt = min(dx, dy, dz)

    #Calculates deltas between target and source and Euclidean distance 
    dST  = trgs - srcs #(nRays, 3)
    distance = np.linalg.norm(dST, axis=-1) #(nRays)
    max_steps = np.ceil(distance/dt).astype(int)
    max_steps_max = np.ceil(distance.max()/dt).astype(int)

    nBins, nAngles, _ = srcs.shape
    nRays = nBins * nAngles

    if mode == 'ragged':
        Rays = np.empty(np.shape(trgs)[:-1], dtype=object)
    elif mode == 'flat':
        # estimate max_steps
        base_flat = np.empty((nRays*max_steps_max, 3), dtype=np.int32)
        w_flat    = np.empty((nRays*max_steps_max, 8), dtype=np.float32)
        offsets   = np.zeros(nRays+1, dtype=np.int32)
        counts    = np.zeros(nRays, dtype=np.int32)
        ptr = 0
        ray_idx = 0
    else:
        raise ValueError("mode must be 'ragged' or 'flat'")

    for i in range(nBins):
        for j in range(nAngles):
            direction = dST[i,j] / distance[i,j]

            if mode == 'ragged':
                base_list = []
                w_list = []
            elif mode == 'flat':
                idx = ptr

            for n in range(max_steps[i,j]):
                p = srcs[i,j] + n*dt*direction
                ix = (p[0]-origin[0])/dx
                iy = (p[1]-origin[1])/dy
                iz = (p[2]-origin[2])/dz

                if (0 <= ix < nx-1 and 0 <= iy < ny-1 and 0 <= iz < nz-1):                    
                    i0, j0, k0 = int(ix), int(iy), int(iz)
                    wx,wy,wz = ix-i0, iy-j0, iz-k0
    
                    weights = dt*np.array([(1-wx)*(1-wy)*(1-wz),
                                           wx*(1-wy)*(1-wz),
                                           (1-wx)*wy*(1-wz),
                                           (1-wx)*(1-wy)*wz,
                                           wx*wy*(1-wz),
                                           wx*(1-wy)*wz,
                                           (1-wx)*wy*wz,
                                           wx*wy*wz], dtype=np.float32)
    
                    if mode == 'ragged':
                        base_list.append((i0,j0,k0))
                        w_list.append(weights)
                    elif mode == 'flat':
                        base_flat[ptr,:] = (i0,j0,k0)
                        w_flat[ptr,:] = weights
                        ptr += 1

            if mode == 'ragged':
                Rays[i,j] = {
                    'base': np.array(base_list, dtype=np.int32),
                    'w': np.array(w_list, dtype=np.float32)
                }
            elif mode == 'flat':
                offsets[ray_idx] = idx
                counts[ray_idx] = ptr - idx
                ray_idx += 1

    if mode == 'flat':
        offsets[ray_idx] = ptr
        base_flat = base_flat[:ptr]
        w_flat = w_flat[:ptr]
        
        return base_flat, w_flat, offsets, counts
    else:
        return Rays


def joseph_forward(img, geometry, mode='ragged'):
    if mode == 'ragged':
        nBins, nAngles = geometry.shape
        proj = np.zeros((nBins, nAngles), dtype=np.float32)
        for i in range(nBins):
            for j in range(nAngles):
                ray = geometry[i,j]
                base = ray['base']
                w = ray['w']
                acc = 0.0
                for n in range(base.shape[0]):
                    i0,j0,k0 = base[n]
                    wn = w[n]
                    acc += (
                        img[i0, j0, k0]*wn[0] +
                        img[i0+1, j0, k0]*wn[1] +
                        img[i0, j0+1, k0]*wn[2] +
                        img[i0, j0, k0+1]*wn[3] +
                        img[i0+1,j0+1,k0]*wn[4] +
                        img[i0+1,j0,k0+1]*wn[5] +
                        img[i0,j0+1,k0+1]*wn[6] +
                        img[i0+1,j0+1,k0+1]*wn[7]
                    )
                proj[i,j] = acc
        return proj
    elif mode == 'flat':
        base_flat, w_flat, offsets, counts = geometry
        nRays = counts.shape[0]
        proj = np.zeros(nRays, dtype=np.float32)
        for r in range(nRays):
            acc = 0.0
            for n in range(counts[r]):
                idx = offsets[r]+n
                i0,j0,k0 = base_flat[idx]
                wn = w_flat[idx]
                acc += (
                    img[i0,j0,k0]*wn[0] +
                    img[i0+1,j0,k0]*wn[1] +
                    img[i0,j0+1,k0]*wn[2] +
                    img[i0,j0,k0+1]*wn[3] +
                    img[i0+1,j0+1,k0]*wn[4] +
                    img[i0+1,j0,k0+1]*wn[5] +
                    img[i0,j0+1,k0+1]*wn[6] +
                    img[i0+1,j0+1,k0+1]*wn[7]
                )
            proj[r] = acc
        return proj
    else:
        raise ValueError("mode must be 'ragged' or 'flat'")
        
        
def joseph_backproj(proj, geometry, img_shape, mode='ragged'):
    
    img = np.zeros(img_shape, dtype=np.float32)
    
    if mode == 'ragged':
        nBins, nAngles = geometry.shape
        for i in range(nBins):
            for j in range(nAngles):
                ray = geometry[i,j]
                base = ray['base']
                w = ray['w']
                pr = proj[i,j]
                for n in range(base.shape[0]):
                    i0,j0,k0 = base[n]
                    wn = w[n]
                    img[i0, j0, k0] += pr*wn[0]
                    img[i0+1, j0, k0] += pr*wn[1]
                    img[i0, j0+1, k0] += pr*wn[2]
                    img[i0, j0, k0+1] += pr*wn[3]
                    img[i0+1,j0+1,k0] += pr*wn[4]
                    img[i0+1,j0,k0+1] += pr*wn[5]
                    img[i0,j0+1,k0+1] += pr*wn[6]
                    img[i0+1,j0+1,k0+1] += pr*wn[7]
    elif mode == 'flat':
        base_flat, w_flat, offsets, counts = geometry
        nRays = offsets.shape[0] -1
        for r in range(nRays):
            pr = proj[r]
            for n in range(counts[r]):
                idx = offsets[r]+n
                i0,j0,k0 = base_flat[idx]
                wn = w_flat[idx]
                img[i0,j0,k0] += pr*wn[0]
                img[i0+1,j0,k0] += pr*wn[1]
                img[i0,j0+1,k0] += pr*wn[2]
                img[i0,j0,k0+1] += pr*wn[3]
                img[i0+1,j0+1,k0] += pr*wn[4]
                img[i0+1,j0,k0+1] += pr*wn[5]
                img[i0,j0+1,k0+1] += pr*wn[6]
                img[i0+1,j0+1,k0+1] += pr*wn[7]
    else:
        raise ValueError("mode must be 'ragged' or 'flat'")

    return img
        
import matplotlib.pyplot as plt

# --- Assume the precompute and Joseph forward/back functions are already defined ---
# precompute_joseph, joseph_forward, joseph_backproj

# --- 1. Create a simple 3D phantom ---
nx, ny, nz = 32, 32, 32
img = np.zeros((nx, ny, nz), dtype=np.float32)

# put a small cube in the center
img[12:20, 12:20, 12:20] = 1.0

# voxel spacing and origin
spacing = (1.0, 1.0, 1.0)
origin = (0.0, 0.0, 0.0)

# --- 2. Define parallel-beam circular geometry ---
nAngles = 30
nBins = 32  # number of detectors along x

radius = 40.0


angles = np.linspace(0, 2*np.pi, nAngles, endpoint=False)

sources = np.zeros((nBins, nAngles, 3), dtype=np.float32)
detectors = np.zeros_like(sources)

detector_spacing = 1.0
detector_offset = (-(nBins-1)/2) * detector_spacing

for j, theta in enumerate(angles):
    # unit vectors along the rotation plane
    sin_t, cos_t = np.sin(theta), np.cos(theta)

    # source position (rotating around center)
    src = np.array([radius*cos_t, radius*sin_t, 16.0])  # z=16 center
    # detector positions along x
    for i in range(nBins):
        det_x = detector_offset + i*detector_spacing
        det = np.array([-radius*cos_t + det_x*sin_t, -radius*sin_t - det_x*cos_t, 16.0])
        sources[i,j] = src
        detectors[i,j] = det



import vir.proj_geom as pg
import vir

#Parallel Beam Circular Trajectory
nPix = 32
nPixels = (nPix,nPix,1)
dPix = 1.0
nDets = nPix*4
dDet = .24
nTheta = 64
det_lets = 1
src_lets = 1

d = vir.Detector1d(nDets=nDets,dDet=dDet, det_lets=det_lets)
Thetas = np.linspace(0,np.pi,nTheta,endpoint=False) + np.pi/4


srcs, trgs = pg.geom_circular(d.Dets, Thetas,geom="par", src_iso=np.ceil(np.sqrt(2*(nPix*dPix/2.)**2)), det_iso=np.ceil(np.sqrt(2*(nPix*dPix/2.)**2)))
srcs[:,:,0] = srcs[:,:,0] + 16+.25
trgs[:,:,0] = trgs[:,:,0] + 16+.25
srcs[:,:,1] = srcs[:,:,1] + 16+.25
trgs[:,:,1] = trgs[:,:,1] + 16+.25
srcs[:,:,2] = 16.25
trgs[:,:,2] = 16.25



# --- 3. Precompute geometry ---
mode = 'flat'  # or 'flat'
geometry = precompute_joseph(srcs, trgs, (nx,ny,nz), origin, spacing, mode=mode)

# --- 4. Forward projection ---
proj = joseph_forward(img, geometry, mode=mode)

# --- 5. Backprojection ---
recon = joseph_backproj(proj, geometry, (nx,ny,nz), mode=mode)

# --- 6. Visualize central slice ---
fig, axs = plt.subplots(1,3, figsize=(12,4))
#axs[0].imshow(img[:,:,16], cmap='gray')
#axs[0].set_title('Original Phantom')
axs[0].plot(recon[:,16,16])
axs[0].plot(recon[:,15,16])



if mode=='ragged':
    proj_2d = proj
else:
    # reshape flat projections to [nBins, nAngles]
    proj_2d = proj.reshape(srcs.shape[:-1])

axs[1].imshow(proj_2d, cmap='gray', aspect='auto')
axs[1].set_title('Forward Projection')

axs[2].imshow(recon[:,:,16], cmap='gray')
axs[2].set_title('Backprojection')

plt.show()