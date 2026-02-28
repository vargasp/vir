# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 17:26:41 2026

@author: varga
"""


import matplotlib.pyplot as plt
import numpy as np
import vir.sys_mat.dd as dd
import time
import vir.sys_mat.pf as pf




#Half image grid + distacance to front of detector + midpoint of detecot
#(120m  + 21.38m + .01m)
DSO = 141.381
DSD = 261.381
du, dv = 2.61381, 2.61381




psino = np.load("C:\\Users\\varga\\Desktop\\sino_spheres_phantom_inter.npy")
img3d = np.load("C:\\Users\\varga\\Desktop\\phantomSpheres250.npy")

nx, ny, nz = img3d.shape
nu, nv, nsrc_p, nsrc_z, nsides = psino.shape
d_pix = .96
dsrc_p, dsrc_z = .48, .48
#ssrc_p, ssrc_z = 0.0, -74.4
#ssrc_p, ssrc_z = 0.0, -100
#ssrc_p, ssrc_z = 0.0, -50.4
#ssrc_p, ssrc_z = 0.0, -25
#ssrc_p, ssrc_z = 0.0, -40
ssrc_p, ssrc_z = 0.0, -74.4 #*DSO/DSD


su, sv = 0.0, 158.1355
su, sv = 0.0, 165
su, sv = 0.0, 170
su, sv = 0.0,  158.13554 #*DSO/DSD


z_bnd_arr = pf.boundspace(nz,d_pix)  # vertical
src_z_arr = pf.censpace(nsrc_z,dsrc_z,ssrc_z)
v_bnd_arr = pf.boundspace(nv,dv,sv +ssrc_z)


start = time.time()

dsino2 = dd.dd_fp_square(img3d,nu,nv,nsrc_p,nsrc_z,DSO,DSD,
                       du=du,dv=dv,dsrc_p=dsrc_p,dsrc_z=dsrc_z,
                       su=su,sv=sv,ssrc_p=ssrc_p,ssrc_z=ssrc_z,
                       d_pix=d_pix)

end = time.time()
print(end - start)







start = time.time()
recS = dd.dd_bp_square(dsino1,(nx,ny,nz), DSO,DSD,
                       du=du,dv=dv,dsrc_p=dsrc_p,dsrc_z=dsrc_z,
                       su=su,sv=sv,d_pixd_pix)

end = time.time()
print(end - start)






"""
# Thread-local accumulation buffer for a z-strip
tile_size = 8  # or tune based on L2/L3 cache
tile = np.zeros((tile_size, nu, 4), dtype=np.float32)

# Loop over z slices in tiles
for iz_start in range(0, nz, tile_size):
    iz_end = min(iz_start + tile_size, nz)
    tile[:,:,:] = 0.0  # reset local buffer

    # precompute projected bounds for this tile
    proj_z_l = proj_z_bnd_arr[iz_start:iz_end] - M*src_z_arr[:, None]
    proj_z_r = proj_z_bnd_arr[iz_start+1:iz_end+1] - M*src_z_arr[:, None]

    iv_min_tile = np.clip(((proj_z_l - v0) * inv_dv).astype(np.int32), 0, nv)
    iv_max_tile = np.clip(((proj_z_r - v0) * inv_dv).astype(np.int32)+1, 0, nv)

    for i_sp in range(nsrc_p):
        proj_src_p = proj_src_p_arr[i_sp]

        # compute iu_min/iu_max as before
        for ip in range(nP):
            p_l = proj_p_bnd_arr[ip] - proj_src_p
            p_r = proj_p_bnd_arr[ip+1] - proj_src_p
            iu_min = max(0, int((p_l - u_bnd[0]) * inv_du))
            iu_max = min(nu, int((p_r - u_bnd[0]) * inv_du)+1)
            if iu_min >= iu_max:
                continue

            v0f = colY[ip]
            v1f = colX[nP-1-ip]
            v2f = colYF[nP-1-ip]
            v3f = colXF[ip]

            for iu in range(iu_min, iu_max):
                tmp = np.array([v0f, v1f, v2f, v3f], dtype=np.float32)
                for iv_rel, (ivmin, ivmax) in enumerate(zip(iv_min_tile[i_sp,:], iv_max_tile[i_sp,:])):
                    tile[iv_rel, iu, :] += tmp * (overlap_v)  # compute overlap_v as before

    # Once tile is filled, write back contiguous block
    for iv_rel, iz in enumerate(range(iz_start, iz_end)):
        for iu in range(u_lo, u_hi):
            for f in range(4):
                sino[i_sp, i_sz, iv, iu, f] += tile[iv_rel, iu, f]
                
"""