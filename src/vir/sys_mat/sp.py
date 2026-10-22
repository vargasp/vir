# -*- coding: utf-8 -*-
"""
Created on Fri Jan  2 19:44:02 2026

@author: varga
"""

import numpy as np



def sf_fp_fan_2d(
    img, angs, nu,
    DSO, DSD,
    du=1.0, d_pix=1.0
):
    """
    Separable-footprint fan-beam forward projector (2D).

    Parameters
    ----------
    img : (nx, ny) ndarray
    angs : (na,) ndarray, radians
    nu : int
        Number of detector bins
    DSO, DSD : float
    du : float
        Detector bin width
    d_pix : float
        Pixel width

    Returns
    -------
    sino : (na, nu) ndarray
    """

    nx, ny = img.shape
    na = len(angs)

    sino = np.zeros((na, nu), dtype=np.float32)

    # pixel centers
    x = d_pix * (np.arange(nx) - nx / 2 + 0.5)
    y = d_pix * (np.arange(ny) - ny / 2 + 0.5)

    # detector centers
    u = du * (np.arange(nu) - nu / 2 + 0.5)

    for ia, theta in enumerate(angs):
        c, s = np.cos(theta), np.sin(theta)

        for ix in range(nx):
            for iy in range(ny):
                val = img[ix, iy]
                if val == 0:
                    continue

                # rotate pixel center
                xp =  c * x[ix] + s * y[iy]
                yp = -s * x[ix] + c * y[iy]

                # project to detector
                u0 = DSD * xp / (DSO - yp)

                # footprint width (Jacobian)
                du_fp = DSD * d_pix / (DSO - yp)

                # affected bins
                umin = u0 - du_fp
                umax = u0 + du_fp

                iu0 = int(np.floor((umin / du) + nu / 2))
                iu1 = int(np.ceil ((umax / du) + nu / 2))

                for iu in range(iu0, iu1 + 1):
                    if iu < 0 or iu >= nu:
                        continue

                    # normalized distance
                    t = (u[iu] - u0) / du_fp

                    # linear B-spline footprint
                    w = max(0.0, 1.0 - abs(t))

                    sino[ia, iu] += val * w

    return sino * d_pix

# —————————————————————————————————————————
# Geometry & System Setup
# —————————————————————————————————————————

class CTGeometry:
    def __init__(self,
                 vol_shape,           # (nx, ny, nz)
                 voxel_size,          # (dx, dy, dz)
                 det_shape,           # (nu, nv)
                 det_spacing,         # (su, sv)
                 source_to_iso,       # distance source → iso
                 iso_to_det):         # distance iso → detector
        self.nx, self.ny, self.nz = vol_shape
        self.dx, self.dy, self.dz = voxel_size
        self.nu, self.nv = det_shape
        self.su, self.sv = det_spacing
        self.Ds = source_to_iso
        self.Dd = source_to_iso + iso_to_det

        # Voxel centers
        self.xs = (np.arange(self.nx) - self.nx/2 + 0.5)*self.dx
        self.ys = (np.arange(self.ny) - self.ny/2 + 0.5)*self.dy
        self.zs = (np.arange(self.nz) - self.nz/2 + 0.5)*self.dz

        # Detector coordinates
        self.u = (np.arange(self.nu) - self.nu/2 + 0.5)*self.su
        self.v = (np.arange(self.nv) - self.nv/2 + 0.5)*self.sv


# —————————————————————————————————————————
# Separable Footprint Basis Functions
# —————————————————————————————————————————

def footprint1D(r):
    """
    Simplest separable footprint.
    You can replace this with a triangular, Kaiser-Bessel, or
    other footprint depending on desired accuracy.
    """
    return np.clip(1 - np.abs(r), a_min=0, a_max=None)


# —————————————————————————————————————————
# Forward Projector
# —————————————————————————————————————————

def forward_project(volume, geom: CTGeometry):
    proj = np.zeros((geom.nv, geom.nu), dtype=np.float32)

    # Grid of pixel coordinates
    U, V = np.meshgrid(geom.u, geom.v, indexing='xy')

    # Loop detector
    for iv in range(geom.nv):
        for iu in range(geom.nu):
            u = U[iv, iu]
            v = V[iv, iu]

            # Source ray direction per detector pixel
            # Derived from similar triangles
            alpha = u / geom.Dd
            beta  = v / geom.Dd

            # Project ray to volume coords
            # Param t from source to voxel grid
            # Source is at (0,0,-Ds)
            # Intersection
            sum_val = 0.0
            for ix, x in enumerate(geom.xs):
                wx = footprint1D(alpha - x/(geom.Ds))
                if wx == 0: continue
                for iy, y in enumerate(geom.ys):
                    wy = footprint1D(beta  - y/(geom.Ds))
                    if wy == 0: continue
                    for iz, z in enumerate(geom.zs):
                        wz = footprint1D(z/geom.Ds)
                        sum_val += volume[ix, iy, iz]*wx*wy*wz
            proj[iv, iu] = sum_val

    return proj

# —————————————————————————————————————————
# Back Projector
# —————————————————————————————————————————

def back_project(proj, geom: CTGeometry):
    vol = np.zeros((geom.nx, geom.ny, geom.nz), dtype=np.float32)

    # Precompute detector coords
    U, V = np.meshgrid(geom.u, geom.v, indexing='xy')

    # Loop voxels
    for ix, x in enumerate(geom.xs):
        for iy, y in enumerate(geom.ys):
            for iz, z in enumerate(geom.zs):

                # Source to voxel direction
                # scaled distances
                px = x/geom.Ds
                py = y/geom.Ds
                pz = z/geom.Ds

                accum = 0.0
                for iv in range(geom.nv):
                    for iu in range(geom.nu):
                        u = U[iv, iu]
                        v = V[iv, iu]

                        # Det ray
                        alpha = u/geom.Dd
                        beta  = v/geom.Dd

                        w1 = footprint1D(alpha - px)
                        if w1 == 0: continue
                        w2 = footprint1D(beta  - py)
                        if w2 == 0: continue
                        w3 = footprint1D(pz)

                        accum += proj[iv, iu]*w1*w2*w3

                vol[ix, iy, iz] = accum

    return vol


# Create geometry
geom = CTGeometry(
    vol_shape=(128,128,128),
    voxel_size=(1.0,1.0,1.0),
    det_shape=(256,256),
    det_spacing=(1.0,1.0),
    source_to_iso=500,
    iso_to_det=500
)

# Random phantom
vol = np.random.rand(128,128,128).astype(np.float32)

# Forward projection
proj = forward_project(vol, geom)

# Back projection
rec = back_project(proj, geom)