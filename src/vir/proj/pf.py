# -*- coding: utf-8 -*-
"""
Shared primitives for the rd (ray-driven), pd (pixel-driven / separable
footprint) and dd (distance-driven) projector modules.

Conventions used by every module
--------------------------------
Coordinates
    Image is centred on the origin.  Voxel ix spans
    [x_bnd_arr[ix], x_bnd_arr[ix+1]] with x_bnd_arr = boundspace(nx, d_pix).
    Gantry angle `ang`: source at  DSO*( cos, sin),  detector centre at
    -(DSD-DSO)*(cos, sin).  Detector u axis = (-sin, cos), v axis = z.
    Parallel beam uses the same u axis, rays travel along (cos, sin).

Rotated image coordinates (used by pd and dd)
    p = -sin*x + cos*y    (parallel to detector u)
    o =  cos*x + sin*y    (towards the source)
    Fan/cone magnification of a point: M = DSD / (DSO - o),  u = M*p, v = M*z

Detector shifts
    su, sv are in units of detector bins (a shift of su moves the detector
    by su*du).  Source shifts in the translational geometry (ssrc_p,
    ssrc_z) are in world units.

Forward projection
    Every forward projector returns line integrals (world-length units,
    multiplied by the attenuation value).  rd samples the line integral at
    the bin centre; pd and dd return the bin-averaged line integral.

Back projection
    Every back projector computes the exact adjoint A^T of its own forward
    projector, then (by default) multiplies by
        bp_norm = du*dv / (na * d_pix**ndim)
    which reproduces the previous 2D scaling of all three modules.
    Pass norm=False to get the bare adjoint (for iterative methods).

Affixes
    n*  count          d*  spacing        s*  shift          i*  index
    *_arr  1D array    *_bnd  boundaries  *_c  centre        *_l/_r  left/right
    *_min/_max         *_hat  unit vector *_vec  unnormalised vector
    *_org  ray origin  *_drv / *_orth  driving / orthogonal axis (dd)
    inv_*  reciprocal  t*_entry/t*_exit/t*_next/t*_step  ray parameters
"""

import numpy as np

try:
    from numba import njit, prange
except ImportError:                       # pure-python fallback
    def _identity_decorator(func):
        return func

    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]                # @njit
        return _identity_decorator        # @njit(...)

    prange = range


# Small epsilon: rays with |direction component| <= EPS are treated as
# parallel to that axis.
EPS = np.float32(1e-6)


# ----------------------------------------------------------------------------
# dtype helpers (python side)
# ----------------------------------------------------------------------------
def as_float32(*args):
    out = []
    for x in args:
        if np.isscalar(x):
            out.append(np.float32(x))
        else:
            out.append(np.ascontiguousarray(x, dtype=np.float32))
    return out


def as_int32(x):
    if np.isscalar(x):
        return np.int32(x)
    return np.asarray(x, dtype=np.int32)


def check_fov(img_shape, d_pix, DSO):
    """Divergent-beam models need every voxel in front of the source."""
    r_max = 0.5*d_pix*np.hypot(img_shape[0], img_shape[1])
    if r_max >= DSO:
        raise ValueError(f"Image half-diagonal ({r_max:g}) must be smaller "
                         f"than DSO ({DSO:g})")


def upsample(img, factor):
    """Replicate each voxel factor times along every axis."""
    for ax in range(img.ndim):
        img = np.repeat(img, factor, axis=ax)
    return img


def bp_norm(na, d_pix, du, dv=1.0, ndim=2):
    """Default back-projection normalisation (see module docstring)."""
    return np.float32(du * dv / (na * d_pix**ndim))


# ----------------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------------
@njit(inline='always', cache=True)
def censpace(n, d=1.0, s=0.0):
    """n centres, spacing d, centred on s."""
    arr = np.arange(n, dtype=np.float32)
    arr *= d
    arr += (s - d*((n - 1)*np.float32(0.5)))
    return arr


@njit(inline='always', cache=True)
def boundspace(n, d=1.0, s=0.0):
    """n+1 boundaries of n cells, spacing d, centred on s."""
    arr = np.arange(n + 1, dtype=np.float32)
    arr *= d
    arr += (s - d*(n*np.float32(0.5)))
    return arr


@njit(inline='always', cache=True)
def det_censpace(n, d, s_bins):
    """Detector bin centres; shift given in bins."""
    return censpace(n, d, s_bins*d)


@njit(inline='always', cache=True)
def det_boundspace(n, d, s_bins):
    """Detector bin boundaries; shift given in bins."""
    return boundspace(n, d, s_bins*d)


# ----------------------------------------------------------------------------
# Rotated coordinates and fan projection (pd, dd)
# ----------------------------------------------------------------------------
@njit(fastmath=True, inline='always', cache=True)
def rot_p(x, y, cos_ang, sin_ang):
    """Coordinate parallel to the detector u axis."""
    return -sin_ang*x + cos_ang*y


@njit(fastmath=True, inline='always', cache=True)
def rot_o(x, y, cos_ang, sin_ang):
    """Coordinate along the central ray, towards the source."""
    return cos_ang*x + sin_ang*y


@njit(fastmath=True, inline='always', cache=True)
def proj_fan(p, o, DSO, DSD):
    """Flat-detector projection of a point at (p, o): u = DSD*p/(DSO - o)."""
    return DSD*p/(DSO - o)


# ----------------------------------------------------------------------------
# Rays (rd) and ray directions (pd, dd weights)
# ----------------------------------------------------------------------------
@njit(fastmath=True, inline='always', cache=True)
def ray_par_2d(cos_ang, sin_ang, u):
    """Parallel-beam ray through detector coordinate u: origin, unit dir."""
    return -sin_ang*u, cos_ang*u, cos_ang, sin_ang


@njit(fastmath=True, inline='always', cache=True)
def ray_vec_fan(cos_ang, sin_ang, u, v, DSD):
    """Unnormalised source->detector vector for detector point (u, v)."""
    return -DSD*cos_ang - u*sin_ang, -DSD*sin_ang + u*cos_ang, v


@njit(fastmath=True, inline='always', cache=True)
def ray_cone_3d(cos_ang, sin_ang, u, v, DSO, DSD):
    """Cone-beam (fan if v=0) ray: origin, unit direction (float64).

    Built in float64 and re-anchored at the point of the ray closest to the
    isocentre, so ray parameters stay O(image size) even for DSO ~ 1e8
    (with a float32 origin at the source, positions lose ~DSO*6e-8 units).
    """
    c = np.float64(cos_ang)
    s = np.float64(sin_ang)
    ray_x_vec = -DSD*c - u*s
    ray_y_vec = -DSD*s + u*c
    ray_z_vec = np.float64(v)
    inv_mag = 1.0/np.sqrt(ray_x_vec*ray_x_vec + ray_y_vec*ray_y_vec
                          + ray_z_vec*ray_z_vec)
    ray_x_hat = ray_x_vec*inv_mag
    ray_y_hat = ray_y_vec*inv_mag
    ray_z_hat = ray_z_vec*inv_mag
    src_x = DSO*c
    src_y = DSO*s
    t0 = -(src_x*ray_x_hat + src_y*ray_y_hat)
    return (src_x + t0*ray_x_hat, src_y + t0*ray_y_hat, t0*ray_z_hat,
            ray_x_hat, ray_y_hat, ray_z_hat)


@njit(fastmath=True, cache=True)
def dd_weight_arr(cos_ang, sin_ang, u_arr, v_arr, DSD, fan, x_drv):
    """Distance-driven path-length weight per detector bin, shape (nu, nv).

    Path length through one slab of the orthogonal axis, per unit voxel
    size:  |r| / |r_orth|  with r the ray through the bin centre.
    x_drv=True: slabs are rows of constant y (r_orth = r_y).
    Parallel beam (fan=False): 1/|sin| or 1/|cos| (v is ignored).
    """
    nu = u_arr.size
    nv = v_arr.size
    w = np.empty((nu, nv), dtype=np.float32)
    for iu in range(nu):
        for iv in range(nv):
            if fan:
                rx, ry, rz = ray_vec_fan(cos_ang, sin_ang, u_arr[iu],
                                         v_arr[iv], DSD)
            else:
                rx, ry, rz = cos_ang, sin_ang, np.float32(0.0)
            r_orth = abs(ry) if x_drv else abs(rx)
            w[iu, iv] = np.sqrt(rx*rx + ry*ry + rz*rz)/r_orth
    return w


@njit(fastmath=True, inline='always', cache=True)
def cos_psi(u, v, DSD):
    """Cosine of the angle between the ray to detector point (u, v) and the
    central ray (flat detector)."""
    return DSD/np.sqrt(DSD*DSD + u*u + v*v)


@njit(fastmath=True, inline='always', cache=True)
def sort4(a, b, c, d):
    """Sort four scalars (5-comparator network)."""
    if a > b:
        a, b = b, a
    if c > d:
        c, d = d, c
    if a > c:
        a, c = c, a
    if b > d:
        b, d = d, b
    if b > c:
        b, c = c, b
    return a, b, c, d


@njit(fastmath=True, inline='always', cache=True)
def trap_cdf(x, t0, t1, t2, t3):
    """Integral from -inf to x of the unit-height trapezoid with corners
    t0 <= t1 <= t2 <= t3 (ramps may have zero width)."""
    if x <= t0:
        return 0.0
    if x >= t3:
        return 0.5*(t3 + t2 - t1 - t0)
    if x < t1:
        return 0.5*(x - t0)*(x - t0)/(t1 - t0)
    if x <= t2:
        return 0.5*(t1 - t0) + (x - t1)
    return 0.5*(t3 + t2 - t1 - t0) - 0.5*(t3 - x)*(t3 - x)/(t3 - t2)


@njit(fastmath=True, inline='always', cache=True)
def trap_overlap(a, b, t0, t1, t2, t3):
    """Integral of the unit trapezoid over [a, b]."""
    return trap_cdf(b, t0, t1, t2, t3) - trap_cdf(a, t0, t1, t2, t3)


# ----------------------------------------------------------------------------
# Ray/box intersection (rd)
# ----------------------------------------------------------------------------
@njit(fastmath=True, inline='always', cache=True)
def intersect_slab(r_org, r_hat, r_min, r_max):
    """Ray-parameter interval in which r_org + t*r_hat lies in [r_min, r_max].

    A ray parallel to the slab returns (-inf, inf) if it lies inside the
    slab and an empty interval (inf, -inf) if it lies outside.
    """
    if abs(r_hat) > EPS:
        t1 = (r_min - r_org)/r_hat
        t2 = (r_max - r_org)/r_hat
        return min(t1, t2), max(t1, t2)
    if r_min <= r_org and r_org <= r_max:
        return -np.inf, np.inf
    return np.inf, -np.inf


@njit(fastmath=True, inline='always', cache=True)
def entry_index(r_org, r_hat, t_entry, r_min, d_pix, n):
    """Voxel index containing the ray entry point, clamped to [0, n-1]."""
    r = r_org + t_entry*r_hat
    ir = int(np.floor((r - r_min)/d_pix))
    return min(max(ir, 0), n - 1)


@njit(fastmath=True, inline='always', cache=True)
def step_init(r_org, r_hat, ir, r_min, d_pix):
    """Amanatides-Woo stepping state along one axis.

    Returns (ir_dir, tr_step, tr_next).  Axes the ray is parallel to get
    infinite tr_step and tr_next so they are never selected.
    """
    if abs(r_hat) > EPS:
        if r_hat > 0:
            ir_dir = 1
            r_next = r_min + (ir + 1)*d_pix
        else:
            ir_dir = -1
            r_next = r_min + ir*d_pix
        return ir_dir, d_pix/abs(r_hat), (r_next - r_org)/r_hat
    return 1, np.inf, np.inf


# ----------------------------------------------------------------------------
# Interval overlap (pd, dd)
# ----------------------------------------------------------------------------
@njit(fastmath=True, inline='always', cache=True)
def overlap(a_l, a_r, b_l, b_r):
    """Length of [a_l, a_r] ∩ [b_l, b_r] (0 if disjoint)."""
    return max(min(a_r, b_r) - max(a_l, b_l), 0.0)


@njit(inline='always', cache=True)
def bin_range(bnd_arr, r_min, r_max):
    """Half-open index range [i0, i1) of cells of bnd_arr overlapping
    [r_min, r_max] (cells only touching at an edge are excluded)."""
    n = bnd_arr.size - 1
    i0 = np.searchsorted(bnd_arr, r_min, side='right') - 1
    i1 = np.searchsorted(bnd_arr, r_max, side='left')
    return max(i0, 0), min(i1, n)
