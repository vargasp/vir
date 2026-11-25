# -*- coding: utf-8 -*-
"""
Analytic phantoms and sinograms for validating the vir.sys_mat projectors.

Geometry matches vir.sys_mat.pf:
    circular: source at DSO*(cos, sin, 0), detector centre at
              -(DSD-DSO)*(cos, sin, 0), u_hat = (-sin, cos, 0), v_hat = z.
    parallel: rays along (cos, sin), detector u axis (-sin, cos).
    square:   4 sides, side k is side 0 rotated by k*90 deg; source
              translated along the side (p) and z; detector rigidly attached
              to its source.  Layout [nsrc_p, nsrc_z, nv, nu, 4] (dd layout).

Shape tuples:  circle (x0, y0, r[, val]),  sphere (x0, y0, z0, r[, val])
All lengths in world units.

Quantities
    *line integral*    value * chord length.  Point-sampled at bin centres
                       by default; pass n_sub > 1 for Gauss-Legendre bin
                       averages (what pd and dd compute; rd samples centres).
    *volume*           volume of the object inside each pixel's frustum.  A
                       different physical quantity (sum over pixels = object
                       volume); not directly comparable to the projectors
                       except in the parallel limit (volume = mean line
                       integral * du * dv).

@author: varga
"""

import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull, QhullError

EPS = 1e-9


def _shape_val(s, n_geom):
    """Split (geom..., [val]) into geometry tuple and value (default 1)."""
    return tuple(s[:n_geom]), (s[n_geom] if len(s) > n_geom else 1.0)


def _gauss(n):
    """Gauss-Legendre nodes on [-1/2, 1/2] and weights summing to 1."""
    g, w = np.polynomial.legendre.leggauss(n)
    return 0.5*g, 0.5*w


def _sub_bins(arr, d, n_sub):
    """Bin centres -> (n, n_sub) Gauss nodes across each bin, and weights."""
    arr = np.asarray(arr, dtype=float)
    if n_sub <= 1 or (d is None and arr.size < 2):
        # No sub-sampling (or a single row whose width is unknown)
        return arr[:, None], np.ones(1)
    if d is None:
        d = arr[1] - arr[0]
    g, w = _gauss(n_sub)
    return arr[:, None] + d*g[None, :], w


#########################################
# Numerical phantoms - exact voxelization
#########################################

def discretized_sphere_exact(s, nx, ny, nz, d_pix=1.0, ns=8):
    """Voxelized sphere: each voxel = val * (fraction of voxel inside sphere).

    s = (x0, y0, z0, r[, val]) in world units; image centred on the origin
    (same convention as the projectors).  Voxels entirely inside/outside are
    set exactly; boundary voxels use ns x ns Gauss-Legendre quadrature in
    (x, y) with the exact z overlap (accuracy ~1e-4 relative: the integrand
    has a square-root singularity at the sphere's rim).
    """
    (x0, y0, z0, r), val = _shape_val(s, 4)

    def bnds(n, c):
        b = d_pix*(np.arange(n + 1) - n/2) - c
        return b[:-1], b[1:]

    xl, xr = bnds(nx, x0)
    yl, yr = bnds(ny, y0)
    zl, zr = bnds(nz, z0)
    XL, YL, ZL = np.meshgrid(xl, yl, zl, indexing='ij')
    XR, YR, ZR = np.meshgrid(xr, yr, zr, indexing='ij')

    def near(lo, hi):
        return np.clip(0.0, lo, hi)

    def far(lo, hi):
        return np.maximum(np.abs(lo), np.abs(hi))

    d_min = np.sqrt(near(XL, XR)**2 + near(YL, YR)**2 + near(ZL, ZR)**2)
    d_max = np.sqrt(far(XL, XR)**2 + far(YL, YR)**2 + far(ZL, ZR)**2)

    frac = np.zeros((nx, ny, nz))
    frac[d_max <= r] = 1.0
    bd = (d_min < r) & (d_max > r)
    if bd.any():
        g, w = _gauss(ns)
        xq = (0.5*(XL[bd] + XR[bd]))[:, None] + d_pix*g[None, :]     # (Nb, ns)
        yq = (0.5*(YL[bd] + YR[bd]))[:, None] + d_pix*g[None, :]
        h2 = r*r - xq[:, :, None]**2 - yq[:, None, :]**2              # (Nb, ns, ns)
        h = np.sqrt(np.clip(h2, 0, None))
        zlo = np.maximum(ZL[bd][:, None, None], -h)
        zhi = np.minimum(ZR[bd][:, None, None], h)
        ov = np.clip(zhi - zlo, 0, None)*(h2 > 0)
        frac[bd] = np.einsum('bij,i,j->b', ov, w, w)/d_pix
    return (val*frac).astype(np.float32)


def discretized_circle_exact(s, nx, ny, d_pix=1.0, ns=16):
    """2D analogue of discretized_sphere_exact; s = (x0, y0, r[, val])."""
    (x0, y0, r), val = _shape_val(s, 3)
    xb = d_pix*(np.arange(nx + 1) - nx/2) - x0
    yb = d_pix*(np.arange(ny + 1) - ny/2) - y0
    g, w = _gauss(ns)
    xq = 0.5*(xb[:-1] + xb[1:])[:, None] + d_pix*g[None, :]           # (nx, ns)
    h2 = r*r - xq**2
    h = np.sqrt(np.clip(h2, 0, None))
    ov = np.clip(np.minimum(yb[None, None, 1:], h[:, :, None])
                 - np.maximum(yb[None, None, :-1], -h[:, :, None]), 0, None)
    ov *= (h2 > 0)[:, :, None]
    frac = np.einsum('xqy,q->xy', ov, w)/d_pix
    return (val*frac).astype(np.float32)


def discretized_sphere_approx(s, nx, ny, nz, d_pix=1.0):
    """Cheap anti-aliased sphere: linear ramp of the signed distance over
    half a voxel diagonal.  s = (x0, y0, z0, r[, val])."""
    (x0, y0, z0, r), val = _shape_val(s, 4)
    x = d_pix*(np.arange(nx)[:, None, None] + 0.5 - nx/2) - x0
    y = d_pix*(np.arange(ny)[None, :, None] + 0.5 - ny/2) - y0
    z = d_pix*(np.arange(nz)[None, None, :] + 0.5 - nz/2) - z0
    d = np.sqrt(x*x + y*y + z*z) - r
    w = d_pix*np.sqrt(3)/2
    return (val*np.clip(0.5 - d/w, 0, 1)).astype(np.float32)


######################################################
# Line integrals - circles / spheres
######################################################

def analytic_sino_sphere_gen(src, det, sphere, rho=1.0):
    """Chord length * rho of src->det rays through a sphere (broadcasting).

    src, det : (..., 3);  sphere : (cx, cy, cz, r)
    """
    C = np.asarray(sphere[:3], dtype=float)
    r = sphere[3]
    v = np.asarray(det, dtype=float) - np.asarray(src, dtype=float)
    vhat = v/np.linalg.norm(v, axis=-1, keepdims=True)
    w = C - src
    t = np.sum(w*vhat, axis=-1)
    d2 = np.sum(w*w, axis=-1) - t*t
    return 2.0*rho*np.sqrt(np.clip(r*r - d2, 0, None))


def analytic_sino_circle_par(s, ang, u, du=None, n_sub=1):
    """Parallel-beam sinogram of a circle, shape ang.shape + u.shape.

    Signed distance from the centre to ray u is u - u_c with
    u_c = -sin*x0 + cos*y0 (the centre's detector coordinate).
    """
    (x0, y0, r), val = _shape_val(s, 3)
    ang = np.asarray(ang, dtype=float)
    u = np.asarray(u, dtype=float)
    u_sub, w = _sub_bins(u.ravel(), du, n_sub)
    u_c = (-np.sin(ang)*x0 + np.cos(ang)*y0)[..., None, None]
    p = u_sub - u_c
    out = 2.0*val*np.sqrt(np.clip(r*r - p*p, 0, None)) @ w
    return out.reshape(ang.shape + u.shape)


def analytic_sino_circle_fan(s, ang, u, DSO, DSD, du=None, n_sub=1):
    """Flat-detector fan-beam sinogram of a circle, shape ang.shape + u.shape.

    Ray u leaves the source at angle theta + pi - gamma, gamma = atan(u/DSD);
    its signed distance from (x0, y0) is
        x0*sin(theta-gamma) - y0*cos(theta-gamma) + DSO*sin(gamma).
    """
    (x0, y0, r), val = _shape_val(s, 3)
    ang = np.asarray(ang, dtype=float)
    u = np.asarray(u, dtype=float)
    u_sub, w = _sub_bins(u.ravel(), du, n_sub)
    gamma = np.arctan(u_sub/DSD)
    phi = ang[..., None, None] - gamma
    p = x0*np.sin(phi) - y0*np.cos(phi) + DSO*np.sin(gamma)
    out = 2.0*val*np.sqrt(np.clip(r*r - p*p, 0, None)) @ w
    return out.reshape(ang.shape + u.shape)


def analytic_sino_sphere_cone(s, ang, u, v, DSO, DSD, du=None, dv=None, n_sub=1):
    """Flat-panel cone-beam sinogram of a sphere, shape (na, nu, nv).

    Exact 3D point-to-line distance (the previous p_xy/p_z split was only an
    approximation, wrong by up to ~10% of the chord even for a centred
    sphere).  n_sub > 1 averages over each detector element.
    """
    (x0, y0, z0, r), val = _shape_val(s, 4)
    ang = np.atleast_1d(np.asarray(ang, dtype=float))
    u_sub, wu = _sub_bins(np.asarray(u).ravel(), du, n_sub)        # (nu, ku)
    v_sub, wv = _sub_bins(np.asarray(v).ravel(), dv, n_sub)        # (nv, kv)
    c = np.cos(ang)[:, None, None, None, None]
    sn = np.sin(ang)[:, None, None, None, None]
    U = u_sub[None, :, None, :, None]
    V = v_sub[None, None, :, None, :]
    src = np.stack(np.broadcast_arrays(DSO*c, DSO*sn, 0*c), -1)
    det = np.stack(np.broadcast_arrays(-(DSD - DSO)*c - U*sn,
                                       -(DSD - DSO)*sn + U*c, V + 0*c), -1)
    L = analytic_sino_sphere_gen(src, det, (x0, y0, z0, r), rho=val)
    return np.einsum('aijkl,k,l->aij', L, wu, wv)


def square_rays(nu, nv, nsrc_p, nsrc_z, DSO, DSD, du=1.0, dv=1.0, dsrc_p=1.0,
                dsrc_z=1.0, su=0.0, sv=0.0, ssrc_p=0.0, ssrc_z=0.0, n_sub=1):
    """Source and detector points of the square (translational) geometry,
    same conventions as rd/dd square.  Returns src, det with shape
    (4, nsrc_p, nsrc_z, nv, nu, kv, ku, 3) and Gauss weights (wv, wu)."""
    norm = np.array([[-1, 0], [0, -1], [1, 0], [0, 1]], dtype=float)
    side = np.array([[0, 1], [-1, 0], [0, -1], [1, 0]], dtype=float)
    u = du*(np.arange(nu) - (nu - 1)/2 + su)
    v = dv*(np.arange(nv) - (nv - 1)/2 + sv)
    src_p = dsrc_p*(np.arange(nsrc_p) - (nsrc_p - 1)/2) + ssrc_p
    src_z = dsrc_z*(np.arange(nsrc_z) - (nsrc_z - 1)/2) + ssrc_z
    u_sub, wu = _sub_bins(u, du, n_sub)
    v_sub, wv = _sub_bins(v, dv, n_sub)
    K = norm[:, 0][:, None, None, None, None, None, None]
    shp = (slice(None),) + (None,)*6
    nx_, ny_ = norm[:, 0][shp], norm[:, 1][shp]
    sx, sy = side[:, 0][shp], side[:, 1][shp]
    P = src_p[None, :, None, None, None, None, None]
    Z = src_z[None, None, :, None, None, None, None]
    U = u_sub[None, None, None, None, :, None, :]
    V = v_sub[None, None, None, :, None, :, None]
    src = np.stack(np.broadcast_arrays(-DSO*nx_ + P*sx, -DSO*ny_ + P*sy, Z + 0*K), -1)
    det = np.stack(np.broadcast_arrays(src[..., 0] + DSD*nx_ + U*sx,
                                       src[..., 1] + DSD*ny_ + U*sy, Z + V), -1)
    return src, det, wv, wu


def _square_avg(L, wv, wu):
    """(4, p, z, nv, nu, kv, ku) -> dd layout (p, z, nv, nu, 4)."""
    return np.einsum('spzvukl,k,l->pzvus', L, wv, wu)


def analytic_sino_sphere_square(s, nu, nv, nsrc_p, nsrc_z, DSO, DSD,
                                du=1.0, dv=1.0, dsrc_p=1.0, dsrc_z=1.0,
                                su=0.0, sv=0.0, ssrc_p=0.0, ssrc_z=0.0,
                                n_sub=1):
    """Square (translational) geometry sinogram of a sphere, matching
    rd.aw_fp_square / dd.dd_fp_square conventions.  Returns the dd layout
    [nsrc_p, nsrc_z, nv, nu, 4].  su, sv in bins; ssrc_* in world units."""
    (x0, y0, z0, r), val = _shape_val(s, 4)
    src, det, wv, wu = square_rays(nu, nv, nsrc_p, nsrc_z, DSO, DSD, du, dv, dsrc_p,
                                   dsrc_z, su, sv, ssrc_p, ssrc_z, n_sub)
    return _square_avg(analytic_sino_sphere_gen(src, det, (x0, y0, z0, r), rho=val), wv, wu)


def analytic_sino_polyhedron_square(planes, nu, nv, nsrc_p, nsrc_z, DSO, DSD,
                                    du=1.0, dv=1.0, dsrc_p=1.0, dsrc_z=1.0,
                                    su=0.0, sv=0.0, ssrc_p=0.0, ssrc_z=0.0,
                                    n_sub=1, rho=1.0):
    """Square geometry sinogram of a convex polyhedron (dd layout)."""
    src, det, wv, wu = square_rays(nu, nv, nsrc_p, nsrc_z, DSO, DSD, du, dv, dsrc_p,
                                   dsrc_z, su, sv, ssrc_p, ssrc_z, n_sub)
    return _square_avg(polyhedron_chords(src, det, planes, rho), wv, wu)


######################################################
# Line integrals - convex polyhedra
######################################################

def polyhedron_chords(src, det, planes, rho=1.0):
    """Chord length * rho of src -> det rays (broadcasting (..., 3)) through a
    convex polyhedron; planes (M, 4) [n, d] with interior n.x <= d."""
    src = np.asarray(src, dtype=float)
    det = np.asarray(det, dtype=float)
    planes = np.asarray(planes, dtype=float)
    d = det - src
    src, d = np.broadcast_arrays(src, d)
    ray_len = np.linalg.norm(d, axis=-1)
    # Rays are infinite lines: allow t outside [0, 1]
    t_enter = np.full(ray_len.shape, -np.inf)
    t_exit = np.full(ray_len.shape, np.inf)
    for plane in planes:
        n, c = plane[:3], plane[3]
        num = c - src @ n
        den = d @ n
        parallel = np.abs(den) < EPS*ray_len
        t = np.divide(num, den, out=np.zeros_like(den), where=~parallel)
        t_enter = np.where(~parallel & (den < 0), np.maximum(t_enter, t), t_enter)
        t_exit = np.where(~parallel & (den > 0), np.minimum(t_exit, t), t_exit)
        outside = parallel & (num < 0)             # parallel and src.n > c
        t_enter = np.where(outside, np.inf, t_enter)
        t_exit = np.where(outside, -np.inf, t_exit)
    L = np.where(t_exit > t_enter, (t_exit - t_enter)*ray_len, 0.0)
    return rho*np.nan_to_num(L, nan=0.0, posinf=0.0)


def analytic_sino_polyhedron_gen(src, det, u_cnt, v_cnt, u_hat, v_hat, planes):
    """Chord lengths of src -> detector-pixel rays through a convex polyhedron.

    det : (3,) detector centre;  u_cnt, v_cnt : pixel-centre offsets.
    planes : (M, 4) [nx, ny, nz, d] with interior  n.x <= d  (cube_planes).
    Rays are treated as infinite lines (object between source and detector).
    """
    uv = detector_grid(det, u_cnt, v_cnt, u_hat=u_hat, v_hat=v_hat)
    return polyhedron_chords(np.asarray(src, dtype=float), uv, planes)


def analytic_sino_polyhedron_cone(planes, ang, u, v, DSO, DSD, du=None, dv=None,
                                  n_sub=1, rho=1.0):
    """Cone-beam line integrals of a convex polyhedron, (na, nu, nv).
    n_sub > 1 averages over each detector element (Gauss-Legendre)."""
    ang = np.atleast_1d(np.asarray(ang, dtype=float))
    u_sub, wu = _sub_bins(np.asarray(u).ravel(), du, n_sub)
    v_sub, wv = _sub_bins(np.asarray(v).ravel(), dv, n_sub)
    c = np.cos(ang)[:, None, None, None, None]
    sn = np.sin(ang)[:, None, None, None, None]
    U = u_sub[None, :, None, :, None]
    V = v_sub[None, None, :, None, :]
    src = np.stack(np.broadcast_arrays(DSO*c, DSO*sn, 0*c), -1)
    det = np.stack(np.broadcast_arrays(-(DSD - DSO)*c - U*sn,
                                       -(DSD - DSO)*sn + U*c, V + 0*c), -1)
    L = polyhedron_chords(src, det, planes, rho)
    return np.einsum('aijkl,k,l->aij', L, wu, wv)


def analytic_sino_polyhedron_par(planes, ang, u, v=(0.0,), du=None, dv=None,
                                 n_sub=1, rho=1.0):
    """Parallel-beam line integrals of a convex polyhedron, (na, nu, nv);
    rays along (cos, sin, 0) through u*(-sin, cos, 0) + v*z."""
    ang = np.atleast_1d(np.asarray(ang, dtype=float))
    u_sub, wu = _sub_bins(np.asarray(u).ravel(), du, n_sub)
    v_sub, wv = _sub_bins(np.atleast_1d(np.asarray(v, dtype=float)), dv, n_sub)
    c = np.cos(ang)[:, None, None, None, None]
    sn = np.sin(ang)[:, None, None, None, None]
    U = u_sub[None, :, None, :, None]
    V = v_sub[None, None, :, None, :]
    p0 = np.stack(np.broadcast_arrays(-U*sn, U*c, V + 0*c), -1)
    dirn = np.stack(np.broadcast_arrays(c, sn, 0*c), -1)
    L = polyhedron_chords(p0, p0 + dirn, planes, rho)
    return np.einsum('aijkl,k,l->aij', L, wu, wv)


######################################################
# Volume integration (object volume inside each pixel's frustum)
######################################################

def analytic_sino_sphere_frustum_vol(src, det, u_bnd, v_bnd, u_hat, v_hat,
                                     sphere, rho=1.0, nodes=8):
    """Volume of a sphere inside each detector pixel's frustum (apex at the
    point source, truncated by the detector plane).

    V_pix = integral over the pixel of (t_out^3 - t_in^3)/3 * cos(psi)/R^2 dA
    (spherical coordinates about the source).  Gauss-Legendre (nodes^2 per
    pixel); sum over pixels -> 4/3 pi r^3 when the detector covers the sphere.
    """
    src = np.asarray(src, dtype=float)
    det = np.asarray(det, dtype=float)
    u_hat = np.asarray(u_hat, dtype=float)
    v_hat = np.asarray(v_hat, dtype=float)
    C = np.asarray(sphere[:3], dtype=float)
    r = sphere[3]
    u_bnd = np.asarray(u_bnd, dtype=float)
    v_bnd = np.asarray(v_bnd, dtype=float)
    du_arr = np.diff(u_bnd)
    dv_arr = np.diff(v_bnd)
    g, w = _gauss(nodes)
    uq = (0.5*(u_bnd[:-1] + u_bnd[1:]))[:, None] + du_arr[:, None]*g    # (nu, k)
    vq = (0.5*(v_bnd[:-1] + v_bnd[1:]))[:, None] + dv_arr[:, None]*g    # (nv, k)
    P = (det + uq[:, None, :, None, None]*u_hat
         + vq[None, :, None, :, None]*v_hat)                           # (nu,nv,k,k,3)
    ray = P - src
    R = np.linalg.norm(ray, axis=-1)
    rhat = ray/R[..., None]
    n_det = np.cross(u_hat, v_hat)
    n_det /= np.linalg.norm(n_det)
    cos_psi = np.abs(rhat @ n_det)
    wc = C - src
    tc = rhat @ wc
    h2 = r*r - (np.sum(wc*wc) - tc*tc)
    h = np.sqrt(np.clip(h2, 0, None))
    t_in = np.clip(tc - h, 0, R)
    t_out = np.clip(tc + h, 0, R)
    f = (t_out**3 - t_in**3)/3*cos_psi/R**2*(h2 > 0)
    return rho*np.einsum('ijkl,k,l->ij', f, w, w)*du_arr[:, None]*dv_arr[None, :]


def analytic_sino_sphere_gen_vol(src_centers, det_centers, eu, ev, du, dv, sphere,
                                 src_size=(0, 0), src_nodes=2, det_nodes=4, rho=1.0):
    """Pixel- (and focal-spot-) averaged line integrals of a sphere.

    Despite the name this is a *mean line integral* over each detector pixel
    (comparable to pd/dd), not a volume.  Focal-spot offsets are along the
    detector axes eu, ev (previously fixed world y, z, which is wrong for any
    rotated view).  Returns src.shape[:-1] + (Nu, Nv).
    """
    src_centers = np.atleast_2d(np.asarray(src_centers, dtype=float))
    eu = np.asarray(eu, dtype=float)
    ev = np.asarray(ev, dtype=float)
    src_shape = src_centers.shape[:-1]
    src_flat = src_centers.reshape(-1, 3)
    Nu, Nv, _ = det_centers.shape
    det_flat = det_centers.reshape(-1, 3)

    g, w = _gauss(det_nodes)
    det_off = (du*g[:, None, None]*eu + dv*g[None, :, None]*ev).reshape(-1, 3)
    det_w = np.outer(w, w).ravel()
    gs, ws = _gauss(src_nodes)
    src_off = (src_size[0]*gs[:, None, None]*eu
               + src_size[1]*gs[None, :, None]*ev).reshape(-1, 3)
    src_w = np.outer(ws, ws).ravel()

    proj = np.zeros((src_flat.shape[0], Nu*Nv))
    for doff, dw in zip(det_off, det_w):
        for soff, sw in zip(src_off, src_w):
            proj += dw*sw*analytic_sino_sphere_gen((src_flat + soff)[:, None, :],
                                                   (det_flat + doff)[None, :, :],
                                                   sphere, rho=rho)
    return proj.reshape(*src_shape, Nu, Nv)


def analytic_sino_sphere_cone_vol(s, ang, u, v, DSO, DSD, det_nodes=4):
    """Pixel-averaged cone-beam line integrals via analytic_sino_sphere_gen_vol."""
    (x0, y0, z0, r), val = _shape_val(s, 4)
    du = u[1] - u[0]
    dv = v[1] - v[0]
    src0 = np.array([DSO, 0, 0])
    det0 = np.array([-(DSD - DSO), 0, 0])
    det_cnts, src_cnts, u_hats, v_hats = circular_detector_geometry(ang, det0, src0)
    proj = np.empty([np.size(ang), np.size(u), np.size(v)])
    for i, (src_cnt, det_cnt, u_hat, v_hat) in enumerate(zip(src_cnts, det_cnts, u_hats, v_hats)):
        dets = detector_grid(det_cnt, u, v, u_hat=u_hat, v_hat=v_hat)
        proj[i] = analytic_sino_sphere_gen_vol(src_cnt, dets, u_hat, v_hat, du, dv,
                                               (x0, y0, z0, r), det_nodes=det_nodes,
                                               rho=val)[0]
    return proj


def convex_polyhedron_volume(vertices):
    if len(vertices) < 4:
        return 0.0
    try:
        return ConvexHull(vertices).volume
    except QhullError:
        return 0.0


def analytic_sino_polyhedron_gen_vol(src, det, u_bnd, v_bnd, u_hat, v_hat, planes_polyhedron):
    """Volume of a convex polyhedron inside each pixel frustum."""
    src = np.asarray(src, dtype=float)
    u_hat = np.asarray(u_hat, dtype=float)
    v_hat = np.asarray(v_hat, dtype=float)
    Nu = len(u_bnd) - 1
    Nv = len(v_bnd) - 1
    nfaces = planes_polyhedron.shape[0]

    uv = detector_grid(det, u_bnd, v_bnd, u_hat=u_hat, v_hat=v_hat)
    u_coord_bot, u_coord_top = uv[:, 0, :], uv[:, -1, :]
    v_coord_lft, v_coord_rgt = uv[0, :, :], uv[-1, :, :]

    # Orient frustum planes with the *detector* axes (a fixed world y/z
    # reference flips planes for rotated views).
    planes_u = plane_from_pts(src, u_coord_bot, u_coord_top, src + u_hat)
    planes_v = plane_from_pts(src, v_coord_lft, v_coord_rgt, src + v_hat)
    planes_uv = plane_from_pts(u_coord_bot[0], u_coord_top[0], u_coord_bot[-1], src)

    planes = np.empty([nfaces + 5, 4], dtype=float)
    planes[5:, :] = planes_polyhedron
    planes[4, :] = planes_uv
    proj = np.empty((Nu, Nv))
    for i in range(Nu):
        planes[0, :] = planes_u[i, :]
        planes[1, :] = -planes_u[i + 1, :]
        for j in range(Nv):
            planes[2, :] = planes_v[j, :]
            planes[3, :] = -planes_v[j + 1, :]
            proj[i, j] = convex_polyhedron_volume(intersection_vertices(planes))
    return proj


def analytic_sino_polyhedron_cone_vol(planes_polyhedron, ang, u_bnd, v_bnd, DSO, DSD):
    src0 = np.array([DSO, 0, 0])
    det0 = np.array([-(DSD - DSO), 0, 0])
    det_cnts, src_cnts, u_hats, v_hats = circular_detector_geometry(ang, det0, src0)
    proj = np.empty([np.size(ang), len(u_bnd) - 1, len(v_bnd) - 1])
    for i, (src_cnt, det_cnt, u_hat, v_hat) in enumerate(zip(src_cnts, det_cnts, u_hats, v_hats)):
        proj[i] = analytic_sino_polyhedron_gen_vol(src_cnt, det_cnt, u_bnd, v_bnd,
                                                   u_hat, v_hat, planes_polyhedron)
    return proj


def sphere_clip_volume(planes, center, radius, dirs, weights):
    """Volume of sphere ∩ {n.x <= d} by radial integration from the sphere
    centre over Lebedev directions (valid for any convex clip region).

    Lebedev points that lie exactly on a plane through the centre get half
    weight, so symmetric cases (e.g. a plane bisecting the sphere) are not
    biased by the tie.  Converges slowly when planes cut the sphere (the
    integrand has kinks); prefer analytic_sino_sphere_frustum_vol for pixels.
    """
    center = np.asarray(center, dtype=float)
    A = np.asarray(dirs) @ planes[:, :3].T                 # (Nq, M)
    b = planes[:, 3] - planes[:, :3] @ center              # (M,)
    tol = 1e-12
    with np.errstate(divide='ignore', invalid='ignore'):
        t = b[None, :]/A
    rmax = np.min(np.where(A > tol, t, np.inf), axis=1, initial=np.inf)
    rmin = np.max(np.where(A < -tol, t, -np.inf), axis=1, initial=-np.inf)
    par = np.abs(A) <= tol
    w_par = np.prod(np.where(par & (np.abs(b) <= tol), 0.5,
                             np.where(par & (b < 0), 0.0, 1.0)), axis=1)
    rmin = np.maximum(rmin, 0.0)
    rmax = np.minimum(rmax, radius)
    contrib = np.where(rmax > rmin, rmax**3 - rmin**3, 0.0)
    return float(np.sum(weights*w_par*contrib)/3.0)


def analytic_sino_sphere(src, det, u_bnd, v_bnd, u_hat, v_hat, center, radius,
                         rho=1.0, nodes=8):
    """Sphere volume in each pixel frustum, (Nu, Nv).  (Replaces the previous
    work-in-progress Lebedev version.)"""
    return analytic_sino_sphere_frustum_vol(src, det, u_bnd, v_bnd, u_hat, v_hat,
                                            (*center, radius), rho=rho, nodes=nodes)


analytic_sino_sphere2 = analytic_sino_sphere


##################
# Utility functions
##################

def detector_grid(det_center, u_arr, v_arr, u_hat=(1., 0., 0.), v_hat=(0., 0., 1.)):
    """(m, n, 3) coordinates det_center + u*u_hat + v*v_hat."""
    det_center = np.asarray(det_center, dtype=float)
    if det_center.shape != (3,):
        raise ValueError("det_center must be a single (3,) point")
    u_arr = np.asarray(u_arr, dtype=float)
    v_arr = np.asarray(v_arr, dtype=float)
    u_hat = np.asarray(u_hat, dtype=float)
    v_hat = np.asarray(v_hat, dtype=float)
    return (det_center[None, None, :] + u_arr[:, None, None]*u_hat[None, None, :]
            + v_arr[None, :, None]*v_hat[None, None, :])


def detector_grid_quadrature(det_center, u_arr, v_arr, du=None, dv=None,
                             det_nodes=(2, 2), u_hat=(1., 0., 0.), v_hat=(0., 0., 1.)):
    det_center = np.asarray(det_center, dtype=float)
    u_arr = np.asarray(u_arr, dtype=float)
    v_arr = np.asarray(v_arr, dtype=float)
    u_hat = np.asarray(u_hat, dtype=float)
    v_hat = np.asarray(v_hat, dtype=float)
    du = u_arr[1] - u_arr[0] if du is None else du
    dv = v_arr[1] - v_arr[0] if dv is None else dv
    gu, wu = np.polynomial.legendre.leggauss(det_nodes[0])
    gv, wv = np.polynomial.legendre.leggauss(det_nodes[1])
    Uq, Vq = np.meshgrid(gu, gv, indexing='ij')
    uq = (0.5*du*Uq).ravel()
    vq = (0.5*dv*Vq).ravel()
    wq = np.outer(wu, wv).ravel()
    det_positions = (det_center[None, None, None, :]
                     + (u_arr[:, None, None] + uq[None, None, :])[..., None]*u_hat
                     + (v_arr[None, :, None] + vq[None, None, :])[..., None]*v_hat)
    return det_positions, wq


def circular_detector_geometry(angles, det_center0, src_center0, isocenter=(0.0, 0.0, 0.0)):
    """Rotate source/detector about z.  Returns det_center, src_center, u_hat,
    v_hat (each (n, 3)).  u_hat = (-sin, cos, 0) for the default geometry."""
    angles = np.atleast_1d(np.asarray(angles, dtype=float))
    det_center0 = np.asarray(det_center0, dtype=float)
    src_center0 = np.asarray(src_center0, dtype=float)
    isocenter = np.asarray(isocenter, dtype=float)
    r0 = det_center0 - isocenter
    r1 = src_center0 - isocenter
    c, s = np.cos(angles), np.sin(angles)

    def rot(r):
        return np.column_stack([c*r[0] - s*r[1], s*r[0] + c*r[1],
                                np.full_like(c, r[2])]) + isocenter

    det_center = rot(r0)
    src_center = rot(r1)
    # Normal from detector towards source (robust when the detector sits at
    # the isocentre, DSD == DSO)
    n_hat = src_center - det_center
    n_hat[:, 2] = 0.0
    n_hat /= np.linalg.norm(n_hat, axis=1, keepdims=True)
    v_hat = np.tile([0.0, 0.0, 1.0], (len(angles), 1))
    u_hat = np.cross(v_hat, n_hat)
    u_hat /= np.linalg.norm(u_hat, axis=1, keepdims=True)
    return det_center, src_center, u_hat, v_hat


def cube_planes(center, side_lengths):
    """Axis-aligned box as planes [n, d] with interior n.x <= d."""
    center = np.asarray(center, dtype=float)
    mins = center - np.asarray(side_lengths, dtype=float)/2.
    maxs = center + np.asarray(side_lengths, dtype=float)/2.
    return np.array([[1., 0., 0., maxs[0]], [-1., 0., 0., -mins[0]],
                     [0., 1., 0., maxs[1]], [0., -1., 0., -mins[1]],
                     [0., 0., 1., maxs[2]], [0., 0., -1., -mins[2]]])


def plane_from_pts(p0, p1, p2, reference_point):
    """Plane [n, d] (n.x = d, |n| = 1) through three points, oriented so the
    reference point is on the negative side (n.x <= d).  Broadcasts."""
    n = np.cross(p1 - p0, p2 - p0)
    n = n/np.linalg.norm(n, axis=-1, keepdims=True)
    d = np.sum(n*p0, axis=-1)
    plane = np.concatenate((n, np.asarray(d)[..., None]), axis=-1)
    flip = np.sum(n*reference_point, axis=-1) > d + EPS
    return np.where(np.asarray(flip)[..., None], -plane, plane)


def intersection_vertices(planes):
    verts = []
    for idx in combinations(range(len(planes)), 3):
        A = planes[idx, :3]
        if abs(np.linalg.det(A)) < EPS:
            continue
        x = np.linalg.solve(A, planes[idx, 3])
        if np.all(planes[:, :3] @ x <= planes[:, 3] + EPS):
            verts.append(x)
    if not verts:
        return np.empty((0, 3))
    return np.unique(np.round(np.asarray(verts), 8), axis=0)


def unique_vertices(vertices, tol=1e-10):
    vertices = np.asarray(vertices, dtype=float)
    unique = []
    for v in vertices:
        if not any(np.linalg.norm(v - u) < tol for u in unique):
            unique.append(v)
    return np.asarray(unique)


# ----------------------------------------------------------------------------
# Names used by the test scripts
# ----------------------------------------------------------------------------
sphere_phantom_exact = discretized_sphere_exact
analytic_circle_sino_par_2d = analytic_sino_circle_par
analytic_circle_sino_fan_2d = analytic_sino_circle_fan
analytic_sphere_sino_cone_3d = analytic_sino_sphere_cone
analytic_sino_sphere_cone0 = analytic_sino_sphere_cone
sphere_projection_gauss = analytic_sino_sphere_gen_vol
