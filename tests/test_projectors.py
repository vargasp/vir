#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validation suite for vir.sys_mat projectors (rd, pd, dd) and analytic_sino.

Run:   pytest -q test_projectors.py          (or)   python test_projectors.py
       python test_projectors.py --size full --bench
Env:   VIR_TEST_SIZE=small|full   (small is fine without numba; full ~ 4x)

Coverage (what each of the original scripts was checking):
  test_orientation.py / test_fp_bp.py
      -> test_analytic_2d, test_analytic_cone: off-centre phantom vs analytic
         sinograms (accuracy + per-view centroid = orientation), all angles
         over 2*pi, detector shifts, rd vs point samples, pd/dd vs bin averages
      -> test_bp_localisation: bp of analytic sinograms peaks on the object
  test_dd.py
      -> test_bp_normalisation: single-view bp of a band of ones == 1
      -> test_fan_parallel_limit: DSO -> inf fan == parallel (fp and bp)
  test_fp_imp.py
      -> test_impulse_response: single off-centre voxel, footprint mass and
         centroid at the projected voxel centre, all methods
  test_scaling.py / test_fp_3d.py
      -> test_slab_plateau: slab/cube plateau == thickness at 0 and 45 deg,
         par/fan/cone
      -> test_dpix_scaling: fp scales with d_pix, normalised bp is invariant
  test_fp_siddons.py
      -> test_gridline_rays: rays exactly on pixel boundaries / image edges,
         near-parallel fan
  test_squre.py / build_sinos.py / m_testing.py
      -> test_square_vs_analytic: rd and dd square vs analytic square sinogram
         with source shifts (ssrc_p, ssrc_z), every side
      -> test_square_side_localisation: per-side bp localises the sphere
      -> test_square_su_sweep: fp/bp with detector shifts stays consistent
  (new) test_box_exact: voxel-aligned boxes vs exact line integrals (no
         phantom discretization error; rd AW must match to ~1e-5)
  (new) test_phantom_convergence, test_adjoint, test_fov_check, test_dd_sample, test_analytic_module
  time_testing.py -> --bench (fixed benchmark: see notes in benchmark())
"""

import os
import sys
import time
import argparse
from functools import lru_cache

import numpy as np

from vir import rd, pd, dd
from vir import analytic_sino as asino

SIZE = os.environ.get("VIR_TEST_SIZE", "small")


# ============================================================================
# Configuration
# ============================================================================
def cfg():
    if SIZE == "full":
        return dict(n2=64, nu2=96, na2=32, n3=32, nu3=48, nv3=48, na3=8)
    return dict(n2=32, nu2=48, na2=12, n3=16, nu3=24, nv3=24, na3=4)


def ucen(n, d, s_bins=0.0):
    return d*(np.arange(n) - (n - 1)/2 + s_bins)


# ----------------------------------------------------------------------------
# Uniform method registry: fp(img, ang, nu[, nv], **geom), bp(sino, ang, shape, **geom)
# ----------------------------------------------------------------------------
def _m2(fp, bp, **extra):
    return (lambda img, ang, nu, **k: fp(img, ang, nu, **k, **extra),
            lambda s, ang, sh, **k: bp(s, ang, sh, **k, **extra))


PAR2 = {"rd_aw": _m2(rd.aw_fp_par_2d, rd.aw_bp_par_2d),
        "rd_jo": _m2(rd.aw_fp_par_2d, rd.aw_bp_par_2d, joseph=True),
        "pd": _m2(pd.pd_fp_par_2d, pd.pd_bp_par_2d),
        "dd": _m2(dd.dd_fp_par_2d, dd.dd_bp_par_2d)}
FAN2 = {"rd_aw": _m2(rd.aw_fp_fan_2d, rd.aw_bp_fan_2d),
        "rd_jo": _m2(rd.aw_fp_fan_2d, rd.aw_bp_fan_2d, joseph=True),
        "pd": _m2(pd.pd_fp_fan_2d, pd.pd_bp_fan_2d),
        "dd": _m2(dd.dd_fp_fan_2d, dd.dd_bp_fan_2d)}
CONE = {"rd_aw": (rd.aw_fp_cone_3d, rd.aw_bp_cone_3d, {}),
        "rd_jo": (rd.aw_fp_cone_3d, rd.aw_bp_cone_3d, {"joseph": True}),
        "pd": (pd.pd_fp_cone_3d, pd.pd_bp_cone_3d, {}),
        "dd": (dd.dd_fp_cone_3d, dd.dd_bp_cone_3d, {})}
BIN_AVG = {"rd_aw": False, "rd_jo": False, "pd": True, "dd": True}


def cone_fp(m, img, ang, nu, nv, DSO, DSD, **k):
    f, _, extra = CONE[m]
    return f(img, ang, nu, nv, DSO, DSD, **k, **extra)


def cone_bp(m, sino, ang, shape, DSO, DSD, **k):
    _, b, extra = CONE[m]
    return b(sino, ang, shape, DSO, DSD, **k, **extra)


class Checks:
    """Collect (name, value, tolerance) rows; fail at the end with every
    violated check listed, so one bad number does not hide the others."""

    def __init__(self):
        self.rows = []

    def __call__(self, name, val, tol):
        val = float(val)
        self.rows.append((name, val, tol))
        return val

    def done(self):
        bad = [f"{n}: {v:.3g} >= {t:.3g}" for n, v, t in self.rows if not v < t]
        if bad:
            raise AssertionError("; ".join(bad), self.rows)
        return self.rows


def rel_l2(a, b):
    return float(np.linalg.norm(np.asarray(a, float) - b)/np.linalg.norm(b))


def centroid(s, axis_vals, axis):
    """Weighted mean coordinate of s along `axis` for every other index."""
    s = np.asarray(s, float)
    shape = [1]*s.ndim
    shape[axis] = -1
    w = s.sum(axis)
    return (s*axis_vals.reshape(shape)).sum(axis)/np.where(w > 0, w, np.nan)


# ============================================================================
# Shared geometry / phantoms (cached so pytest and CLI reuse projections)
# ============================================================================
G2 = dict(d_pix=0.8, su=0.25)
CIRCLE = (4.0, -2.4, 6.0)                    # off-centre: orientation sensitive
G3 = dict(d_pix=1.0, su=0.3, sv=-0.4)
SPHERE = (2.5, -1.5, 1.0, 4.0)


@lru_cache(None)
def geom2():
    c = cfg()
    n, nu, na, d = c["n2"], c["nu2"], c["na2"], G2["d_pix"]
    ang = np.linspace(0, 2*np.pi, na, endpoint=False)
    du = d*0.9
    DSO = 1.6*n*d                          # > image half-diagonal
    DSD = 2.0*DSO
    duf = du*DSD/DSO
    img = asino.discretized_circle_exact(CIRCLE, n, n, d_pix=d)
    return dict(n=n, nu=nu, ang=ang, du=du, duf=duf, DSO=DSO, DSD=DSD, img=img)


@lru_cache(None)
def sino2(geo, m):
    g = geom2()
    if geo == "par":
        return PAR2[m][0](g["img"], g["ang"], g["nu"], du=g["du"], **G2)
    return FAN2[m][0](g["img"], g["ang"], g["nu"], DSO=g["DSO"], DSD=g["DSD"],
                      du=g["duf"], **G2)


@lru_cache(None)
def ref2(geo, avg):
    g = geom2()
    n_sub = 8 if avg else 1
    if geo == "par":
        u = ucen(g["nu"], g["du"], G2["su"])
        return asino.analytic_sino_circle_par(CIRCLE, g["ang"], u, du=g["du"], n_sub=n_sub)
    u = ucen(g["nu"], g["duf"], G2["su"])
    return asino.analytic_sino_circle_fan(CIRCLE, g["ang"], u, g["DSO"], g["DSD"],
                                          du=g["duf"], n_sub=n_sub)


@lru_cache(None)
def geom3():
    c = cfg()
    n, d = c["n3"], G3["d_pix"]
    ang = np.linspace(0, 2*np.pi, c["na3"], endpoint=False) + 0.3   # oblique views
    DSO = 1.0*n*d
    DSD = 1.8*DSO
    du = dv = d*DSD/DSO*0.9
    img = asino.discretized_sphere_exact(SPHERE, n, n, n, d_pix=d)
    return dict(n=n, nu=c["nu3"], nv=c["nv3"], ang=ang, du=du, dv=dv,
                DSO=DSO, DSD=DSD, img=img)


@lru_cache(None)
def sino3(m):
    g = geom3()
    return cone_fp(m, g["img"], g["ang"], g["nu"], g["nv"], g["DSO"], g["DSD"],
                   du=g["du"], dv=g["dv"], **G3)


@lru_cache(None)
def ref3(avg):
    g = geom3()
    u = ucen(g["nu"], g["du"], G3["su"])
    v = ucen(g["nv"], g["dv"], G3["sv"])
    return asino.analytic_sino_sphere_cone(SPHERE, g["ang"], u, v, g["DSO"], g["DSD"],
                                           du=g["du"], dv=g["dv"], n_sub=6 if avg else 1)


# ============================================================================
# Tests
# ============================================================================
METHODS = ("rd_aw", "rd_jo", "pd", "dd")
# Smooth-phantom tolerances (rel L2).  Point-sampled methods are limited by
# the voxelization of the phantom, bin-averaged ones much less so.
TOL_SMOOTH_2D = {"rd_aw": 0.07, "rd_jo": 0.05, "pd": 0.03, "dd": 0.03}
TOL_SMOOTH_3D = {"rd_aw": 0.16, "rd_jo": 0.16, "pd": 0.10, "dd": 0.10}
# Ray-driven bp is aliased (number of rays hitting a pixel varies)
TOL_BP_NORM = {"rd_aw": 0.10, "rd_jo": 0.05, "pd": 0.02, "dd": 0.02}
TOL_CENTROID = {"rd_aw": 0.15, "rd_jo": 0.12, "pd": 0.05, "dd": 0.05}


def test_analytic_2d():
    """Accuracy and orientation vs analytic circle (par + fan, 0..2pi)."""
    g = geom2()
    chk = Checks()
    for geo in ("par", "fan"):
        du = g["du"] if geo == "par" else g["duf"]
        u = ucen(g["nu"], du, G2["su"])
        for m in METHODS:
            s, ref = sino2(geo, m), ref2(geo, BIN_AVG[m])
            err = rel_l2(s, ref)
            dc = np.nanmax(np.abs(centroid(s, u, 1) - centroid(ref, u, 1)))/du
            # Point-sampled rd sees the voxel staircase of the phantom (error
            # halves per refinement, see test_phantom_convergence); the strict
            # rd check is test_box_exact.
            chk(f"{geo} {m} rel L2", err, TOL_SMOOTH_2D[m])
            chk(f"{geo} {m} centroid [bins]", dc, TOL_CENTROID[m])
    return chk.done()


def test_analytic_cone():
    """Accuracy and orientation (u and v centroids) vs analytic sphere."""
    g = geom3()
    u = ucen(g["nu"], g["du"], G3["su"])
    v = ucen(g["nv"], g["dv"], G3["sv"])
    chk = Checks()
    for m in METHODS:
        s, ref = sino3(m), ref3(BIN_AVG[m])
        err = rel_l2(s, ref)
        su_ = np.nanmax(np.abs(centroid(s.sum(2), u, 1) - centroid(ref.sum(2), u, 1)))/g["du"]
        sv_ = np.nanmax(np.abs(centroid(s.sum(1), v, 1) - centroid(ref.sum(1), v, 1)))/g["dv"]
        chk(f"cone {m} rel L2", err, TOL_SMOOTH_3D[m])
        chk(f"cone {m} centroid [bins]", max(su_, sv_), TOL_CENTROID[m])
    return chk.done()


def test_bp_localisation():
    """bp of the analytic sinogram peaks at the phantom (orientation of bp)."""
    g = geom2()
    chk = Checks()
    x = (np.arange(g["n"]) - (g["n"] - 1)/2)*G2["d_pix"]
    for geo, reg in (("par", PAR2), ("fan", FAN2)):
        for m in METHODS:
            kw = dict(du=g["du"], **G2) if geo == "par" else \
                dict(du=g["duf"], DSO=g["DSO"], DSD=g["DSD"], **G2)
            b = reg[m][1](ref2(geo, BIN_AVG[m]), g["ang"], (g["n"], g["n"]), **kw)
            b = np.clip(b - np.percentile(b, 90), 0, None)          # keep the peak
            cx = (b.sum(1)*x).sum()/b.sum()
            cy = (b.sum(0)*x).sum()/b.sum()
            dist = np.hypot(cx - CIRCLE[0], cy - CIRCLE[1])/G2["d_pix"]
            chk(f"bp {geo} {m} peak offset [px]", dist, 0.5)
    return chk.done()


def test_adjoint():
    """<Ax, y> == <x, A^T y> with norm=False, every projector."""
    rng = np.random.default_rng(0)
    ang = np.linspace(0, 2*np.pi, 5, endpoint=False) + 0.1
    sh2, nu, d = (9, 7), 14, 0.7
    DSO, DSD = 20.0, 35.0
    chk = Checks()

    def check(name, fp, bp, xs, ys):
        x = rng.random(xs).astype(np.float32)
        y = rng.random(ys).astype(np.float32)
        r = float(np.vdot(fp(x), y))/float(np.vdot(x, bp(y)))
        chk(f"adjoint {name} |ratio-1|", abs(r - 1), 1e-4)

    for m in METHODS:
        f, b = PAR2[m]
        check(f"par {m}", lambda x: f(x, ang, nu, su=0.3, d_pix=d),
              lambda y: b(y, ang, sh2, su=0.3, d_pix=d, norm=False), sh2, (5, nu))
        f, b = FAN2[m]
        check(f"fan {m}", lambda x: f(x, ang, nu, DSO=DSO, DSD=DSD, su=0.3, d_pix=d),
              lambda y: b(y, ang, sh2, DSO=DSO, DSD=DSD, su=0.3, d_pix=d, norm=False),
              sh2, (5, nu))
        sh3 = (6, 5, 4)
        check(f"cone {m}",
              lambda x: cone_fp(m, x, ang, nu, 8, DSO, DSD, su=0.3, sv=-0.2, d_pix=d),
              lambda y: cone_bp(m, y, ang, sh3, DSO, DSD, su=0.3, sv=-0.2, d_pix=d, norm=False),
              sh3, (5, nu, 8))
    shs = (6, 6, 4)
    kw = dict(dsrc_p=1.5, ssrc_p=0.4, ssrc_z=0.3, su=0.2, d_pix=d)
    check("square rd", lambda x: rd.aw_fp_square(x, 9, 7, 3, 2, DSO, DSD, **kw),
          lambda y: rd.aw_bp_square(y, shs, DSO, DSD, norm=False, **kw), shs, (4, 3, 2, 9, 7))
    check("square dd", lambda x: dd.dd_fp_square(x, 9, 7, 3, 2, DSO, DSD, **kw),
          lambda y: dd.dd_bp_square(y, shs, DSO, DSD, norm=False, **kw), shs, (3, 2, 7, 9, 4))
    return chk.done()


def test_bp_normalisation():
    """Single view, band of ones: normalised bp == 1 inside the band
    (test_dd.py checked rec.max() ~ 1)."""
    n, nu = 24, 32
    ang = np.array([np.deg2rad(10.0)])
    sino = np.zeros((1, nu), np.float32)
    sino[0, 4:-4] = 1.0
    chk = Checks()
    for m in METHODS:
        b = PAR2[m][1](sino, ang, (n, n))
        core = b[n//2 - 3:n//2 + 3, n//2 - 3:n//2 + 3]
        dev = float(np.abs(core - 1).max())
        chk(f"bp norm par {m} |bp-1|", dev, TOL_BP_NORM[m])
    return chk.done()


def test_fan_parallel_limit():
    """Near-parallel fan (DSO=1e6) equals parallel, fp and bp (test_dd.py)."""
    n, nu = 24, 32
    ang = np.linspace(0, 2*np.pi, 8, endpoint=False)
    img = asino.discretized_circle_exact((3.0, -2.0, 5.0), n, n)
    DSO, DSD = 1e6, 1e6 + 1.0
    chk = Checks()
    for m in METHODS:
        sp = PAR2[m][0](img, ang, nu)
        sf = FAN2[m][0](img, ang, nu, DSO=DSO, DSD=DSD, du=DSD/DSO)
        # Fan view at theta looks from the opposite side: u axis is shared
        e = rel_l2(sf, sp)
        bpp = PAR2[m][1](sp, ang, (n, n))
        bpf = FAN2[m][1](sp, ang, (n, n), DSO=DSO, DSD=DSD, du=DSD/DSO)
        eb = rel_l2(bpf, bpp)
        chk(f"fan->par fp {m}", e, 1e-3)
        chk(f"fan->par bp {m}", eb, 1e-3)
    return chk.done()


def test_impulse_response():
    """Single off-centre voxel (test_fp_imp.py).  Each view is projected on
    a small, fine detector (du = 0.1) centred on the projected voxel via
    su/sv, so point-sampled methods are not aliased.  Every footprint must
    integrate to the exact d^3 M^2 / cos(psi) and be centred at (M p, M z)."""
    n = 13
    d = 1.0
    ix, iy, iz = 3, 7, 10
    img = np.zeros((n, n, n), np.float32)
    img[ix, iy, iz] = 1.0
    DSO, DSD = 14.0, 28.0
    du = dv = 0.1
    nu = nv = 48
    xc, yc, zc = [(i - (n - 1)/2)*d for i in (ix, iy, iz)]
    chk = Checks()
    for m in METHODS:
        err_c, err_m = 0.0, 0.0
        for th in np.linspace(0, np.pi/2, 4):
            p = -np.sin(th)*xc + np.cos(th)*yc
            o = np.cos(th)*xc + np.sin(th)*yc
            M = DSD/(DSO - o)
            su, sv = M*p/du, M*zc/dv                 # centre detector on voxel
            s = cone_fp(m, img, np.array([th]), nu, nv, DSO, DSD, du=du, dv=dv,
                        su=su, sv=sv, d_pix=d)[0]
            u = ucen(nu, du, su)
            v = ucen(nv, dv, sv)
            cu = (s.sum(1)*u).sum()/s.sum()
            cv = (s.sum(0)*v).sum()/s.sum()
            mass_ref = d**3*M**2*np.sqrt(DSD**2 + (M*p)**2 + (M*zc)**2)/DSD
            err_c = max(err_c, abs(cu - M*p)/du, abs(cv - M*zc)/dv)
            err_m = max(err_m, abs(s.sum()*du*dv/mass_ref - 1))
        # Perspective shifts the footprint centroid slightly off M*p (near half
        # of the voxel is magnified more): allow 0.05 world units = 0.5 bins
        chk(f"impulse {m} centroid [0.1-unit bins]", err_c, 0.5)
        chk(f"impulse {m} mass vs d^3 M^2/cos(psi)", err_m, 0.02 if m.startswith("rd") else 0.005)
    return chk.done()


def test_slab_plateau():
    """Slab/cube plateau == path length (test_scaling.py, test_fp_3d.py)."""
    chk = Checks()
    n, T = 40, 10
    img = np.zeros((n, n), np.float32)
    img[n//2 - T//2:n//2 + T//2, 6:-6] = 1.0                 # thin in x, long in y
    ang = np.array([0.0, np.pi/4])
    nu = 64
    du = 0.5
    expect = [T, T*np.sqrt(2)]
    for geo, reg, kw in (("par", PAR2, {}),
                         ("fan", FAN2, dict(DSO=60.0, DSD=120.0))):
        for m in METHODS:
            s = reg[m][0](img, ang, nu, du=du, **kw)
            for ia in range(2):
                c = s[ia, nu//2 - 2:nu//2 + 2].mean()
                e = abs(c/expect[ia] - 1)
                chk(f"plateau {geo} {m} {int(np.degrees(ang[ia]))}deg", e,
                    0.06 if m == "rd_jo" else 0.02)            # Joseph edge blur
    # Cone: centre ray through a centred cube == side length
    n3, S = 16, 8
    cube = np.zeros((n3, n3, n3), np.float32)
    cube[4:12, 4:12, 4:12] = 1.0
    for m in METHODS:
        s = cone_fp(m, cube, np.array([0.0]), 15, 15, 20.0, 40.0, du=0.5, dv=0.5)
        e = abs(s[0, 7, 7]/S - 1)
        chk(f"plateau cone {m}", e, 0.05 if m == "rd_jo" else 0.01)
    return chk.done()


def test_dpix_scaling():
    """Same object in voxel units with every length scaled by k: fp scales by
    k, normalised bp is unchanged."""
    k = 0.5
    n, nu = 20, 30
    ang = np.linspace(0, np.pi, 6, endpoint=False) + 0.2
    img = asino.discretized_circle_exact((2.0, 1.0, 5.0), n, n)
    chk = Checks()
    for geo, reg in (("par", PAR2), ("fan", FAN2)):
        g1 = dict(d_pix=1.0, du=1.0) | ({} if geo == "par" else dict(DSO=40.0, DSD=70.0))
        gk = {a: b*k for a, b in g1.items()}
        for m in METHODS:
            s1 = reg[m][0](img, ang, nu, **g1)
            sk = reg[m][0](img, ang, nu, **gk)
            b1 = reg[m][1](s1, ang, (n, n), **g1)
            bk = reg[m][1](sk/k, ang, (n, n), **gk)
            e = max(rel_l2(sk, k*s1), rel_l2(bk, b1))
            chk(f"d_pix scaling {geo} {m}", e, 1e-4)
    return chk.done()


def test_gridline_rays():
    """Rays exactly on pixel boundaries (test_fp_siddons.py)."""
    n = 4
    img = np.ones((n, n), np.float32)
    ang = np.array([0.0, np.pi/2, np.pi])
    chk = Checks()
    # du=1, su=0.5 -> bin centres at integers: interior rays lie on grid lines
    for geo, reg, kw in (("par", PAR2, {}),
                         ("fan~par", FAN2, dict(DSO=1e6, DSD=1e6 + 2.0))):
        for m in ("rd_aw",):                       # ray-driven sampling only
            s = reg[m][0](img, ang, 8, su=0.5, **kw)
            u = ucen(8, 1.0, 0.5)
            inner = s[:, np.abs(u) < n/2 - 1e-9]
            outer = s[:, np.abs(u) > n/2 + 1e-9]
            e = max(np.abs(inner - n).max(), np.abs(outer).max())
            chk(f"gridline {geo} {m}", e, 1e-4)
            # Edge rays (|u| == n/2) lie on the image boundary: any value in
            # [0, n] is legitimate (which side wins is a rounding decision)
            edge = s[:, np.isclose(np.abs(u), n/2)]
            chk(f"gridline {geo} {m} edge outside [0,n]",
                np.clip(-edge, 0, None).max() + np.clip(edge - n, 0, None).max(), 1e-4)
    # Footprint methods: bins straddling the edge integrate half a bin
    for m in ("pd", "dd"):
        s = PAR2[m][0](img, ang, 8, su=0.5)
        e = abs(s.sum(1) - n*n).max()
        chk(f"gridline mass par {m}", e, 1e-4)
    return chk.done()


def _sq_geom():
    n, nz = 20, 12
    d = 0.6
    sphere = (1.5, -2.0, 0.8, 3.0)
    img = asino.discretized_sphere_exact(sphere, n, n, nz, d_pix=d)
    kw = dict(du=0.8, dv=0.8, dsrc_p=1.5, dsrc_z=1.0, su=0.25, sv=-0.3,
              ssrc_p=1.0, ssrc_z=0.5, d_pix=d)
    return n, nz, sphere, img, kw, 20.0, 34.0, 18, 14, 5, 3


def test_square_vs_analytic():
    """rd/dd square vs analytic square sinogram, all sides, with source
    shifts (build_sinos.py, m_testing.py)."""
    n, nz, sphere, img, kw, DSO, DSD, nu, nv, nsp, nsz = _sq_geom()
    ref_c = asino.analytic_sino_sphere_square(sphere, nu, nv, nsp, nsz, DSO, DSD,
                                              **{k: v for k, v in kw.items() if k != "d_pix"})
    ref_a = asino.analytic_sino_sphere_square(sphere, nu, nv, nsp, nsz, DSO, DSD, n_sub=6,
                                              **{k: v for k, v in kw.items() if k != "d_pix"})
    s_rd = rd.aw_fp_square(img, nu, nv, nsp, nsz, DSO, DSD, **kw).transpose(1, 2, 4, 3, 0)
    s_dd = dd.dd_fp_square(img, nu, nv, nsp, nsz, DSO, DSD, **kw)
    chk = Checks()
    for k in range(4):
        e_rd = rel_l2(s_rd[..., k], ref_c[..., k])
        e_dd = rel_l2(s_dd[..., k], ref_a[..., k])
        chk(f"square side{k} rd rel L2", e_rd, 0.12)
        chk(f"square side{k} dd rel L2", e_dd, 0.08)
    return chk.done()


def test_square_side_localisation():
    """Per-side bp of the analytic square sinogram localises the sphere in
    the directions that side resolves (test_squre.py)."""
    n, nz, sphere, img, kw, DSO, DSD, nu, nv, nsp, nsz = _sq_geom()
    kw = dict(kw, dsrc_p=0.5, ssrc_p=0.0)
    nsp = 25
    ref = asino.analytic_sino_sphere_square(sphere, nu, nv, nsp, nsz, DSO, DSD, n_sub=4,
                                            **{k: v for k, v in kw.items() if k != "d_pix"})
    x = (np.arange(n) - (n - 1)/2)*kw["d_pix"]
    z = (np.arange(nz) - (nz - 1)/2)*kw["d_pix"]
    # side 0/2 rays along x -> resolve y and z; side 1/3 along y -> resolve x, z
    resolved = {0: (1,), 1: (0,), 2: (1,), 3: (0,)}
    chk = Checks()
    for k in range(4):
        sk = np.zeros_like(ref)
        sk[..., k] = ref[..., k]
        b = dd.dd_bp_square(sk.astype(np.float32), (n, n, nz), DSO, DSD, **kw)
        b = np.clip(b - np.percentile(b, 85), 0, None)
        for ax in resolved[k]:
            c = (b.sum(axis=tuple(i for i in range(3) if i != ax))*x).sum()/b.sum()
            e = abs(c - sphere[ax])
            chk(f"square bp side{k} {'xyz'[ax]}-centroid error", e, 0.6)
        cz = (b.sum((0, 1))*z).sum()/b.sum()
        chk(f"square bp side{k} z-centroid error", abs(cz - sphere[2]), 0.6)
    return chk.done()


def test_square_su_sweep():
    """fp and bp with the same detector shift su give the same localisation
    (test_squre.py su sweep)."""
    n, nz, sphere, img, kw, DSO, DSD, nu, nv, nsp, nsz = _sq_geom()
    x = (np.arange(n) - (n - 1)/2)*kw["d_pix"]
    cents = []
    for su in (0.0, 0.25, 0.5, 0.75, 1.0):
        k2 = dict(kw, su=su)
        s = dd.dd_fp_square(img, nu, nv, nsp, nsz, DSO, DSD, **k2)
        b = dd.dd_bp_square(s, (n, n, nz), DSO, DSD, **k2)
        b = np.clip(b - np.percentile(b, 85), 0, None)
        cents.append([(b.sum((1, 2))*x).sum()/b.sum(), (b.sum((0, 2))*x).sum()/b.sum()])
    spread = float(np.ptp(np.array(cents), axis=0).max())/kw["d_pix"]
    chk = Checks()
    chk("square su sweep centroid spread [px]", spread, 0.3)
    return chk.done()


def test_fov_check():
    """Divergent beams with the image corner at/behind the source are
    rejected (DSO = half-diagonal, as in test_fp_3d/test_scaling, is
    degenerate: the corner projects to infinity)."""
    n = 16
    DSO = n*np.sqrt(2)/2
    img = np.zeros((n, n, n), np.float32)
    for f in (lambda: dd.dd_fp_cone_3d(img, np.zeros(1), 8, 8, DSO, 2*DSO),
              lambda: pd.pd_fp_cone_3d(img, np.zeros(1), 8, 8, DSO, 2*DSO),
              lambda: rd.aw_fp_cone_3d(img, np.zeros(1), 8, 8, DSO, 2*DSO)):
        try:
            f()
        except ValueError:
            continue
        raise AssertionError("degenerate geometry not rejected")
    return [("degenerate geometries rejected", 0.0, 1.0)]


def test_dd_sample():
    """dd_fp_cone_3d(sample=2) on a smooth phantom ~ sample=1."""
    g = geom3()
    s1 = sino3("dd")
    s2 = dd.dd_fp_cone_3d(g["img"], g["ang"], g["nu"], g["nv"], g["DSO"], g["DSD"],
                          du=g["du"], dv=g["dv"], sample=2, **G3)
    e = rel_l2(s2, s1)
    chk = Checks()
    chk("dd sample=2 vs 1", e, 0.02)
    return chk.done()


def test_analytic_module():
    """Internal consistency of analytic_sino and agreement with rd AW, which
    is exact for voxel-aligned boxes."""
    chk = Checks()
    ang = np.linspace(0, 2*np.pi, 6, endpoint=False) + 0.2
    u = ucen(30, 0.6)
    v = ucen(12, 0.6)
    DSO, DSD = 30.0, 50.0
    c3 = (3.0, -2.0, 0.0, 4.0)
    e = np.abs(asino.analytic_sino_circle_fan(c3[:2] + c3[3:], ang, u, DSO, DSD)
               - asino.analytic_sino_sphere_cone(c3, ang, u, np.zeros(1), DSO, DSD)[..., 0]).max()
    chk("analytic fan == cone mid-plane", e, 1e-9)
    e = np.abs(asino.analytic_sino_circle_par((3.0, -2.0, 4.0), ang, u)
               - asino.analytic_sino_circle_fan((3.0, -2.0, 4.0), ang, u, 1e9, 1e9)).max()
    chk("analytic par == fan(DSO->inf)", e, 1e-3)
    # Polyhedron line integrals vs rd AW on a voxel-aligned box (both exact)
    box = np.zeros((10, 10, 10), np.float32)
    box[2:7, 3:9, 4:8] = 1.0
    planes = asino.cube_planes((-0.5, 1.0, 1.0), (5, 6, 4))
    pa = asino.analytic_sino_polyhedron_cone(planes, ang, u, v, DSO, DSD)
    pr = rd.aw_fp_cone_3d(box, ang, 30, 12, DSO, DSD, du=0.6, dv=0.6)
    e = np.abs(pa - pr).max()
    chk("polyhedron cone vs rd AW (exact)", e, 1e-3)
    # Mass checks: frustum volumes sum to the object volume at every view
    src_, det_, uh_, vh_ = None, None, None, None
    ub = np.linspace(-15, 15, 16)
    for t in (0.0, 0.7, np.pi/2):
        src_ = np.array([DSO*np.cos(t), DSO*np.sin(t), 0])
        det_ = -(DSD - DSO)*np.array([np.cos(t), np.sin(t), 0])
        uh_ = np.array([-np.sin(t), np.cos(t), 0])
        vh_ = np.array([0, 0, 1.0])
        vs = asino.analytic_sino_sphere_frustum_vol(src_, det_, ub, ub, uh_, vh_, (1.0, -2.0, 0.5, 4.0))
        vp = asino.analytic_sino_polyhedron_gen_vol(src_, det_, ub, ub, uh_, vh_, planes)
        es = abs(vs.sum()/(4/3*np.pi*64) - 1)
        ep = abs(vp.sum()/120.0 - 1)
        chk(f"sphere frustum vol sum t={t:.1f}", es, 1e-3)
        chk(f"box frustum vol sum t={t:.1f}", ep, 1e-6)
    vol = asino.discretized_sphere_exact((0.3, -0.2, 0.1, 4.0), 12, 12, 12).sum()
    e = abs(vol/(4/3*np.pi*64) - 1)
    chk("discretized sphere volume", e, 1e-3)
    return chk.done()


# ----------------------------------------------------------------------------
# Discretization-free tests: voxel-aligned boxes have exact line integrals,
# and rd AW is exact for them (machine precision).
# ----------------------------------------------------------------------------
TOL_BOX = {"rd_aw": 1e-4, "rd_jo": 0.12, "pd": 0.03, "dd": 0.03}


def _box_planes(lo, hi, n, d, nz=None):
    """Planes of voxel range [lo, hi) per axis (image centred on origin)."""
    lo = np.asarray(lo, float)
    hi = np.asarray(hi, float)
    nn = np.array([n, n, nz if nz else n], float)
    a = (lo - nn/2)*d
    b = (hi - nn/2)*d
    return asino.cube_planes((a + b)/2, b - a)


def test_box_exact():
    """Voxel-aligned boxes vs exact analytic line integrals: par, fan, cone,
    square.  rd AW must match to ~1e-5; others to their model accuracy
    (pd/dd vs bin averages, rd_jo vs centre samples)."""
    chk = Checks()
    d = 0.7
    ang = np.linspace(0, 2*np.pi, 7, endpoint=False) + 0.15
    # 2D: box in the image plane = 3D box with large z extent
    n, nu = 16, 30
    img = np.zeros((n, n), np.float32)
    img[3:10, 5:14] = 1.0
    planes = _box_planes((3, 5, -50), (10, 14, 50), n, d, nz=1)
    du, su = 0.6, 0.3
    u = ucen(nu, du, su)
    DSO, DSD = 25.0, 45.0
    duf = du*DSD/DSO
    uf = ucen(nu, duf, su)
    refs = {("par", False): asino.analytic_sino_polyhedron_par(planes, ang, u)[..., 0],
            ("par", True): asino.analytic_sino_polyhedron_par(planes, ang, u, du=du, n_sub=16)[..., 0],
            ("fan", False): asino.analytic_sino_polyhedron_cone(planes, ang, uf, [0.0], DSO, DSD)[..., 0],
            ("fan", True): asino.analytic_sino_polyhedron_cone(planes, ang, uf, [0.0], DSO, DSD,
                                                              du=duf, n_sub=16)[..., 0]}
    for m in METHODS:
        sp = PAR2[m][0](img, ang, nu, du=du, su=su, d_pix=d)
        sf = FAN2[m][0](img, ang, nu, DSO=DSO, DSD=DSD, du=duf, su=su, d_pix=d)
        for geo, s in (("par", sp), ("fan", sf)):
            e = rel_l2(s, refs[(geo, BIN_AVG[m])])
            chk(f"box {geo} {m}", e, TOL_BOX[m])
    # Cone
    n3, nv = 12, 16
    vol = np.zeros((n3, n3, n3), np.float32)
    vol[2:8, 4:11, 3:9] = 1.0
    planes = _box_planes((2, 4, 3), (8, 11, 9), n3, d)
    DSO, DSD = 20.0, 36.0
    du = dv = 0.75
    sv = -0.2
    u = ucen(nu, du, su)
    v = ucen(nv, dv, sv)
    for m in METHODS:
        s = cone_fp(m, vol, ang[:4], nu, nv, DSO, DSD, du=du, dv=dv, su=su, sv=sv, d_pix=d)
        ref = asino.analytic_sino_polyhedron_cone(planes, ang[:4], u, v, DSO, DSD, du=du, dv=dv,
                                                  n_sub=8 if BIN_AVG[m] else 1)
        e = rel_l2(s, ref)
        chk(f"box cone {m}", e, TOL_BOX[m])
    # Square (with source shifts)
    kw = dict(du=0.8, dv=0.8, dsrc_p=1.5, dsrc_z=1.0, su=0.25, sv=-0.3,
              ssrc_p=1.0, ssrc_z=0.5)
    vol = np.zeros((n3, n3, 8), np.float32)
    vol[2:8, 4:11, 1:6] = 1.0
    planes = _box_planes((2, 4, 1), (8, 11, 6), n3, d, nz=8)
    DSO, DSD = 15.0, 26.0
    args = (18, 14, 5, 3, DSO, DSD)
    s_rd = rd.aw_fp_square(vol, *args, d_pix=d, **kw).transpose(1, 2, 4, 3, 0)
    s_dd = dd.dd_fp_square(vol, *args, d_pix=d, **kw)
    r_c = asino.analytic_sino_polyhedron_square(planes, *args, **kw)
    r_a = asino.analytic_sino_polyhedron_square(planes, *args, n_sub=8, **kw)
    for k in range(4):
        e_rd = rel_l2(s_rd[..., k], r_c[..., k])
        e_dd = rel_l2(s_dd[..., k], r_a[..., k])
        chk(f"box square side{k} rd_aw", e_rd, TOL_BOX["rd_aw"])
        chk(f"box square side{k} dd", e_dd, TOL_BOX["dd"])
    return chk.done()


def test_phantom_convergence():
    """Point-sampled rd error vs a smooth circle is phantom discretization:
    it roughly halves when the phantom is refined 2x."""
    C = (4.0, -2.4, 6.0)
    ang = np.linspace(0, 2*np.pi, 8, endpoint=False)
    nu, du = 40, 0.72
    u = ucen(nu, du)
    ref = asino.analytic_sino_circle_par(C, ang, u)
    errs = []
    for k in (1, 2):
        n, d = 32*k, 0.8/k                      # circle must fit inside the image
        img = asino.discretized_circle_exact(C, n, n, d_pix=d)
        errs.append(rel_l2(rd.aw_fp_par_2d(img, ang, nu, du=du, d_pix=d), ref))
    ratio = errs[1]/errs[0]
    chk = Checks()
    chk("rd AW rel L2, phantom x1", errs[0], 0.1)
    chk("rd AW rel L2, phantom x2", errs[1], 0.1)
    chk("error ratio x2/x1 (expect ~0.5)", ratio, 0.65)
    return chk.done()


ALL_TESTS = [test_analytic_module, test_box_exact, test_phantom_convergence, test_analytic_2d, test_analytic_cone,
             test_bp_localisation, test_adjoint, test_bp_normalisation,
             test_fan_parallel_limit, test_impulse_response, test_slab_plateau,
             test_dpix_scaling, test_gridline_rays, test_square_vs_analytic,
             test_square_side_localisation, test_square_su_sweep, test_fov_check,
             test_dd_sample]


# ============================================================================
# Benchmark (replaces time_testing.benchmark)
# ============================================================================
def benchmark(func, *args, n_runs=10, warmup=2, n_rep=5, reset=None, label=None, **kwargs):
    """Median wall time over n_rep repetitions of n_runs calls.

    Fixes vs time_testing.benchmark: `times` was never reset so each
    "median" included all previous repetitions; and p_time passed cone bp
    arguments positionally (..., du, dv, d_pix), which put d_pix into su.
    Always pass geometry by keyword.
    """
    for _ in range(warmup):
        if reset:
            reset()
        func(*args, **kwargs)
    medians = []
    for _ in range(n_rep):
        times = []
        for _ in range(n_runs):
            if reset:
                reset()
            t0 = time.perf_counter()
            func(*args, **kwargs)
            times.append(time.perf_counter() - t0)
        medians.append(np.median(times))
    name = label or getattr(func, "__name__", "func")
    print(f"{name:28s} " + ", ".join(f"{m*1e3:9.3f}" for m in medians) + "  ms")
    return np.median(medians)


def run_benchmarks(n_runs=3):
    g2, g3 = geom2(), geom3()
    print("\n=== Benchmarks (median ms per call) ===")
    for geo, reg in (("par", PAR2), ("fan", FAN2)):
        kw = dict(du=g2["du"], **G2) if geo == "par" else \
            dict(du=g2["duf"], DSO=g2["DSO"], DSD=g2["DSD"], **G2)
        for m in METHODS:
            benchmark(reg[m][0], g2["img"], g2["ang"], g2["nu"], n_runs=n_runs,
                      label=f"fp {geo} {m}", **kw)
            s = sino2(geo, m)
            benchmark(reg[m][1], s, g2["ang"], g2["img"].shape, n_runs=n_runs,
                      label=f"bp {geo} {m}", **kw)
    for m in METHODS:
        kw = dict(du=g3["du"], dv=g3["dv"], **G3)
        benchmark(cone_fp, m, g3["img"], g3["ang"], g3["nu"], g3["nv"], g3["DSO"],
                  g3["DSD"], n_runs=n_runs, label=f"fp cone {m}", **kw)
        benchmark(cone_bp, m, sino3(m), g3["ang"], g3["img"].shape, g3["DSO"],
                  g3["DSD"], n_runs=n_runs, label=f"bp cone {m}", **kw)


# ============================================================================
# CLI
# ============================================================================
def main():
    global SIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default=SIZE, choices=("small", "full"))
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("-k", default="", help="run tests whose name contains this")
    a = ap.parse_args()
    SIZE = a.size
    n_fail = 0
    for t in ALL_TESTS:
        if a.k not in t.__name__:
            continue
        t0 = time.time()
        try:
            rows = t()
            status = "PASS"
        except AssertionError as e:
            rows = e.args[1] if len(e.args) > 1 else []
            status = "FAIL" + ("" if rows else f": {e}")
            n_fail += 1
        print(f"\n[{status}] {t.__name__}  ({time.time() - t0:.0f}s)")
        for name, val, tol in rows:
            flag = "" if val < tol else "   <-- FAIL"
            print(f"    {name:44s} {val:10.2e}  (tol {tol:.0e}){flag}")
    if a.bench:
        run_benchmarks()
    print(f"\n{n_fail} test(s) failed")
    return n_fail


if __name__ == "__main__":
    sys.exit(main())
