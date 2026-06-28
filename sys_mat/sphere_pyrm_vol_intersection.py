#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 07:24:30 2026

@author: pvargas21
"""

def analytic_sino_sphere2(
        src,
        det,
        u_bnd,
        v_bnd,
        u_hat,
        v_hat,
        center,
        radius,
        rho,
        dirs,
        weights):

    Nu = len(u_bnd) - 1
    Nv = len(v_bnd) - 1

    uv = detector_grid(
        det,
        u_bnd,
        v_bnd,
        u_hat=u_hat,
        v_hat=v_hat
    )

    u_coord_bot = uv[:, 0, :]
    u_coord_top = uv[:, -1, :]

    v_coord_lft = uv[0, :, :]
    v_coord_rgt = uv[-1, :, :]

    ref_u = src + np.array([0.0, 1.0, 0.0])
    ref_v = src + np.array([0.0, 0.0, 1.0])

    planes_u = plane_from_pts(
        src,
        u_coord_bot,
        u_coord_top,
        ref_u
    )
    
    n = planes_u[:, :3]
    print("ere")
    for i in range(10):
        print(i, np.dot(n[i], n[i+1]))

    planes_v = plane_from_pts(
        src,
        v_coord_lft,
        v_coord_rgt,
        ref_v
    )

    planes_uv = plane_from_pts(
        u_coord_bot[0],
        u_coord_top[0],
        u_coord_bot[-1],
        src
    )

    proj = np.zeros((Nu, Nv), dtype=float)

    #
    # ---------------------------------------------------------
    # PRECOMPUTE ALL DIR·NORMAL PRODUCTS
    # ---------------------------------------------------------
    #

    Au = dirs @ planes_u[:, :3].T          # (Nq, Nu+1)

    Av = dirs @ planes_v[:, :3].T          # (Nq, Nv+1)

    Auv = dirs @ planes_uv[:3]             # (Nq,)

    #
    # ---------------------------------------------------------
    # PRECOMPUTE ALL b = d - n·center TERMS
    # ---------------------------------------------------------
    #

    bu = planes_u[:, 3] - planes_u[:, :3] @ center
    bv = planes_v[:, 3] - planes_v[:, :3] @ center
    buv = planes_uv[3] - planes_uv[:3] @ center

    #
    # Reusable work arrays
    #

    A = np.empty((len(weights), 5), dtype=float)
    b = np.empty(5, dtype=float)

    for i in range(Nu):

        #
        # u planes
        #

        A[:, 0] = Au[:, i]
        A[:, 1] = -Au[:, i + 1]

        b[0] = bu[i]
        b[1] = -bu[i + 1]

        for j in range(Nv):

            #
            # v planes
            #

            A[:, 2] = Av[:, j]
            A[:, 3] = -Av[:, j + 1]

            b[2] = bv[j]
            b[3] = -bv[j + 1]

            #
            # detector plane
            #

            A[:, 4] = Auv
            b[4] = buv

            #
            # quick reject:
            # sphere completely outside any half-space
            #

            reject = False

            for p in range(5):

                # signed distance to plane
                if -b[p] > radius:
                    reject = True
                    break

            if reject:
                continue

            vol = sphere_clip_volume_precomp(
                A,
                b,
                radius,
                weights
            )

            proj[i, j] = rho * vol

    return proj



def analytic_sino_sphere3(
        src,
        det,
        u_bnd,
        v_bnd,
        u_hat,
        v_hat,
        center,
        radius,
        rho,
        dirs,
        weights,
        R_lebedev=None):

    print("1")
    if R_lebedev is not None:
        dirs = dirs @ R_lebedev.T

    print("2")

    Nu = len(u_bnd) - 1
    Nv = len(v_bnd) - 1

    uv = detector_grid(
        det,
        u_bnd,
        v_bnd,
        u_hat=u_hat,
        v_hat=v_hat
    )

    u_coord_bot = uv[:, 0, :]
    u_coord_top = uv[:, -1, :]

    v_coord_lft = uv[0, :, :]
    v_coord_rgt = uv[-1, :, :]

    ref_u = src + np.array([0.0, 1.0, 0.0])
    ref_v = src + np.array([0.0, 0.0, 1.0])

    print("3")

    planes_u = plane_from_pts(
        src,
        u_coord_bot,
        u_coord_top,
        ref_u
    )

    planes_v = plane_from_pts(
        src,
        v_coord_lft,
        v_coord_rgt,
        ref_v
    )

    planes_uv = plane_from_pts(
        u_coord_bot[0],
        u_coord_top[0],
        u_coord_bot[-1],
        src
    )
    print("4")
    print("dirs shape =", dirs.shape)
    #print("R shape    =", np.shape(R))
    #print("dirs_rot shape =", np.shape(dirs_rot))

    #print(type(rotations))
    #print(np.shape(rotations))

    proj = np.zeros((Nu, Nv), dtype=float)

    #
    # Precompute rotated-dir dot products
    #

    print(dirs.shape)
    print(planes_u[:, :3].shape)
    print(planes_uv[:3].shape)

    Au = dirs @ planes_u[:, :3].T
    Av = dirs @ planes_v[:, :3].T
    Auv = dirs @ planes_uv[:3]
    print("5")

    bu = planes_u[:, 3] - planes_u[:, :3] @ center
    bv = planes_v[:, 3] - planes_v[:, :3] @ center
    buv = planes_uv[3] - planes_uv[:3] @ center

    A = np.empty((len(weights), 5), dtype=float)
    b = np.empty(5, dtype=float)

    print("6")

    for i in range(Nu):

        A[:, 0] = Au[:, i]
        A[:, 1] = -Au[:, i + 1]

        b[0] = bu[i]
        b[1] = -bu[i + 1]

        for j in range(Nv):

            A[:, 2] = Av[:, j]
            A[:, 3] = -Av[:, j + 1]

            b[2] = bv[j]
            b[3] = -bv[j + 1]

            A[:, 4] = Auv
            b[4] = buv

            vol = sphere_clip_volume_precomp(
                A,
                b,
                radius,
                weights
            )

            proj[i, j] = rho * vol

    return proj


def analytic_sino_sphere2_avg(
        *args,
        dirs,
        weights,
        rotations):

    print("Hi")
    proj = None

    for R in rotations:
        print("dirs:", dirs.shape)
        print("R:", np.shape(R))
        
        dirs_rot = dirs @ R.T
        
        print("dirs_rot:", np.shape(dirs_rot))
        
        print("Here")
        p = analytic_sino_sphere3(
            *args,
            dirs=dirs,
            weights=weights,
            R_lebedev=R
        )

        if proj is None:
            proj = p
        else:
            proj += p

    return proj / len(rotations)




def sphere_clip_volume_precomp(A, b, radius, weights):
    """
    Parameters
    ----------
    A : (Nq,M)
        A[k,j] = dirs[k] · normals[j]

    b : (M,)
        b[j] = d_j - normals[j] · center

    radius : float

    weights : (Nq,)

    Returns
    -------
    volume : float
    """

    volume = 0.0

    Nq, M = A.shape

    for k in range(Nq):

        rmin = 0.0
        rmax = radius
        valid = True

        for j in range(M):

            a = A[k, j]
            bj = b[j]

            if abs(a) < EPS:

                if bj < 0.0:
                    valid = False
                    break

            elif a > 0.0:

                t = bj / a

                if t < rmax:
                    rmax = t

            else:

                t = bj / a

                if t > rmin:
                    rmin = t

            if rmax <= rmin:
                valid = False
                break

        if valid:
            volume += weights[k] * (rmax**3 - rmin**3)

    return volume / 3.0





dirs, weights = lebedev_rule(131)

u_hat_rot = np.array([
    0.0,
    0.956304756,
    0.292371705
])

v_hat_rot = np.array([
    0.0,
   -0.292371705,
    0.956304756
])




def rotation_matrix(axis, angle_deg):

    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)

    th = np.deg2rad(angle_deg)

    c = np.cos(th)
    s = np.sin(th)

    x, y, z = axis

    K = np.array([
        [0, -z,  y],
        [z,  0, -x],
        [-y, x,  0]
    ])

    return np.eye(3) + s*K + (1-c)*(K @ K)

R = rotation_matrix(
    axis=[1, 1, 1],
    angle_deg=17
)

rotations = [
    np.eye(3),
    rotation_matrix([1,0,0], 17),
    rotation_matrix([0,1,0], 31),
    rotation_matrix([0,0,1], 47),
]




import vir.sys_mat.analytic_sino as asino
dirs, weights = lebedev_rule(131)

dirs = np.asarray(dirs, dtype=float).T
test = asino.analytic_sino_sphere2_avg(src0,det0,u_bnd,v_bnd,u_hat_rot,v_hat_rot,\
                                (0.0,0.0,0.0),r,1,
    dirs=dirs,
    weights=weights,
    rotations=rotations)

    
    
        
from scipy.integrate import lebedev_rule


dirs, weights = lebedev_rule(83)

test83=asino.analytic_sino_sphere(src0,det0,u_bnd,v_bnd,u_hat0,v_hat0,\
                                  (0,0,0),50,1,dirs.T,weights)


dirs, weights = lebedev_rule(83)
test83=asino.analytic_sino_sphere2(src0,det0,u_bnd,v_bnd,u_hat0,v_hat0,\
                                (0,0,0),50,1,dirs.T,weights)

    
dirs, weights = lebedev_rule(95)
test95=asino.analytic_sino_sphere2(src0,det0,u_bnd,v_bnd,u_hat0,v_hat0,\
                                (0,0,0),50,1,dirs.T,weights)


    
dirs, weights = lebedev_rule(101)
test101=asino.analytic_sino_sphere2(src0,det0,u_bnd,v_bnd,u_hat0,v_hat0,\
                                (0,0,0),50,1,dirs.T,weights)

dirs, weights = lebedev_rule(107)
test107=asino.analytic_sino_sphere2(src0,det0,u_bnd,v_bnd,u_hat0,v_hat0,\
                                (0,0,0),50,1,dirs.T,weights)

dirs, weights = lebedev_rule(113)
test113=asino.analytic_sino_sphere2(src0,det0,u_bnd,v_bnd,u_hat0,v_hat0,\
                                (0,0,0),50,1,dirs.T,weights)


dirs, weights = lebedev_rule(131)
test131=asino.analytic_sino_sphere2(src0,det0,u_bnd,v_bnd,u_hat0,v_hat0,\
                                (0,0,0),50,1,dirs.T,weights)

    