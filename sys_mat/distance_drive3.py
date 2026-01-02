# -*- coding: utf-8 -*-
"""
Created on Wed Dec 31 08:00:19 2025

@author: varga
"""

import numpy as np
import matplotlib.pyplot as plt

def forward_project_dd_separable(
    volume,          # (Nx, Ny, Nz)
    dVoxel,
    sources,         # (V, 3)
    detectors,       # list of (D0, u_hat, v_hat)
    du, dv,
    Nu, Nv
):
    Nx, Ny, Nz = volume.shape
    V = len(sources)

    proj = np.zeros((V, Nu, Nv), dtype=np.float32)

    # volume origin (centered)
    ox = -0.5 * Nx * dVoxel
    oy = -0.5 * Ny * dVoxel
    oz = -0.5 * Nz * dVoxel

    for v in range(V):
        S = sources[v]
        D0, u_hat, v_hat = detectors[v]
        n = np.cross(u_hat, v_hat)

        for iu in range(Nu):
            u0 = iu * du
            u1 = (iu + 1) * du

            for iv in range(Nv):
                v0 = iv * dv
                v1 = (iv + 1) * dv

                # detector bin center
                Dc = D0 + 0.5 * (u0 + u1) * u_hat + 0.5 * (v0 + v1) * v_hat
                ray = Dc - S
                rx, ry, rz = ray

                #
                L = np.linalg.norm(ray)
                ray_hat = ray / L
                jac = abs(np.dot(ray_hat, n)) / (L * L)

                # choose driving axis
                ax = np.argmax(np.abs(ray))

                # =========================
                # X-driving
                # =========================
                if ax == 0:
                    if abs(rx) < 1e-6:
                        continue

                    for ix in range(Nx):
                        x0 = ox + ix * dVoxel
                        x1 = x0 + dVoxel

                        t0 = (x0 - S[0]) / rx
                        t1 = (x1 - S[0]) / rx
                        if t0 > t1:
                            t0, t1 = t1, t0
                        if t1 <= 0:
                            continue

                        tmid = 0.5 * (t0 + t1)
                        y = S[1] + tmid * ry
                        z = S[2] + tmid * rz

                        iy = int((y - oy) / dVoxel)
                        iz = int((z - oz) / dVoxel)
                        if iy < 0 or iy >= Ny or iz < 0 or iz >= Nz:
                            continue

                        Q0 = S + t0 * ray
                        Q1 = S + t1 * ray

                        uQ = np.dot(np.vstack((Q0 - D0, Q1 - D0)), u_hat)
                        vQ = np.dot(np.vstack((Q0 - D0, Q1 - D0)), v_hat)

                        ou = max(0.0, min(u1, uQ.max()) - max(u0, uQ.min()))
                        ov = max(0.0, min(v1, vQ.max()) - max(v0, vQ.min()))

                        if ou > 0 and ov > 0:
                            proj[v, iu, iv] += volume[ix, iy, iz] * ou * ov * jac

                # =========================
                # Y-driving
                # =========================
                elif ax == 1:
                    if abs(ry) < 1e-6:
                        continue

                    for iy in range(Ny):
                        y0 = oy + iy * dVoxel
                        y1 = y0 + dVoxel

                        t0 = (y0 - S[1]) / ry
                        t1 = (y1 - S[1]) / ry
                        if t0 > t1:
                            t0, t1 = t1, t0
                        if t1 <= 0:
                            continue

                        tmid = 0.5 * (t0 + t1)
                        x = S[0] + tmid * rx
                        z = S[2] + tmid * rz

                        ix = int((x - ox) / dVoxel)
                        iz = int((z - oz) / dVoxel)
                        if ix < 0 or ix >= Nx or iz < 0 or iz >= Nz:
                            continue

                        Q0 = S + t0 * ray
                        Q1 = S + t1 * ray

                        uQ = np.dot(np.vstack((Q0 - D0, Q1 - D0)), u_hat)
                        vQ = np.dot(np.vstack((Q0 - D0, Q1 - D0)), v_hat)

                        ou = max(0.0, min(u1, uQ.max()) - max(u0, uQ.min()))
                        ov = max(0.0, min(v1, vQ.max()) - max(v0, vQ.min()))

                        if ou > 0 and ov > 0:
                            proj[v, iu, iv] += volume[ix, iy, iz] * ou * ov

                # =========================
                # Z-driving
                # =========================
                else:
                    if abs(rz) < 1e-6:
                        continue

                    for iz in range(Nz):
                        z0 = oz + iz * dVoxel
                        z1 = z0 + dVoxel

                        t0 = (z0 - S[2]) / rz
                        t1 = (z1 - S[2]) / rz
                        if t0 > t1:
                            t0, t1 = t1, t0
                        if t1 <= 0:
                            continue

                        tmid = 0.5 * (t0 + t1)
                        x = S[0] + tmid * rx
                        y = S[1] + tmid * ry

                        ix = int((x - ox) / dVoxel)
                        iy = int((y - oy) / dVoxel)
                        if ix < 0 or ix >= Nx or iy < 0 or iy >= Ny:
                            continue

                        Q0 = S + t0 * ray
                        Q1 = S + t1 * ray

                        uQ = np.dot(np.vstack((Q0 - D0, Q1 - D0)), u_hat)
                        vQ = np.dot(np.vstack((Q0 - D0, Q1 - D0)), v_hat)

                        ou = max(0.0, min(u1, uQ.max()) - max(u0, uQ.min()))
                        ov = max(0.0, min(v1, vQ.max()) - max(v0, vQ.min()))

                        if ou > 0 and ov > 0:
                            proj[v, iu, iv] += volume[ix, iy, iz] * ou * ov

    return proj



def backproject_dd_separable(
    proj,            # (V, Nu, Nv)
    vol_shape,       # (Nx, Ny, Nz)
    dVoxel,
    sources,         # list of (3,)
    detectors,       # list of (D0, u_hat, v_hat)
    du, dv,
    Nu, Nv
):
    Nx, Ny, Nz = vol_shape
    V = len(sources)

    volume = np.zeros(vol_shape, dtype=np.float32)

    # volume origin (centered grid)
    ox = -0.5 * Nx * dVoxel
    oy = -0.5 * Ny * dVoxel
    oz = -0.5 * Nz * dVoxel

    for v in range(V):
        S = sources[v]
        D0, u_hat, v_hat = detectors[v]
        n = np.cross(u_hat, v_hat)

        for iu in range(Nu):
            u0 = iu * du
            u1 = (iu + 1) * du

            for iv in range(Nv):
                v0 = iv * dv
                v1 = (iv + 1) * dv

                pval = proj[v, iu, iv]
                if pval == 0:
                    continue

                # detector bin center
                Dc = (
                    D0
                    + 0.5 * (u0 + u1) * u_hat
                    + 0.5 * (v0 + v1) * v_hat
                )

                ray = Dc - S
                rx, ry, rz = ray

                L = np.linalg.norm(ray)
                if L < 1e-6:
                    continue

                ray_hat = ray / L

                # cone-beam Jacobian (must match forward projector)
                jac = abs(np.dot(ray_hat, n)) / (L * L)

                val = pval * jac

                # choose driving axis
                ax = np.argmax(np.abs(ray))

                # =========================
                # X-driving
                # =========================
                if ax == 0 and abs(rx) > 1e-6:
                    for ix in range(Nx):
                        x0 = ox + ix * dVoxel
                        x1 = x0 + dVoxel

                        t0 = (x0 - S[0]) / rx
                        t1 = (x1 - S[0]) / rx
                        if t0 > t1:
                            t0, t1 = t1, t0
                        if t1 <= 0:
                            continue

                        tmid = 0.5 * (t0 + t1)

                        y = S[1] + tmid * ry
                        z = S[2] + tmid * rz

                        iy = int((y - oy) / dVoxel)
                        iz = int((z - oz) / dVoxel)

                        if iy < 0 or iy >= Ny or iz < 0 or iz >= Nz:
                            continue

                        # project slab endpoints
                        Q0 = S + t0 * ray
                        Q1 = S + t1 * ray

                        uQ = np.dot(
                            np.vstack((Q0 - D0, Q1 - D0)),
                            u_hat
                        )
                        vQ = np.dot(
                            np.vstack((Q0 - D0, Q1 - D0)),
                            v_hat
                        )

                        ou = max(
                            0.0,
                            min(u1, uQ.max()) - max(u0, uQ.min())
                        )
                        ov = max(
                            0.0,
                            min(v1, vQ.max()) - max(v0, vQ.min())
                        )

                        if ou > 0 and ov > 0:
                            volume[ix, iy, iz] += val * ou * ov

                # =========================
                # Y-driving
                # =========================
                elif ax == 1 and abs(ry) > 1e-6:
                    for iy in range(Ny):
                        y0 = oy + iy * dVoxel
                        y1 = y0 + dVoxel

                        t0 = (y0 - S[1]) / ry
                        t1 = (y1 - S[1]) / ry
                        if t0 > t1:
                            t0, t1 = t1, t0
                        if t1 <= 0:
                            continue

                        tmid = 0.5 * (t0 + t1)

                        x = S[0] + tmid * rx
                        z = S[2] + tmid * rz

                        ix = int((x - ox) / dVoxel)
                        iz = int((z - oz) / dVoxel)

                        if ix < 0 or ix >= Nx or iz < 0 or iz >= Nz:
                            continue

                        Q0 = S + t0 * ray
                        Q1 = S + t1 * ray

                        uQ = np.dot(
                            np.vstack((Q0 - D0, Q1 - D0)),
                            u_hat
                        )
                        vQ = np.dot(
                            np.vstack((Q0 - D0, Q1 - D0)),
                            v_hat
                        )

                        ou = max(
                            0.0,
                            min(u1, uQ.max()) - max(u0, uQ.min())
                        )
                        ov = max(
                            0.0,
                            min(v1, vQ.max()) - max(v0, vQ.min())
                        )

                        if ou > 0 and ov > 0:
                            volume[ix, iy, iz] += val * ou * ov

                # =========================
                # Z-driving
                # =========================
                elif ax == 2 and abs(rz) > 1e-6:
                    for iz in range(Nz):
                        z0 = oz + iz * dVoxel
                        z1 = z0 + dVoxel

                        t0 = (z0 - S[2]) / rz
                        t1 = (z1 - S[2]) / rz
                        if t0 > t1:
                            t0, t1 = t1, t0
                        if t1 <= 0:
                            continue

                        tmid = 0.5 * (t0 + t1)

                        x = S[0] + tmid * rx
                        y = S[1] + tmid * ry

                        ix = int((x - ox) / dVoxel)
                        iy = int((y - oy) / dVoxel)

                        if ix < 0 or ix >= Nx or iy < 0 or iy >= Ny:
                            continue

                        Q0 = S + t0 * ray
                        Q1 = S + t1 * ray

                        uQ = np.dot(
                            np.vstack((Q0 - D0, Q1 - D0)),
                            u_hat
                        )
                        vQ = np.dot(
                            np.vstack((Q0 - D0, Q1 - D0)),
                            v_hat
                        )

                        ou = max(
                            0.0,
                            min(u1, uQ.max()) - max(u0, uQ.min())
                        )
                        ov = max(
                            0.0,
                            min(v1, vQ.max()) - max(v0, vQ.min())
                        )

                        if ou > 0 and ov > 0:
                            volume[ix, iy, iz] += val * ou * ov

    return volume




def make_sphere_phantom(Nx, Ny, Nz, radius):
    x = np.linspace(-1, 1, Nx)
    y = np.linspace(-1, 1, Ny)
    z = np.linspace(-1, 1, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    phantom = ((X**2 + Y**2 + Z**2) <= radius**2).astype(np.float32)
    return phantom


def make_geometry(V, R_orbit=50.0, R_SD=100.0):
    """
    Generate source and detector positions for circular cone-beam geometry.

    Parameters
    ----------
    V : int
        Number of views
    R_orbit : float
        Radius of source orbit around the object
    R_SD : float
        Source-to-detector distance

    Returns
    -------
    sources : list of ndarray, shape (3,)
        Source positions for each view
    detectors : list of tuples
        Each tuple: (D0, u_hat, v_hat)
        D0 = detector center position
        u_hat, v_hat = detector axes (orthonormal)
    """
    sources = []
    detectors = []

    for v in range(V):
        angle = 2 * np.pi * v / V

        # Source position on circular orbit in XY plane
        S = np.array([R_orbit * np.cos(angle),
                      R_orbit * np.sin(angle),
                      0.0])

        # Detector plane normal (points toward source)
        n_hat = -S / np.linalg.norm(S)  # points from detector center to source

        # Arbitrary up vector (avoid parallel to n_hat)
        up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(up, n_hat)) > 0.99:
            up = np.array([1.0, 0.0, 0.0])

        # Detector axes
        u_hat = np.cross(up, n_hat)
        u_hat /= np.linalg.norm(u_hat)

        v_hat = np.cross(n_hat, u_hat)

        # Detector center at distance R_SD along normal from source
        D0 = S + n_hat * R_SD

        sources.append(S)
        detectors.append((D0, u_hat, v_hat))

    return sources, detectors



# -------------------------
# Parameters
# -------------------------
Nx = Ny = Nz = 32
Nu = Nv = 32
V = 16

dVoxel = 1.0
du = dv = 1.0

# -------------------------
# Phantom
# -------------------------
volume = make_sphere_phantom(Nx, Ny, Nz, radius=0.5)

# -------------------------
# Geometry
# -------------------------
sources, detectors = make_geometry(V)

# -------------------------
# Forward projection
# -------------------------
print("Running forward projection...")
proj = forward_project_dd_separable(
    volume,
    dVoxel,
    sources,
    detectors,
    du, dv,
    Nu, Nv
)

# -------------------------
# Backprojection
# -------------------------
print("Running backprojection...")
backproj = backproject_dd_separable(
    proj,
    volume.shape,
    dVoxel,
    sources,
    detectors,
    du, dv,
    Nu, Nv
)

# -------------------------
# Visualization
# -------------------------
midz = Nz // 2
midv = V // 2

plt.figure(figsize=(12, 4))

# Phantom slice
plt.subplot(1, 3, 1)
plt.imshow(volume[:, :, midz], cmap="gray")
plt.title("Phantom (central slice)")
plt.colorbar()

# Sinogram (one view)
plt.subplot(1, 3, 2)
plt.imshow(proj[midv], cmap="inferno")
plt.title(f"Projection (view {midv})")
plt.xlabel("v")
plt.ylabel("u")
plt.colorbar()

# Backprojection slice
plt.subplot(1, 3, 3)
plt.imshow(backproj[:, :, midz], cmap="gray")
plt.title("Backprojection (central slice)")
plt.colorbar()

plt.tight_layout()
plt.show()

# -------------------------
# Optional: adjoint test
# -------------------------
lhs = np.sum(proj * proj)
rhs = np.sum(volume * backproj)
print("Adjoint check:")
print(" <Ax, Ax> =", lhs)
print(" <x, A^T Ax> =", rhs)
print(" Relative diff =", abs(lhs - rhs) / max(lhs, rhs))

