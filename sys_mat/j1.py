import numpy as np

def joseph_forward_3d(volume, sources, detectors):
    """
    3D Joseph forward projection with arbitrary ray geometry.

    Parameters
    ----------
    volume : 3D numpy array (Z, Y, X)
    sources : (N, 3) array of (x, y, z)
    detectors : (N, 3) array of (x, y, z)

    Returns
    -------
    projections : (N,) numpy array
    """

    Z, Y, X = volume.shape
    projections = np.zeros(len(sources), dtype=np.float64)

    for i, (src, det) in enumerate(zip(sources, detectors)):
        x0, y0, z0 = src
        x1, y1, z1 = det

        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0

        # Choose dominant axis
        if abs(dx) >= abs(dy) and abs(dx) >= abs(dz):
            xs = np.arange(np.ceil(min(x0, x1)), np.floor(max(x0, x1)) + 1)
            ts = (xs - x0) / dx
            ys = y0 + ts * dy
            zs = z0 + ts * dz
        elif abs(dy) >= abs(dx) and abs(dy) >= abs(dz):
            ys = np.arange(np.ceil(min(y0, y1)), np.floor(max(y0, y1)) + 1)
            ts = (ys - y0) / dy
            xs = x0 + ts * dx
            zs = z0 + ts * dz
        else:
            zs = np.arange(np.ceil(min(z0, z1)), np.floor(max(z0, z1)) + 1)
            ts = (zs - z0) / dz
            xs = x0 + ts * dx
            ys = y0 + ts * dy

        s = 0.0
        for x, y, z in zip(xs, ys, zs):
            ix = int(np.floor(x))
            iy = int(np.floor(y))
            iz = int(np.floor(z))

            if (
                0 <= ix < X - 1 and
                0 <= iy < Y - 1 and
                0 <= iz < Z - 1
            ):
                wx = x - ix
                wy = y - iy
                wz = z - iz

                # Trilinear interpolation
                for dz_i in (0, 1):
                    for dy_i in (0, 1):
                        for dx_i in (0, 1):
                            w = (
                                (1 - wx if dx_i == 0 else wx) *
                                (1 - wy if dy_i == 0 else wy) *
                                (1 - wz if dz_i == 0 else wz)
                            )
                            s += w * volume[
                                iz + dz_i,
                                iy + dy_i,
                                ix + dx_i
                            ]

        projections[i] = s

    return projections


def joseph_backprojection_3d(projections, sources, detectors, volume_shape):
    """
    3D Joseph backprojection with arbitrary ray geometry.
    """

    Z, Y, X = volume_shape
    volume = np.zeros((Z, Y, X), dtype=np.float64)

    for val, src, det in zip(projections, sources, detectors):
        x0, y0, z0 = src
        x1, y1, z1 = det

        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0

        if abs(dx) >= abs(dy) and abs(dx) >= abs(dz):
            xs = np.arange(np.ceil(min(x0, x1)), np.floor(max(x0, x1)) + 1)
            ts = (xs - x0) / dx
            ys = y0 + ts * dy
            zs = z0 + ts * dz
        elif abs(dy) >= abs(dx) and abs(dy) >= abs(dz):
            ys = np.arange(np.ceil(min(y0, y1)), np.floor(max(y0, y1)) + 1)
            ts = (ys - y0) / dy
            xs = x0 + ts * dx
            zs = z0 + ts * dz
        else:
            zs = np.arange(np.ceil(min(z0, z1)), np.floor(max(z0, z1)) + 1)
            ts = (zs - z0) / dz
            xs = x0 + ts * dx
            ys = y0 + ts * dy

        for x, y, z in zip(xs, ys, zs):
            ix = int(np.floor(x))
            iy = int(np.floor(y))
            iz = int(np.floor(z))

            if (
                0 <= ix < X - 1 and
                0 <= iy < Y - 1 and
                0 <= iz < Z - 1
            ):
                wx = x - ix
                wy = y - iy
                wz = z - iz

                for dz_i in (0, 1):
                    for dy_i in (0, 1):
                        for dx_i in (0, 1):
                            w = (
                                (1 - wx if dx_i == 0 else wx) *
                                (1 - wy if dy_i == 0 else wy) *
                                (1 - wz if dz_i == 0 else wz)
                            )
                            volume[
                                iz + dz_i,
                                iy + dy_i,
                                ix + dx_i
                            ] += w * val

    return volume


import numpy as np
import matplotlib.pyplot as plt

# Small 3D phantom
N = 32
vol = np.zeros((N, N, N))
vol[10:22, 10:22, 10:22] = 1.0  # cube

# Simple parallel-beam geometry
sources = []
detectors = []

cx = cy = cz = (N - 1) / 2

for theta in np.linspace(0, np.pi, 10, endpoint=False):
    for z in np.linspace(0, N-1, 8):
        sx = cx + 80 * np.cos(theta)
        sy = cy + 80 * np.sin(theta)
        sz = z

        dx = cx - 80 * np.cos(theta)
        dy = cy - 80 * np.sin(theta)
        dz = z

        sources.append((sx, sy, sz))
        detectors.append((dx, dy, dz))

sources = np.array(sources)
detectors = np.array(detectors)

# Forward + backprojection
proj = joseph_forward_3d(vol, sources, detectors)
recon = joseph_backprojection_3d(proj, sources, detectors, vol.shape)

# Display central slice
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.imshow(vol[N//2], cmap="gray")
plt.title("Original (central slice)")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(recon[N//2], cmap="gray")
plt.title("Backprojection (central slice)")
plt.axis("off")

plt.show()