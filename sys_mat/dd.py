import numpy as np


def project_point_to_detector(P, S, D0, u_hat, v_hat):
    """
    Project a 3D point onto a planar detector with uniform spacing.

    Parameters
    ----------
    P : ndarray, shape (8,3,)
        3D point in world coordinates.
    S : ndarray, shape (3,)
        Source position.
    D0 : ndarray, shape (3,)
        Detector origin.
    u_hat, v_hat : ndarray, shape (3,)
        Unit vectors defining detector axes.

    Returns
    -------
    u, v : float
        Detector coordinates in physical units.
    """
    r = P - S
    n = np.cross(u_hat, v_hat)
    
    t = np.sum( (D0 - S)* n) / np.sum(r*n,axis=-1)
    
    Q = S + t[:,np.newaxis] * r

    u = np.sum((Q - D0)* u_hat,axis=-1)
    v = np.sum((Q - D0)* v_hat,axis=-1)

    return u, v


def precompute_dd_weights_3d_uniform(
    nVoxel,
    dVoxel,
    sources,
    detectors,
    du,
    dv,
    Nu,
    Nv
):
    """
    Precompute 3D distance-driven weights for uniform detector bins.

    Parameters
    ----------
    nVoxel : tuple
        (Nx, Ny, Nz) volume dimensions.
    dVoxel : float
        Isotropic voxel edge length.
    sources : ndarray, shape (V, 3)
        Source positions.
    detectors : list of tuples
        Each entry: (D0, u_hat, v_hat, Nu, Nv)
    du, dv : float
        Detector bin spacing along u and v.

    Returns
    -------
    weights : list
        weights[v] is a list of tuples:
        (ix, iy, iz, iu, iv, w)
    """
    Nx, Ny, Nz = nVoxel
    V = len(sources)
    weights = [[] for _ in range(V)]

    corners = np.empty([8,3],dtype=np.float32)
    ox = - (Nx * voxel_size) / 2
    oy = - (Ny * voxel_size) / 2
    oz = - (Nz * voxel_size) / 2

    for v in range(V):
        S = sources[v]
        D0, u_hat, v_hat = detectors[v]

        for ix in range(Nx):
            corners[:4,0] = ox + ix*voxel_size
            corners[4:,0] = ox + ix*voxel_size + voxel_size
            
            for iy in range(Ny):
                corners[[0,1,4,5],1] = oy + iy*voxel_size
                corners[[2,3,6,7],1] = oy + iy*voxel_size + voxel_size
                
                for iz in range(Nz):
                    corners[::2,2] = oz + iz*voxel_size
                    corners[1::2,2] = oz + iz*voxel_size + voxel_size

                   # project the voxel corner
                    #u_vals, v_vals = project_point_to_detector(corners, S, D0, u_hat, v_hat)
                    
                    r = corners - S
                    n = np.cross(u_hat, v_hat)
                    
                    t = np.sum( (D0 - S)* n) / np.sum(r*n,axis=-1)
                    
                    Q = S + t[:,np.newaxis] * r

                    u_vals = np.sum((Q - D0)* u_hat,axis=-1)
                    v_vals = np.sum((Q - D0)* v_hat,axis=-1)
                    
                    

                    u_min, u_max = min(u_vals), max(u_vals)
                    v_min, v_max = min(v_vals), max(v_vals)

                    iu0 = int(np.floor(u_min / du))
                    iu1 = int(np.ceil(u_max / du))
                    iv0 = int(np.floor(v_min / dv))
                    iv1 = int(np.ceil(v_max / dv))

                    for iu in range(iu0, iu1 + 1):
                        for iv in range(iv0, iv1 + 1):
                            if 0 <= iu < Nu and 0 <= iv < Nv:
                                ou = max(
                                    0.0,
                                    min(u_max, (iu + 1) * du) -
                                    max(u_min, iu * du)
                                )
                                ov = max(
                                    0.0,
                                    min(v_max, (iv + 1) * dv) -
                                    max(v_min, iv * dv)
                                )
                                w = ou * ov
                                if w > 0:
                                    weights[v].append((ix, iy, iz, iu, iv, w))

    return weights



def forward_project(vol, weights, Nu, Nv):
    """
    Forward-project volume using precomputed weights.
    """
    proj = np.zeros((len(weights), Nu, Nv))
    for v in range(len(weights)):
        for ix, iy, iz, iu, iv, w in weights[v]:
            proj[v, iu, iv] += vol[ix, iy, iz] * w
    return proj



def back_project(proj, weights, vol_shape):
    """
    Backproject projections using precomputed weights.
    """
    vol = np.zeros(vol_shape)
    for v in range(len(weights)):
        for ix, iy, iz, iu, iv, w in weights[v]:
            vol[ix, iy, iz] += proj[v, iu, iv] * w
    return vol



import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Import projector functions
# ----------------------------

# (Assume these are already defined or imported)
# - project_point_to_detector
# - precompute_dd_weights_3d_uniform
# - forward_project
# - back_project


# ----------------------------
# Volume definition
# ----------------------------

Nx, Ny, Nz = 16, 16, 16
voxel_size = 1.0

vol = np.zeros((Nx, Ny, Nz))
vol[Nx//2, Ny//2, Nz//2] = 1.0  # point object at center


# ----------------------------
# Cone-beam circular geometry
# ----------------------------

num_views = 20
radius = 40.0          # source-to-rotation-center
det_dist = 60.0        # source-to-detector distance

Nu, Nv = 32, 32
du, dv = 1.0, 1.0

sources = []
detectors = []

angles = np.linspace(0, 2*np.pi, num_views, endpoint=False)

for theta in angles:
    # Source position
    S = np.array([
        radius * np.cos(theta),
        radius * np.sin(theta),
        0.0
    ])
    sources.append(S)

    # Unit vector from source to rotation center
    src_to_iso = -S / np.linalg.norm(S)

    # Detector center (SDD away from source)
    det_center = S + det_dist * src_to_iso

    # Detector axes
    u_hat = np.array([-np.sin(theta), np.cos(theta), 0.0])
    v_hat = np.array([0.0, 0.0, 1.0])

    # Detector origin (lower-left corner)
    D0 = (
        det_center
        - (Nu * du / 2) * u_hat
        - (Nv * dv / 2) * v_hat
    )

    detectors.append((D0, u_hat, v_hat))

sources = np.array(sources)


# ----------------------------
# Precompute weights
# ----------------------------

print("Precomputing distance-driven weights...")
weights = precompute_dd_weights_3d_uniform(
    nVoxel=(Nx, Ny, Nz),
    dVoxel=voxel_size,
    sources=sources,
    detectors=detectors,
    du=du,
    dv=dv,
    Nu=Nu,
    Nv=Nv
)

print("Done.")


# ----------------------------
# Forward projection
# ----------------------------

print("Forward projecting...")
proj = forward_project(vol, weights, Nu, Nv)
print("Done.")


# ----------------------------
# Backprojection
# ----------------------------

print("Backprojecting...")
bp = back_project(proj, weights, vol.shape)
print("Done.")


# ----------------------------
# Visualization
# ----------------------------

view_id = 0
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.title("Projection (u-v)")
plt.imshow(proj[view_id], cmap="gray")
plt.colorbar()

plt.subplot(1, 3, 2)
plt.title("Backprojection (z mid-slice)")
plt.imshow(bp[:, :, Nz//2], cmap="hot")
plt.colorbar()

plt.subplot(1, 3, 3)
plt.title("Original volume (z mid-slice)")
plt.imshow(vol[:, :, Nz//2], cmap="hot")
plt.colorbar()

plt.tight_layout()
plt.show()