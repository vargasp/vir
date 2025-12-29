import numpy as np

def precompute_rays(sources, detectors, volume_shape, voxel_size, n_samples=100):
    """
    Precompute ray paths and Joseph weights for all rays.
    
    Returns:
        rays: list of dictionaries with keys:
            'indices': list of (x, y, z) voxel indices
            'weights': corresponding interpolation weights
    """
    nx, ny, nz = volume_shape
    rays = []

    for src, det in zip(sources, detectors):
        ray_info = {'indices': [], 'weights': []}

        src = np.array(src)
        det = np.array(det)
        ray_vec = det - src
        ray_len = np.linalg.norm(ray_vec)
        ray_dir = ray_vec / ray_len

        # Sample along the ray (Joseph)
        ts = np.linspace(0, ray_len, n_samples)
        for t in ts:
            pos = src + ray_dir * t
            ix, iy, iz = pos / voxel_size
            ix0, iy0, iz0 = np.floor([ix, iy, iz]).astype(int)
            wx, wy, wz = ix - ix0, iy - iy0, iz - iz0

            # Store voxel indices and weights for trilinear interpolation
            for dx in [0,1]:
                for dy in [0,1]:
                    for dz in [0,1]:
                        cx, cy, cz = ix0+dx, iy0+dy, iz0+dz
                        if 0 <= cx < nx and 0 <= cy < ny and 0 <= cz < nz:
                            weight = ((1-wx) if dx==0 else wx) * \
                                     ((1-wy) if dy==0 else wy) * \
                                     ((1-wz) if dz==0 else wz) / n_samples
                            ray_info['indices'].append((cx,cy,cz))
                            ray_info['weights'].append(weight)
        rays.append(ray_info)
    return rays

def forward_projection_precomputed(volume, rays):
    """Fast forward projection using precomputed ray geometry."""
    projections = []
    for ray in rays:
        val = sum(volume[ix,iy,iz]*w for (ix,iy,iz), w in zip(ray['indices'], ray['weights']))
        projections.append(val)
    return np.array(projections)

def back_projection_precomputed(volume, rays, projections):
    """Fast backprojection using precomputed ray geometry."""
    for ray, proj_val in zip(rays, projections):
        for (ix,iy,iz), w in zip(ray['indices'], ray['weights']):
            volume[ix,iy,iz] += proj_val * w
            
            
# Volume parameters
nx, ny, nz = 64, 64, 64
voxel_size = np.array([1.0, 1.0, 1.0])
volume = np.zeros((nx, ny, nz))

# Define rays (sources and detector points)
sources = [np.array([-32,32,32]), np.array([-32,0,32])]
detectors = [np.array([96,32,32]), np.array([96,0,32])]

# Precompute all rays
rays = precompute_rays(sources, detectors, volume.shape, voxel_size, n_samples=50)

# Forward projection
proj = forward_projection_precomputed(volume, rays)
print("Forward projection:", proj)

# Example: Backprojection
back_projection_precomputed(volume, rays, proj)
print("Volume after backprojection sum:", np.sum(volume))