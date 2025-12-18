# -*- coding: utf-8 -*-
"""
Created on Wed Dec 17 15:12:10 2025

@author: varga
"""

import numpy as np
from numba import cuda
import time

# CUDA kernel
@cuda.jit
def vector_add_gpu(a, b, c):
    """
    Kernel function to add two vectors on the GPU.
    Each thread calculates one element of the result.
    """
    # Calculate the global thread position
    pos = cuda.grid(1)
    
    # Ensure the thread is within the array bounds
    if pos < c.size:
        c[pos] = a[pos] + b[pos]

def run_test():
    N = 10000000 # Size of the vectors
    
    # 1. Create data on the CPU
    a_host = np.random.random(N).astype(np.float32)
    b_host = np.random.random(N).astype(np.float32)
    c_host_gpu = np.zeros(N).astype(np.float32)
    
    # 2. Copy data from CPU to GPU memory
    d_a = cuda.to_device(a_host)
    d_b = cuda.to_device(b_host)
    d_c = cuda.device_array_like(d_a) # Create an empty device array for the result

    # Configure the kernel launch dimensions
    threads_per_block = 256
    blocks_per_grid = (N + threads_per_block - 1) // threads_per_block
    
    # 3. Launch the GPU kernel
    start_gpu_time = time.time()
    vector_add_gpu[blocks_per_grid, threads_per_block](d_a, d_b, d_c)
    cuda.synchronize() # Wait for the GPU to finish
    end_gpu_time = time.time()

    # 4. Copy the results from GPU to CPU memory
    d_c.copy_to_host(c_host_gpu)

    # 5. Verify the results with a CPU calculation
    start_cpu_time = time.time()
    c_host_cpu = a_host + b_host
    end_cpu_time = time.time()

    # Check for correctness
    assert np.allclose(c_host_cpu, c_host_gpu), "Results do not match!"
    print("Vector addition successful! CPU and GPU results match.")

    print(f"\nTime taken on CPU: {(end_cpu_time - start_cpu_time):.4f} seconds")
    print(f"Time taken on GPU: {(end_gpu_time - start_gpu_time):.4f} seconds")
    print(f"Speedup: {(end_cpu_time - start_cpu_time) / (end_gpu_time - start_gpu_time):.2f}x")

if __name__ == "__main__":
    run_test()