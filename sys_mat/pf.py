# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 10:07:49 2026

@author: varga
"""

def _identity_decorator(func):
    return func

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]      # @njit
        return _identity_decorator  # @njit(...)



import numpy as np




def as_float32(*args):
    out = []
    for x in args:
        if np.isscalar(x):
            out.append(np.float32(x))
        else:
            out.append(np.asarray(x, dtype=np.float32))
    return out


def as_int32(x):
    if np.isscalar(x):
        return np.int32(x)
    return np.asarray(x, dtype=np.int32)



@njit(inline='always', cache=True)
def censpace(d, n, s=0):
    arr = np.arange(n, dtype=np.float32)
    arr *= d
    arr += d*(s - (n - 1) * np.float32(0.5))
    return arr




@njit(inline='always', cache=True)
def boundspace(d, n, s=0):
    arr = np.arange(n + 1, dtype=np.float32)
    arr *= d
    arr += d*(s - n * np.float32(0.5))
    return arr

