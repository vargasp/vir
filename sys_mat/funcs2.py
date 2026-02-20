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


@njit
def censpace(du, nu):
    u_arr = np.empty(nu, dtype=np.float32)
    center = np.float32(nu - 1) * 0.5
    np.arange(nu, dtype=np.float32, out=u_arr)
    u_arr -= center
    u_arr *= du
    return u_arr