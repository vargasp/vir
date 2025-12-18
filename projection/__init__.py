#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 22:27:39 2020

@author: vargasp
"""


modules = ["proj"]

try:
    import multiprocessing as mp
    import ctypes
    modules.append("proj_ctype")
except:
    print("proj_mpct not laoded")

try:
    import multiprocessing as mp
    import ctypes
    modules.append("proj_mp")
except:
    print("proj_mp not laoded")


try:
    import numba
    modules.append("proj_numba")
except:
    print("proj_numba not laoded")


try:
    from numba import cuda
    modules.append("proj_cuda")
except:
    print("proj_cuda not laoded")


__all__ = modules