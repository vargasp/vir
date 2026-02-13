#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 13:50:14 2026

@author: pvargas21
"""

@njit(inline='always')
def _accumulate_fp(tmp_u, iu, img_val, overlap, ray_scl):
    tmp_u[iu] += img_val * overlap * ray_scl


@njit(inline='always')
def _accumulate_bp(tmp_u, iu, img_val, overlap, ray_scl):
    tmp_u[iu] += img_val * overlap * ray_scl  # different line here


@njit(fastmath=True)
def _dd_sweep_core(..., accumulate):
    while ip < ip_max and iu < iu_max:
        ...
        if overlap_r > overlap_l:
            accumulate(tmp_u, iu, img_vec[ip], overlap_r - overlap_l, ray_scl)
        ...
