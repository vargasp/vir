#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:23:54 2022

@author: vargasp
"""

#from os.path import join, sep, abspath

import os


def file_path(fname):
    home_dir = os.path.expanduser('~')
    work_dir = os.getcwd()
    
    #If fname correctly starts with home_dir assume fname is a correctly setup
    if fname.find(home_dir) == 0:
        return fname
        
    #Remove seperators to correct for them in os.path.join
    if fname.count('\\') > 0:
        dirs = fname.split('\\')
    elif fname.count('/') > 0:
        dirs = fname.split('/')
    else:
        dirs = list(fname)
     
    return os.path.join(home_dir,*dirs)



def file_path_box(fname):
    
    
    file_path(fname):
    os.path.exuts(fname)

    os.system('rclone lsf Box' + fname)




rclone lsf remote:path-to-file

fname = '/Users/vargasp/Box/ktyt\params_arr.npy'
print(file_path(fname))


