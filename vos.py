#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:23:54 2022

@author: vargasp
"""

#from os.path import join, sep, abspath

import os
import re

def file_path(fname):
    home_dir = os.path.expanduser('~')
    #work_dir = os.getcwd()
    
    #If fname correctly starts with home_dir assume fname is a correctly setup
    if fname.find(home_dir) == 0:
        fname = fname.replace(home_dir,'')
    else:
        home_dir = ''
    
    #Remove seperators to correct for them in os.path.join
    dirs = re.split('\\\\|/',fname)
    
    return os.path.join(home_dir,*dirs)
    """
    if home:
        return os.path.join(home_dir,*dirs)
    else:
        return os.path.join('/',*dirs)
     """  
    
def box_dir():

    box_drive_location1 = 'C:\\Users\\vargasp\\Box\\'
    box_drive_location2 = '/Users/pvargas21/Library/CloudStorage/Box-Box/'
    box_drive_location3 = '/Users/vargasp/Library/CloudStorage/Box-Box/'

    if os.path.exists(box_drive_location1):
        return box_drive_location1

    if os.path.exists(box_drive_location2):
        return box_drive_location2

    if os.path.exists(box_drive_location3):
        return box_drive_location3

    return ''



"""

def file_path_box(fname):
    
    
    file_path(fname):
    os.path.exuts(fname)

    os.system('rclone lsf Box' + fname)




rclone lsf remote:path-to-file

fname = '/Users/vargasp/Box/ktyt\params_arr.npy'
print(file_path(fname))


"""