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

    
def box_dir(research_folder=True):

    box_drive_locations = []

    if research_folder == True:
        research_folder =  os.path.join('Research','Projects') + os.sep 
    else:
        research_folder = ''

    #Mac Location
    box_drive_locations.append('/Users/vargasp/Library/CloudStorage/Box-Box/')
    box_drive_locations.append('/Users/vargasp/Box/')
    box_drive_locations.append('/Users/pvargas21/Library/CloudStorage/Box-Box/')

    #PC location
    box_drive_locations.append('C:\\Users\\vargasp\\Box\\')

    #MEL Location
    box_drive_locations.append('/home/vargasp/Box/')
 
    #Checks box location possibilities
    for box_driveLocation in box_drive_locations:
        if os.path.exists(box_driveLocation):
            return box_driveLocation  + research_folder

    return ''

