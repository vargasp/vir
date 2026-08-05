#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:23:54 2022

@author: vargasp
"""

#from os.path import join, sep, abspath

import os
import re

"""

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


from pathlib import Path, PurePosixPath, PureWindowsPath

def find_box_root():
    home = Path.home()

    candidates = [
        home / "Box",
        home / "Library" / "CloudStorage" / "Box-Box",   # macOS
        home / "Box Sync",                               # older installs
    ]

    for path in candidates:
        if path.exists():
            return path

    return None

BOX_ROOT = find_box_root()


def file_path(fname):
    """Return a path appropriate for the current OS."""

    home = Path.home()

    # ~/...
    if fname.startswith("~"):
        return str(Path(fname).expanduser())

    # /home/<user>/...
    posix = PurePosixPath(fname)
    if len(posix.parts) >= 3 and posix.parts[:2] == ("/", "home"):
        return str(home.joinpath(*posix.parts[3:]))

    # C:\Users\<user>\...
    win = PureWindowsPath(fname)
    if (win.drive and
            len(win.parts) >= 4 and
            win.parts[1].lower() == "users"):
        return str(home.joinpath(*win.parts[3:]))




    fname = fname.replace("\\", "/")
    return str(Path(*PurePosixPath(fname).parts))


        # Relative path (or anything else): just normalize separators
    #parts = [p for p in win.parts if p not in (win.drive, "\\", "/")]
    #return str(Path(*parts))



def file_path(fname, box_root=False):
    """
    Convert a path to one appropriate for the current operating system.

    Supports:

        ~/...
        /home/<user>/...
        C:\\Users\\<user>\\...
        box/...
        box\\...
        relative/path
        relative\\path
    """

    home = Path.home()

    # Normalize separators for parsing
    normalized = fname.replace("\\", "/")
    posix = PurePosixPath(normalized)

    if normalized.startswith("~"):
        return str(Path(normalized).expanduser())

    # box/...
    if box_root==True:
        if BOX_ROOT is None:
            raise FileNotFoundError("Could not locate the user's Box folder.")
        return str(BOX_ROOT.joinpath(*posix.parts))

    if posix.parts[0].lower() == "box":
        if BOX_ROOT is None:
            raise FileNotFoundError("Could not locate the user's Box folder.")
        return str(BOX_ROOT.joinpath(*posix.parts[1:]))


    # /home/<user>/...
    if len(posix.parts) >= 3 and posix.parts[:2] == ("/", "home"):
        return str(home.joinpath(*posix.parts[3:]))

    # C:\Users\<user>\...
    win = PureWindowsPath(fname)
    if (
        win.drive
        and len(win.parts) >= 4
        and win.parts[1].lower() == "users"
    ):
        return str(home.joinpath(*win.parts[3:]))

    # ------------------------------------------------------------------
    # Relative path
    # ------------------------------------------------------------------
    return str(Path(*posix.parts))