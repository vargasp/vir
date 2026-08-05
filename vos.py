#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 16:23:54 2022

@author: vargasp
"""
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

    if box_root==True and BOX_ROOT is None:
        raise FileNotFoundError("Could not locate the user's Box folder.")


    if posix.parts[0].lower() == "box":
        if BOX_ROOT is None:
            raise FileNotFoundError("Could not locate the user's Box folder.")




    home = Path.home()

    # Normalize separators for parsing
    normalized = fname.replace("\\", "/")
    posix = PurePosixPath(normalized)

    if posix.parts[0].lower() == "box":
        if BOX_ROOT is None:
            raise FileNotFoundError("Could not locate the user's Box folder.")
        return str(BOX_ROOT.joinpath(*posix.parts[1:]))


    if normalized.startswith("~"):
        return str(Path(normalized).expanduser())

    # box/...
    if box_root==True:
        if BOX_ROOT is None:
            raise FileNotFoundError("Could not locate the user's Box folder.")
        return str(BOX_ROOT.joinpath(*posix.parts))



    # /home/<user>/...
    if (
        len(posix.parts) >= 4
        and posix.parts[0] == "/"
        and posix.parts[1] in ("home", "Users")
    ):
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