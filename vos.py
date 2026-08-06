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


    home = Path.home()

    # Normalize separators for parsing
    normalized = fname.replace("\\", "/")
    posix = PurePosixPath(normalized)

    if box_root==True and BOX_ROOT is None:
        raise FileNotFoundError("Could not locate the user's Box cloud folder.")


    if posix.parts[0].lower() == "box":
        if BOX_ROOT is None:
            raise FileNotFoundError("Could not locate the user's Box cloud folder.")


    targets = {"box", "box-box", "box sync"}
    idx = next((i for i, part in enumerate(posix.parts, start=1) if part.lower() in targets), None)
    if idx:
        #if box_root is False:
        #    print("Box folder in path, but BOX_ROOT=False")

        return str(BOX_ROOT.joinpath(*posix.parts[idx:]))

    # box/...
    if box_root:
        print("here")
        print("BOX_ROOT:", BOX_ROOT,posix)
        if BOX_ROOT is None:
            raise FileNotFoundError("Could not locate the user's Box folder.")
        return str(BOX_ROOT.joinpath(*posix.parts))


    if normalized.startswith("~"):
        return str(Path(normalized).expanduser())


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