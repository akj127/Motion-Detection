#! /usr/bin/env python
#  -*- coding: utf-8 -*-
#
# Support module generated by PAGE version 4.26
#  in conjunction with Tcl version 8.6
#    Nov 17, 2019 03:32:19 PM IST  platform: Windows NT
#    Nov 17, 2019 03:44:03 PM IST  platform: Windows NT

import sys

try:
    import Tkinter as tk
except ImportError:
    import tkinter as tk

try:
    import ttk
    py3 = False
except ImportError:
    import tkinter.ttk as ttk
    py3 = True

def Callibrate(p1):
    print('test_support.Callibrate')
    sys.stdout.flush()

def show_stream(p1):
    print('test_support.show_stream')
    sys.stdout.flush()

def start(p1):
    print('test_support.start')
    sys.stdout.flush()

def init(top, gui, *args, **kwargs):
    global w, top_level, root
    w = gui
    top_level = top
    root = top

def destroy_window():
    # Function which closes the window.
    global top_level
    top_level.destroy()
    top_level = None

if __name__ == '__main__':
    import test
    test.vp_start_gui()




