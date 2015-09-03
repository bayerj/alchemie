# # # # # # # # # # # # # # # # # # # # #
# This is a hack to prevent problems when handling Ctrl-C and Ctrl-Break events after importing, e.g., scipy.stats on
# Windows. The fortran compiler coming along with scipy throws uncatchable errors when confronted with these events,
# thus the workaround is to catch the event before the fortran compiler gets aware and not passing it on.
# It is based on http://stackoverflow.com/questions/15457786/ctrl-c-crashes-python-after-importing-scipy-stats
import platform
if platform.system() == 'Windows':
    import ctypes
    import thread
    import win32api
    import imp
    import os

    # Load the DLL manually to ensure its handler gets
    # set before our handler.
    basepath = imp.find_module('numpy')[1]
    ctypes.CDLL(os.path.join(basepath, 'core', 'libmmd.dll'))
    ctypes.CDLL(os.path.join(basepath, 'core', 'libifcoremd.dll'))

    # Now set our handler for CTRL_C_EVENT. Other control event
    # types will chain to the next handler.
    def handler(dwCtrlType, hook_sigint=thread.interrupt_main):
        if dwCtrlType in [0,1]: # Ctrl-C or Ctrl-Break
            make_checkpoint()
            hook_sigint()
            return 1 # don't chain to the next handler
        return 0 # chain to the next handler

    win32api.SetConsoleCtrlHandler(handler, 1)
# end of hack
# # # # # # # # # # # # # # # # # # # # #


import sys
# double recursion depth to keep large models picklable
rl = sys.getrecursionlimit()
sys.setrecursionlimit(int(2*rl))