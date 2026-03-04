import numpy as np
import mss
import cv2
import threading
import subprocess

_local = threading.local()

def _get_sct():
    if not hasattr(_local, 'sct'):
        _local.sct = mss.mss()
    return _local.sct

def getScreenFrame():
    sct = _get_sct()

    # Find Google Meet window position using Windows
    try:
        import pygetwindow as gw
        windows = gw.getWindowsWithTitle("Meet")
        if not windows:
            windows = gw.getWindowsWithTitle("Google Meet")
        if not windows:
            print("[SOVA] Google Meet window not found, falling back to full screen")
            monitor = sct.monitors[1]
        else:
            win = windows[0]
            monitor = {
                "top":    win.top,
                "left":   win.left,
                "width":  win.width,
                "height": win.height,
            }
    except Exception as e:
        print(f"[SOVA] Window detection failed: {e}, falling back to full screen")
        monitor = sct.monitors[1]

    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = frame[:, :, :3]
    return frame