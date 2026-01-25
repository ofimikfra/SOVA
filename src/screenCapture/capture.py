import mss
import numpy as np
import cv2

_sct = mss.mss()              # create ONCE
_monitor = _sct.monitors[1]   # primary screen


def getScreenFrame():
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary screen
        screenshot = sct.grab(monitor)
        
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        return frame