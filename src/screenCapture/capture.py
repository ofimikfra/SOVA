import mss  
import numpy as np  
import cv2  

_sct = mss.mss()
_monitor = _sct.monitors[1]  # get primary monitor

# capture & return current screen frame as BGR img
def getScreenFrame():

    with mss.mss() as sct:
        monitor = sct.monitors[1]  # primary monitor
        screenshot = sct.grab(monitor)  # capture monitor
        
        frame = np.array(screenshot)  # screenshot to numpy array
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)  # bgra to bgr color conversion
        
        return frame