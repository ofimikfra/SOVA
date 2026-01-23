import mss
import numpy as np
import cv2

_sct = mss.mss()              # create ONCE
_monitor = _sct.monitors[1]   # primary screen


def getScreenFrame():
    screenshot = _sct.grab(_monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame
cap = cv2.VideoCapture(0)  # webcam

def getScreenFrame():
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read frame from webcam")
    return frame

def releaseCamera():
    cap.release()
