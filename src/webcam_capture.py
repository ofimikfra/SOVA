import numpy as np
import mss
import cv2

sct = mss.mss()

def getCameraFrame():
    monitor = sct.monitors[1]  # full screen (change to [2] if you have dual monitors and Meet is on second screen)
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)
    frame = frame[:, :, :3]  # drop alpha, keep BGR
    frame = cv2.resize(frame, (1280, 720))  # normalize resolution
    return frame