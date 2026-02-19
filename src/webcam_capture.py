import cv2

cap = cv2.VideoCapture(0)  # webcam

def getCameraFrame():
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read frame from webcam")
    return frame
