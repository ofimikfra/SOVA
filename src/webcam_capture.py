import cv2

cap = cv2.VideoCapture(0)  # webcam

def getScreenFrame():
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read frame from webcam")
    return frame

def releaseCamera():
    cap.release()
