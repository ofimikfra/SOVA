import cv2

cap = cv2.VideoCapture(0)

def getCameraFrame():
    ret, frame = cap.read()
    if not ret:
        return None  # ← must be return None, not raise
    return frame