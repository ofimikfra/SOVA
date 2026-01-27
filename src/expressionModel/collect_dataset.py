import cv2
import numpy as np
import mss
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SAVE_PATH = os.path.join(BASE_DIR, "data")

EXPRESSIONS = {
    ord('1'): "neutral",
    ord('2'): "smile",
    ord('3'): "frown",
    ord('4'): "left_wink",
    ord('5'): "right_wink"
}

MONITOR = {
    "top": 200,
    "left": 400,
    "width": 500,
    "height": 500
}

os.makedirs(SAVE_PATH, exist_ok=True)
for label in EXPRESSIONS.values():
    os.makedirs(os.path.join(SAVE_PATH, label), exist_ok=True)

sct = mss.mss()

cv2.namedWindow("DEBUG", cv2.WINDOW_NORMAL)

print("PRESS 1–5 to save DUMMY samples | q to quit")

while True:
    img = np.array(sct.grab(MONITOR))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    cv2.imshow("DEBUG", frame)

    key = cv2.waitKey(1) & 0xFF

    if key != 255:
        print("KEY DETECTED:", key)

    if key == ord('q'):
        break

    if key in EXPRESSIONS:
        label = EXPRESSIONS[key]
        folder = os.path.join(SAVE_PATH, label)
        count = len(os.listdir(folder))

        dummy = np.random.rand(10)
        np.save(os.path.join(folder, f"{count}.npy"), dummy)

        print(f"[SAVED] {label} #{count}")

cv2.destroyAllWindows()
