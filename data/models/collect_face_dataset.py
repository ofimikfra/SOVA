import cv2
import numpy as np
import mediapipe as mp
import mss
import os

cv2.namedWindow("KeyCapture", cv2.WINDOW_NORMAL)
cv2.resizeWindow("KeyCapture", 1, 1)
cv2.moveWindow("KeyCapture", -100, -100)

SAVE_PATH = "dataset"
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

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

sct = mss.mss()


def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None

    landmarks = []

    for lm in results.multi_face_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])

    landmarks = np.array(landmarks)

    # 🔥 NORMALIZATION (CRITICAL)
    landmarks = landmarks - landmarks.mean()
    landmarks = landmarks / (landmarks.std() + 1e-6)

    return landmarks


print("Press 1–5 to save expressions. Q to quit.")

while True:
    img = np.array(sct.grab(MONITOR))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    landmarks = extract_landmarks(frame)

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key != 255:
        print("Key pressed:", key)

    if key == ord('s'):
        cv2.imwrite("test.jpg", frame)
        cv2.imwrite(f"{folder}/{count}.jpg", frame)

        print("Saved image!")

    if key in EXPRESSIONS and landmarks is not None:
        label = EXPRESSIONS[key]
        folder = os.path.join(SAVE_PATH, label)
        count = len(os.listdir(folder))
        np.save(f"{folder}/{count}.npy", landmarks)
        print(f"Saved {label} sample #{count}")


cv2.destroyAllWindows()

