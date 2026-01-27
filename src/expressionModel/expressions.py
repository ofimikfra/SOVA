import cv2
import numpy as np
import mediapipe as mp
import mss
import joblib

MODEL = joblib.load("expression_model.pkl")

LABEL_MAP = {
    0: "Neutral",
    1: "Smile",
    2: "Frown",
    3: "Left Wink",
    4: "Right Wink"
}

MONITOR = {
    "top": 100,
    "left": 100,
    "width": 640,
    "height": 480
}

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

sct = mss.mss()


def extract_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape


    mean_intensity = gray.mean()
    std_intensity = gray.std()


    features = np.array([
        mean_intensity,
        std_intensity,
        0, 0, 0, 0, 0, 0, 0, 0   # total = 10
    ], dtype=np.float32)

    # Normalize (same as training!)
    features = (features - features.mean()) / (features.std() + 1e-6)

    return features

cv2.namedWindow("Expression Recognition", cv2.WINDOW_NORMAL)

while True:
    img = np.array(sct.grab(MONITOR))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    landmarks = extract_landmarks(frame)

    if landmarks is not None:
        pred = MODEL.predict([landmarks])[0]
        label = LABEL_MAP[pred]
        cv2.putText(frame, label, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("Expression Recognition", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()