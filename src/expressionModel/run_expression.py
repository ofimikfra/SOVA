import cv2
import numpy as np
import mediapipe as mp
import mss
import joblib
from collections import deque

# ------------------ LOAD TRAINED MODEL ------------------
MODEL = joblib.load("expression_model.pkl")

LABEL_MAP = {
    0: "Neutral",
    1: "Smile",
    2: "Frown",
    3: "Left Wink",
    4: "Right Wink"
}

# ------------------ SCREEN CAPTURE SETTINGS ------------------
MONITOR = {"top": 100, "left": 100, "width": 640, "height": 480}
sct = mss.mss()

# ------------------ MEDIAPIPE FACE MESH ------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------ PREDICTION SMOOTHING ------------------
SMOOTHING_WINDOW = 5
pred_queue = deque(maxlen=SMOOTHING_WINDOW)

def smooth_prediction(pred):
    pred_queue.append(pred)
    counts = np.bincount(list(pred_queue))
    return np.argmax(counts)

# ------------------ LANDMARK EXTRACTION ------------------
def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return None, None

    landmarks = []
    h, w, _ = frame.shape
    xs, ys = [], []

    for lm in results.multi_face_landmarks[0].landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))

    landmarks = np.array(landmarks)

    # Normalize landmarks
    landmarks = landmarks - landmarks.mean()
    landmarks = landmarks / (landmarks.std() + 1e-6)

    # Compute bounding box
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)

    return landmarks, (x1, y1, x2, y2)

# ------------------ CREATE SINGLE WINDOW ------------------
cv2.namedWindow("Expression Recognition", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Expression Recognition", MONITOR["width"], MONITOR["height"])

# ------------------ MAIN LOOP ------------------
while True:
    # Grab screen frame
    img = np.array(sct.grab(MONITOR))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    landmarks, bbox = extract_landmarks(frame)

    if landmarks is not None:
        pred = MODEL.predict([landmarks])[0]
        smoothed_pred = smooth_prediction(pred)
        label = LABEL_MAP[smoothed_pred]

        x1, y1, x2, y2 = bbox

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label above the box
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    # Print detected expression to console
       
    cv2.imshow("Expression Recognition", frame)

    # Quit if 'q' pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
