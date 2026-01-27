import cv2
import numpy as np
import mediapipe as mp
import mss
import joblib

# ---------------- Load trained model ----------------
model = joblib.load("expression_model.pkl")  # replace with your trained model path

# ---------------- MediaPipe Face Mesh ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------- Eye aspect ratio ----------------
def eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    return (A + B) / (2.0 * C)

# ---------------- Screen capture ----------------
_sct = mss.mss()
_monitor = _sct.monitors[1]  # primary monitor

def get_screen_frame():
    screenshot = _sct.grab(_monitor)
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame

# ---------------- Landmark indices ----------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14]  # left, right, top, bottom
LEFT_EYEBROW = [70, 63]
RIGHT_EYEBROW = [300, 293]

# ---------------- Wink state ----------------
left_eye_closed = False
right_eye_closed = False
ear_threshold = 0.25

# ---------------- Main loop ----------------
while True:
    frame = get_screen_frame()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    expression_label = "Neutral"

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark])

            # Face box
            x_min, y_min = np.min(landmarks[:,0]), np.min(landmarks[:,1])
            x_max, y_max = np.max(landmarks[:,0]), np.max(landmarks[:,1])
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 2)

            # -------- 10 Features --------
            left_ear = eye_aspect_ratio([landmarks[i] for i in LEFT_EYE])
            right_ear = eye_aspect_ratio([landmarks[i] for i in RIGHT_EYE])
            mouth_width = np.linalg.norm(landmarks[MOUTH[0]] - landmarks[MOUTH[1]])
            mouth_height = np.linalg.norm(landmarks[MOUTH[2]] - landmarks[MOUTH[3]])
            left_eyebrow_dist = np.linalg.norm(landmarks[LEFT_EYE[0]] - landmarks[LEFT_EYEBROW[0]])
            right_eyebrow_dist = np.linalg.norm(landmarks[RIGHT_EYE[0]] - landmarks[RIGHT_EYEBROW[0]])
            # Slopes for mouth sides
            mouth_left_slope = (landmarks[MOUTH[2]][1] - landmarks[MOUTH[0]][1]) / (landmarks[MOUTH[2]][0] - landmarks[MOUTH[0]][0] + 1e-6)
            mouth_right_slope = (landmarks[MOUTH[2]][1] - landmarks[MOUTH[1]][1]) / (landmarks[MOUTH[2]][0] - landmarks[MOUTH[1]][0] + 1e-6)
            lip_corner_diff = landmarks[MOUTH[0]][1] - landmarks[MOUTH[1]][1]
            eyebrow_slope_diff = ((landmarks[LEFT_EYEBROW[1]][1] - landmarks[LEFT_EYEBROW[0]][1]) /
                                  (landmarks[LEFT_EYEBROW[1]][0] - landmarks[LEFT_EYEBROW[0]][0] + 1e-6)) - \
                                 ((landmarks[RIGHT_EYEBROW[1]][1] - landmarks[RIGHT_EYEBROW[0]][1]) /
                                  (landmarks[RIGHT_EYEBROW[1]][0] - landmarks[RIGHT_EYEBROW[0]][0] + 1e-6))

            feature_vector = np.array([
                left_ear, right_ear, mouth_width, mouth_height,
                left_eyebrow_dist, right_eyebrow_dist,
                mouth_left_slope, mouth_right_slope, lip_corner_diff, eyebrow_slope_diff
            ]).reshape(1, -1)

            # -------- Wink detection --------
            if left_ear < ear_threshold and not left_eye_closed:
                expression_label = "Left Wink"
                left_eye_closed = True
            elif left_ear >= ear_threshold:
                left_eye_closed = False

            if right_ear < ear_threshold and not right_eye_closed:
                expression_label = "Right Wink"
                right_eye_closed = True
            elif right_ear >= ear_threshold:
                right_eye_closed = False

            # -------- Model prediction --------
            if expression_label == "Neutral":
                pred = model.predict(feature_vector)[0]
                expression_label = pred

            # Display label
            cv2.putText(frame, expression_label, (x_min, y_max + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Live Expression Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cv2.destroyAllWindows()
