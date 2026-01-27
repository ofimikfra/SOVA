import cv2
import numpy as np
import mediapipe as mp
import joblib
import mss

model = joblib.load("expression_model.pkl")
label_map = {0: "Neutral", 1: "Smile", 2: "Frown", 3: "Left Wink", 4: "Right Wink"}

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

sct = mss.mss()
monitor = sct.monitors[1]

while True:
    # Capture screen instead of webcam
    screenshot = sct.grab(monitor)
    frame = np.array(screenshot)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            def get_dist(p1_idx, p2_idx):
                p1, p2 = landmarks[p1_idx], landmarks[p2_idx]
                return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


            # Example features: Left EAR, Right EAR, Mouth Ratio
            l_ear = get_dist(159, 145) / (get_dist(33, 133) + 1e-6)
            r_ear = get_dist(386, 374) / (get_dist(362, 263) + 1e-6)
            m_width = get_dist(61, 291)

            features = np.array([l_ear, r_ear, m_width]).reshape(1, -1)

            # --- Prediction ---
            prediction = model.predict(features)[0]
            label = label_map.get(prediction, str(prediction))

            # Draw label on the "Spectator" window
            cv2.putText(frame, f"ZOOM FACE: {label}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    # Show what the AI is seeing
    # Resize the preview so it doesn't take up the whole screen
    preview = cv2.resize(frame, (800, 450))
    cv2.imshow('AI Spectator (Watching Zoom)', preview)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()