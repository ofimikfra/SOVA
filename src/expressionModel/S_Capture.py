import cv2
import numpy as np
import mediapipe as mp
import joblib
import mss

# 1. Load Model
model = joblib.load("expression_model.pkl")

# 2. Updated Labels
label_map = {
    0: "Neutral",
    1: "Smile",
    2: "Frown",
    3: "Left Wink",
    4: "Right Wink",
    5: "Eyebrows Raised"  # Ensure your model was trained with this 6th label!
}

# 3. MediaPipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# 4. Screen Capture Setup
sct = mss.mss()
monitor = sct.monitors[1]  # Main monitor


def get_dist(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


print("AI is watching your screen... Press ESC to stop.")

while True:
    # Capture the screen
    img = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            lms = face_landmarks.landmark

            # --- Feature Extraction ---
            # Eye Aspect Ratios (Winks)
            l_ear = get_dist(lms[159], lms[145]) / (get_dist(lms[33], lms[133]) + 1e-6)
            r_ear = get_dist(lms[386], lms[374]) / (get_dist(lms[362], lms[263]) + 1e-6)

            # Mouth Ratio (Smile/Frown)
            m_width = get_dist(lms[61], lms[291])
            m_height = get_dist(lms[13], lms[14])
            m_ratio = m_height / (m_width + 1e-6)

            # Eyebrow Raise Feature
            # Distance between top of eye (159) and inner eyebrow (107)
            l_brow_dist = get_dist(lms[159], lms[107])
            r_brow_dist = get_dist(lms[386], lms[336])
            avg_brow_dist = (l_brow_dist + r_brow_dist) / 2

            # --- PREDICTION VECTOR ---
            # IMPORTANT: The order of this list must match your Training Data exactly!
            features = np.array([l_ear, r_ear, m_ratio, avg_brow_dist]).reshape(1, -1)

            try:
                prediction = model.predict(features)[0]
                label = label_map.get(prediction, str(prediction))
            except:
                label = "Analyzing..."

            # Visual Feedback
            cv2.putText(frame, f"AI Detected: {label}", (100, 100),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

    # Show a smaller preview so it's not in the way
    preview = cv2.resize(frame, (960, 540))
    cv2.imshow('Zoom Expression Monitor', preview)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()