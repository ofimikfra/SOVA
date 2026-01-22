import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
LEFT_BROW = 70
RIGHT_BROW = 300

def eyeRatio(eye):
    A = np.linalg.norm(eye[1] - eye[0])
    return A

def detectExpression(landmarks, h, w):

    def get(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])

    mouth_open = np.linalg.norm(get(UPPER_LIP) - get(LOWER_LIP))
    mouth_width = np.linalg.norm(get(LEFT_MOUTH) - get(RIGHT_MOUTH))
    mouth_ratio = mouth_open / mouth_width

    face_left = get(234)   # approximate left face jaw landmark
    face_right = get(454)  # approximate right face jaw landmark
    face_width = np.linalg.norm(face_left - face_right)
    normalized_mouth_width = mouth_width / face_width

    # --------------------------- expression thresholds -------------------------- #

    # default
    expression = "Neutral"

    # mouth open
    if mouth_ratio > 0.3:
        expression = "Mouth Open"

    # smiling
    elif normalized_mouth_width > 0.4: # how to fix for people with different sized mouths?
        expression = "Smiling"

    # frowning

    # eyebrows raised

    # wink?

    return expression
