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

def eyeAspectRatio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

def detectExpression(landmarks, h, w):

    def get(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])

    left_eye = np.array([get(i) for i in LEFT_EYE])
    right_eye = np.array([get(i) for i in RIGHT_EYE])

    ear_left = eyeAspectRatio(left_eye)
    ear_right = eyeAspectRatio(right_eye)
    ear = (ear_left + ear_right) / 2

    mouth_open = np.linalg.norm(get(UPPER_LIP) - get(LOWER_LIP))
    mouth_width = np.linalg.norm(get(LEFT_MOUTH) - get(RIGHT_MOUTH))
    mouth_ratio = mouth_open / mouth_width

    brow_left = get(LEFT_BROW)
    brow_right = get(RIGHT_BROW)
    eye_left_center = left_eye.mean(axis=0)
    eye_right_center = right_eye.mean(axis=0)

    brow_dist = (
        np.linalg.norm(brow_left - eye_left_center) +
        np.linalg.norm(brow_right - eye_right_center)
    ) / 2

    expression = "Neutral"

    if mouth_ratio > 0.3:
        expression = "Mouth Open"

    elif mouth_width > 80:
        expression = "Smiling"

    # frowning

    # eyebrows raised

    # head tilt

    # wink?

    return expression
