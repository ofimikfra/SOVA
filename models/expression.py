import os
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
BaseOptions = python.BaseOptions

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, 'face_landmarker.task')

if not os.path.exists(MODEL_PATH):
    print(f"CRITICAL ERROR: Could not find model at {MODEL_PATH}")
else:
    print(f"SUCCESS: Model found at {MODEL_PATH}")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    running_mode=mp.tasks.vision.RunningMode.IMAGE
)
base_options = python.BaseOptions(model_asset_path="models/face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=5,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
face_mesh = FaceLandmarker.create_from_options(options)

# landmark indices

LEFT_EYE   = [33, 160, 158, 133, 153, 144]
RIGHT_EYE  = [362, 385, 387, 263, 373, 380]
UPPER_LIP  = 13
LOWER_LIP  = 14
LEFT_MOUTH = 61
RIGHT_MOUTH= 291
LEFT_BROW  = 70
RIGHT_BROW = 300


# confidence threshold

CONFIDENCE_THRESHOLD = 0.60

THRESHOLDS = {
    "mouth_open":       0.30,   # mouth_ratio minimum
    "smiling":          0.40,   # normalized_mouth_width minimum
    "frowning":         4.0,    # frown_depth minimum (pixels)
    "eyebrows_raised":  1.2,    # ratio above baseline
    "left_wink":        0.6,    # eye ratio below this = closed
    "right_wink":       0.6,
    "wink_open_eye":    0.85,   # other eye must be above this
}

# baselines

baseline_brow = None
baseline_eye  = None



def dist(a, b):
    return np.linalg.norm(a - b)


def _clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def _scale(value, minimum, maximum):
    if maximum <= minimum:
        return 0.0
    return _clamp((value - minimum) / (maximum - minimum))



def detectExpression(landmarks, h, w) -> tuple:

    global baseline_brow, baseline_eye

    def get(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])

    # Mouth measurements
    mouth_open  = dist(get(UPPER_LIP), get(LOWER_LIP))
    mouth_width = dist(get(LEFT_MOUTH), get(RIGHT_MOUTH))
    mouth_ratio = mouth_open / mouth_width if mouth_width > 0 else 0

    face_width            = dist(get(234), get(454))
    normalized_mouth_width = mouth_width / face_width if face_width > 0 else 0

    # Eyebrow measurements
    left_brow_dist  = dist(get(LEFT_BROW),  get(160))
    right_brow_dist = dist(get(RIGHT_BROW), get(385))
    avg_brow_dist   = (left_brow_dist + right_brow_dist) / 2

    if baseline_brow is None:
        baseline_brow = avg_brow_dist

    # Eye openness
    def eye_openness(indices):
        return dist(get(indices[1]), get(indices[5]))

    left_eye_open  = eye_openness(LEFT_EYE)
    right_eye_open = eye_openness(RIGHT_EYE)

    if baseline_eye is None:
        baseline_eye = (left_eye_open + right_eye_open) / 2

    left_ratio  = left_eye_open  / baseline_eye if baseline_eye > 0 else 1.0
    right_ratio = right_eye_open / baseline_eye if baseline_eye > 0 else 1.0

    # Frown
    corner_avg_y = (get(LEFT_MOUTH)[1] + get(RIGHT_MOUTH)[1]) / 2
    lip_y        = get(UPPER_LIP)[1]
    frown_depth  = corner_avg_y - lip_y

    # Brow raise ratio
    brow_ratio = avg_brow_dist / baseline_brow if baseline_brow > 0 else 1.0


    T = THRESHOLDS  

    # Left Wink
    if left_ratio < T["left_wink"] and right_ratio > T["wink_open_eye"]:
        # closed-eye score: how far below threshold (more closed = more confident)
        closed_score = _scale(T["left_wink"] - left_ratio,  0, T["left_wink"])
        open_score   = _scale(right_ratio - T["wink_open_eye"], 0, 0.5)
        confidence   = _clamp(0.5 + 0.3 * closed_score + 0.2 * open_score)
        if confidence >= CONFIDENCE_THRESHOLD:
            return "Left Wink", round(confidence, 2)

    # Right Wink
    if right_ratio < T["right_wink"] and left_ratio > T["wink_open_eye"]:
        closed_score = _scale(T["right_wink"] - right_ratio, 0, T["right_wink"])
        open_score   = _scale(left_ratio - T["wink_open_eye"], 0, 0.5)
        confidence   = _clamp(0.5 + 0.3 * closed_score + 0.2 * open_score)
        if confidence >= CONFIDENCE_THRESHOLD:
            return "Right Wink", round(confidence, 2)

    # Mouth Open
    if mouth_ratio > T["mouth_open"]:
        # scales from threshold (0.30) up to a wide-open mouth (~0.70)
        confidence = _clamp(0.55 + 0.45 * _scale(mouth_ratio, T["mouth_open"], 0.70))
        if confidence >= CONFIDENCE_THRESHOLD:
            return "Mouth Open", round(confidence, 2)

    # Smiling
    frowning = frown_depth > T["frowning"]
    if normalized_mouth_width > T["smiling"] and not frowning:
        confidence = _clamp(0.55 + 0.45 * _scale(normalized_mouth_width, T["smiling"], 0.60))
        if confidence >= CONFIDENCE_THRESHOLD:
            return "Smiling", round(confidence, 2)

    # Frowning
    if frown_depth > T["frowning"]:
        confidence = _clamp(0.55 + 0.45 * _scale(frown_depth, T["frowning"], 20.0))
        if confidence >= CONFIDENCE_THRESHOLD:
            return "Frowning", round(confidence, 2)

    # Eyebrows Raised
    if brow_ratio > T["eyebrows_raised"]:
        confidence = _clamp(0.55 + 0.45 * _scale(brow_ratio, T["eyebrows_raised"], 1.6))
        if confidence >= CONFIDENCE_THRESHOLD:
            return "Eyebrows Raised", round(confidence, 2)

    # Neutral
    return "Neutral", 1.0