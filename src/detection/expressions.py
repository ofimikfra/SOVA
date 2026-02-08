import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# init mediapipe face landmarker solution
base_options = python.BaseOptions(model_asset_path="src/detection/models/face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=5,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
face_mesh = vision.FaceLandmarker.create_from_options(options)

# landmarks for facial features
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
LEFT_BROW = 70
RIGHT_BROW = 300

# calculate euclidean distance
def dist(a, b):
    return np.linalg.norm(a - b)

# detect expressions based on facial landmark distances
def detectExpression(landmarks, h, w):

    # landmark to pixel coords
    def get(i):
        return np.array([landmarks[i].x * w, landmarks[i].y * h])

    # mouth measurements
    mouth_open = dist(get(UPPER_LIP), get(LOWER_LIP))
    mouth_width = dist(get(LEFT_MOUTH), get(RIGHT_MOUTH))
    mouth_ratio = mouth_open / mouth_width

    # normalize mouth width relative to face width
    face_left = get(234)
    face_right = get(454)
    face_width = dist(face_left, face_right)
    normalized_mouth_width = mouth_width / face_width

    # eye & eyebrow measurements
    left_brow = get(LEFT_BROW) 
    right_brow = get(RIGHT_BROW) 
    left_eye_top = get(160)
    right_eye_top = get(385)

    # distance between eyebrows & eyes
    left_brow_dist = dist(left_brow, left_eye_top)
    right_brow_dist = dist(right_brow, right_eye_top)
    avg_brow_dist = (left_brow_dist + right_brow_dist) / 2

    # baseline eyebrow distance on first frame
    if 'baseline_brow' not in globals():
        global baseline_brow
        baseline_brow = avg_brow_dist

    # eye openness measurements
    def eye_openness(indices):
        return dist(get(indices[1]), get(indices[5])) # vertical eye opening using landmark indices

    left_eye_open = eye_openness(LEFT_EYE)
    right_eye_open = eye_openness(RIGHT_EYE)

    # set baseline eye openness on first frame
    if 'baseline_eye' not in globals():
        global baseline_eye
        baseline_eye = (left_eye_open + right_eye_open) / 2

    # detect if eyebrows are raised compared to baseline
    eyebrows_raised = avg_brow_dist / baseline_brow > 1.2

    # eye openness ratios relative to baseline
    left_ratio = left_eye_open / baseline_eye
    right_ratio = right_eye_open / baseline_eye

    # detect winks
    left_wink = left_ratio < 0.6 and right_ratio > 0.85
    right_wink = right_ratio < 0.6 and left_ratio > 0.85

    # frown detection
    left_corner = get(LEFT_MOUTH) 
    right_corner = get(RIGHT_MOUTH)  
    upper_lip = get(UPPER_LIP)  
    corner_avg_y = (left_corner[1] + right_corner[1]) / 2 
    lip_y = upper_lip[1] 
    frown_depth = corner_avg_y - lip_y 
    frowning = frown_depth > 4

    # smile detection
    smiling = normalized_mouth_width > 0.4 and not frowning

    # mouth open detection
    mouth_opened = mouth_ratio > 0.3

    

    # neutral expression by default
    expression = "Neutral"

    # determine expression based on detected features (priority order)
    if left_wink:
        expression = "Left Wink"
    elif right_wink:
        expression = "Right Wink"
    elif mouth_opened:
        expression = "Mouth Open"
    elif smiling:
        expression = "Smiling"
    elif frowning:
        expression = "Frowning"
    elif eyebrows_raised:
        expression = "Eyebrows Raised"

    return expression

    '''
    issues:
    - expression detection affected by distance of face from camera
    - expressions detected differ for people with different facial features
    - some expressions get overruled by others, create hierarchy system?
    
    improvements:
    - set time interval for expression changes to prevent rapid switching
    '''