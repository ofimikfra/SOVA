import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

_base = python.BaseOptions(model_asset_path="models/pose_landmarker_lite.task")
_options = vision.PoseLandmarkerOptions(
    base_options=_base,
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_poses=1,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.6,
)
_pose = vision.PoseLandmarker.create_from_options(_options)

__all__ = ["detectBodyAction"]

def _detect_looking_away(pose_landmarks) -> bool:
    if pose_landmarks is None:
        return False
    left_ear  = pose_landmarks[7]
    right_ear = pose_landmarks[8]
    avg_vis    = (left_ear.visibility + right_ear.visibility) / 2
    ear_x_diff = abs(left_ear.x - right_ear.x)
    return avg_vis < 0.45 or ear_x_diff < 0.04


# API

def detectBodyAction(frame) -> tuple:
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = _pose.detect(image)

    if not results.pose_landmarks:
        return "No Person In Frame", 1.0

    lm = results.pose_landmarks[0]  # first person

    if _detect_looking_away(lm):
        return "Looking Away", 0.80

    return "Person Present", 1.0