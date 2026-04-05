import cv2
import mediapipe as mp

__all__ = ["detectBodyAction"]

_mp_pose = mp.solutions.pose
_pose = _mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

def _detect_looking_away(pose_landmarks) -> bool:
    # True if both ears have low visibility (person turned away).
    if pose_landmarks is None:
        return False
    left_ear  = pose_landmarks.landmark[7]
    right_ear = pose_landmarks.landmark[8]
    avg_vis    = (left_ear.visibility + right_ear.visibility) / 2
    ear_x_diff = abs(left_ear.x - right_ear.x)
    return avg_vis < 0.45 or ear_x_diff < 0.04


# API

def detectBodyAction(frame) -> tuple:

    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _pose.process(rgb)
    lm      = results.pose_landmarks

    if lm is None:
        return "No Person In Frame"

    if _detect_looking_away(lm):
        return "Looking Away", 0.80
    
    return "Person present"