"""
SOVA Gesture Recognition - Module
Detects hand gestures from a BGR frame.
Intended to be imported and called from main.py.
"""

import cv2
import mediapipe as mp
import numpy as np
from collections import deque

__all__ = ["detectGesture"]

_mp_hands = mp.solutions.hands
_hands = _mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=4,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.6
)

_phone_detector = None
_hand_position_history: dict = {} # Per-hand position history used by wave detection

try:
    from ultralytics import YOLO
    _phone_detector = YOLO("models/yolov8n.pt")
    print("✅ YOLO phone detection enabled")
except Exception as e:
    print(f"⚠️  YOLO unavailable, phone detection disabled: {e}")

PHONE_CONFIDENCE_THRESHOLD = 0.55  # minimum YOLO confidence to report phone


# phone detection

def detectPhone(frame) -> bool:
    """
    Return True if a phone is detected in frame (BGR numpy array) with
    sufficient confidence using YOLO. Returns False if YOLO is unavailable.
    """
    if _phone_detector is None:
        return False

    try:
        results = _phone_detector(frame, verbose=False, conf=PHONE_CONFIDENCE_THRESHOLD)
        for result in results:
            for box in result.boxes:
                # COCO class 67 = cell phone
                if int(box.cls[0]) == 67 and float(box.conf[0]) >= PHONE_CONFIDENCE_THRESHOLD:
                    return True
    except Exception:
        pass

    return False



# helper functions

def _finger_dist(p1, p2) -> float:
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def _is_finger_extended(tip, pip, mcp, wrist) -> bool:
    return _finger_dist(wrist, tip) > _finger_dist(wrist, pip) * 1.12


def _is_thumb_extended(thumb_tip, thumb_ip, index_mcp, wrist) -> bool:
    return _finger_dist(thumb_tip, index_mcp) > _finger_dist(thumb_ip, index_mcp) * 1.2


def _detect_wave(hand_id: int) -> bool:
    history = _hand_position_history.get(hand_id)
    if history is None or len(history) < 10:
        return False
    x_diff = np.diff([pos[0] for pos in list(history)[-10:]])
    direction_changes = sum(
        1 for i in range(len(x_diff) - 1) if x_diff[i] * x_diff[i + 1] < 0
    )
    return direction_changes >= 3


def _classify_hand(hand_landmarks, hand_id: int) -> tuple:
    # Return (gesture_label, confidence) for a single hand
    lm = hand_landmarks.landmark

    wrist      = lm[0]
    thumb_tip  = lm[4];  thumb_ip  = lm[3];  thumb_mcp = lm[2]
    index_tip  = lm[8];  index_pip = lm[6];  index_mcp = lm[5]
    middle_tip = lm[12]; middle_pip= lm[10]; middle_mcp= lm[9]
    ring_tip   = lm[16]; ring_pip  = lm[14]; ring_mcp  = lm[13]
    pinky_tip  = lm[20]; pinky_pip = lm[18]; pinky_mcp = lm[17]

    thumb_ext  = _is_thumb_extended(thumb_tip, thumb_ip, index_mcp, wrist)
    index_ext  = _is_finger_extended(index_tip,  index_pip,  index_mcp,  wrist)
    middle_ext = _is_finger_extended(middle_tip, middle_pip, middle_mcp, wrist)
    ring_ext   = _is_finger_extended(ring_tip,   ring_pip,   ring_mcp,   wrist)
    pinky_ext  = _is_finger_extended(pinky_tip,  pinky_pip,  pinky_mcp,  wrist)

    num_ext = sum([thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext])

    # Update wave history
    if hand_id not in _hand_position_history:
        _hand_position_history[hand_id] = deque(maxlen=12)
    _hand_position_history[hand_id].append((wrist.x, wrist.y))

    # Thumbs Up / Down (same finger config, different direction)
    if thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        vert = abs(thumb_tip.y - thumb_mcp.y)
        if vert > 0.10 and thumb_tip.y < thumb_mcp.y:
            return "Thumbs Up", 0.90
        if vert > 0.10 and thumb_tip.y > thumb_mcp.y:
            return "Thumbs Down", 0.90

    # Pointing
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        index_len = _finger_dist(index_tip, index_mcp)
        if index_len > 0.15:
            if not thumb_ext or index_len > _finger_dist(thumb_tip, thumb_mcp) * 1.2:
                return "Pointing", 0.92

    # Peace Sign
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        spread = _finger_dist(index_tip, middle_tip)
        if spread > 0.10:
            if _finger_dist(index_tip, index_mcp) > 0.13 and \
               _finger_dist(middle_tip, middle_mcp) > 0.13:
                return "Peace Sign", 0.94

    # OK Sign
    if _finger_dist(thumb_tip, index_tip) < 0.04 and middle_ext and ring_ext:
        return "OK Sign", 0.90

    # Waving / Hand Raised
    if num_ext >= 4:
        if _detect_wave(hand_id):
            return "Waving", 0.93
        if wrist.y < 0.4:
            return "Hand Raised", 0.89

    return "Unknown", 0.0


# API

CONFIDENCE_THRESHOLD = 0.88


def detectGesture(frame) -> tuple:

    if detectPhone(frame):
        return "Using Phone", 0.95

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _hands.process(rgb)

    if not results.multi_hand_landmarks:
        return "No Gesture", 1.0

    best_gesture, best_conf = "No Gesture", 0.0

    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        gesture, conf = _classify_hand(hand_landmarks, idx)
        if conf >= CONFIDENCE_THRESHOLD and conf > best_conf:
            best_gesture, best_conf = gesture, conf

    return best_gesture, best_conf