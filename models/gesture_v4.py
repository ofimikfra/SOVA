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
_hand_position_history: dict = {}

try:
    from ultralytics import YOLO
    _phone_detector = YOLO("models/yolov8n.pt")
    print("✅ YOLO phone detection enabled")
except Exception as e:
    print(f"⚠️ YOLO unavailable, phone detection disabled: {e}")

PHONE_CONFIDENCE_THRESHOLD = 0.55


# ── PHONE DETECTION ───────────────────────────────────────────────────────────

def detectPhone(frame) -> bool:
    if _phone_detector is None:
        return False

    try:
        results = _phone_detector(frame, verbose=False, conf=PHONE_CONFIDENCE_THRESHOLD)
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == 67 and float(box.conf[0]) >= PHONE_CONFIDENCE_THRESHOLD:
                    return True
    except Exception:
        pass

    return False


# ── LANDMARK HELPERS ──────────────────────────────────────────────────────────

def _calc_distance(p1, p2):
    return np.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


def _get_fingertips(lm):
    return {
        "thumb": lm[4],
        "index": lm[8],
        "middle": lm[12],
        "ring": lm[16],
        "pinky": lm[20],
    }


def _get_wrist(lm):
    return lm[0]


# ── HEART DETECTION ───────────────────────────────────────────────────────────

def _is_two_hand_heart(hands_landmarks):
    if len(hands_landmarks) != 2:
        return False, 0.0

    lm1 = hands_landmarks[0].landmark
    lm2 = hands_landmarks[1].landmark

    f1 = _get_fingertips(lm1)
    f2 = _get_fingertips(lm2)

    thumb_dist = _calc_distance(f1["thumb"], f2["thumb"])
    index_dist = _calc_distance(f1["index"], f2["index"])
    wrist_dist = _calc_distance(_get_wrist(lm1), _get_wrist(lm2))

    if thumb_dist < 0.08 and index_dist < 0.08 and wrist_dist > 0.15:
        confidence = 1 - (thumb_dist + index_dist) / 2
        return True, min(0.97, confidence)

    return False, 0.0


def _is_finger_heart(lm):
    f = _get_fingertips(lm)

    thumb_index_dist = _calc_distance(f["thumb"], f["index"])
    middle_dist = _calc_distance(f["middle"], f["thumb"])
    ring_dist = _calc_distance(f["ring"], f["thumb"])
    pinky_dist = _calc_distance(f["pinky"], f["thumb"])

    if thumb_index_dist < 0.05 and middle_dist > 0.09 and ring_dist > 0.09:
        confidence = 1 - thumb_index_dist
        return True, min(0.95, confidence)

    return False, 0.0


# ── OTHER HELPERS ─────────────────────────────────────────────────────────────

def _finger_dist(p1, p2):
    return _calc_distance(p1, p2)


def _is_finger_extended(tip, pip, mcp, wrist):
    return _finger_dist(wrist, tip) > _finger_dist(wrist, pip) * 1.12


def _is_thumb_extended(thumb_tip, thumb_ip, index_mcp, wrist):
    return _finger_dist(thumb_tip, index_mcp) > _finger_dist(thumb_ip, index_mcp) * 1.2


def _detect_wave(hand_id: int):
    history = _hand_position_history.get(hand_id)
    if history is None or len(history) < 10:
        return False

    x_diff = np.diff([pos[0] for pos in list(history)[-10:]])
    direction_changes = sum(
        1 for i in range(len(x_diff) - 1) if x_diff[i] * x_diff[i + 1] < 0
    )
    return direction_changes >= 3


# ── CLASSIFIER ────────────────────────────────────────────────────────────────

def _classify_hand(hand_landmarks, hand_id: int):
    lm = hand_landmarks.landmark

    wrist      = lm[0]
    thumb_tip  = lm[4];  thumb_ip  = lm[3];  thumb_mcp = lm[2]
    index_tip  = lm[8];  index_pip = lm[6];  index_mcp = lm[5]
    middle_tip = lm[12]; middle_pip= lm[10]; middle_mcp= lm[9]
    ring_tip   = lm[16]; ring_pip  = lm[14]; ring_mcp  = lm[13]
    pinky_tip  = lm[20]; pinky_pip = lm[18]; pinky_mcp = lm[17]

    # ✅ Finger Heart FIRST (priority over OK sign)
    is_fh, conf = _is_finger_heart(lm)
    if is_fh:
        return "Finger Heart 🫰", conf

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

    # Thumbs
    if thumb_ext and not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        vert = abs(thumb_tip.y - thumb_mcp.y)
        if vert > 0.10 and thumb_tip.y < thumb_mcp.y:
            return "Thumbs Up", 0.90
        if vert > 0.10 and thumb_tip.y > thumb_mcp.y:
            return "Thumbs Down", 0.90

    # Pointing
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        if _finger_dist(index_tip, index_mcp) > 0.15:
            return "Pointing", 0.92

    # Peace
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        if _finger_dist(index_tip, middle_tip) > 0.10:
            return "Peace Sign", 0.94

    # OK Sign
    if _finger_dist(thumb_tip, index_tip) < 0.04 and middle_ext and ring_ext:
        return "OK Sign", 0.90

    # Wave / Raised
    if num_ext >= 4:
        if _detect_wave(hand_id):
            return "Waving", 0.93
        if wrist.y < 0.4:
            return "Hand Raised", 0.89

    return "Unknown", 0.0


# ── API ───────────────────────────────────────────────────────────────────────

CONFIDENCE_THRESHOLD = 0.88


def detectGesture(frame):

    # 1. Phone
    if detectPhone(frame):
        return "Using Phone", 0.95

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _hands.process(rgb)

    if not results.multi_hand_landmarks:
        return "No Gesture", 1.0

    # 2. Two-hand heart (global)
    if len(results.multi_hand_landmarks) == 2:
        is_heart, conf = _is_two_hand_heart(results.multi_hand_landmarks)
        if is_heart:
            return "Heart ❤️", conf

    # 3. Single-hand gestures
    best_gesture, best_conf = "No Gesture", 0.0

    for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
        gesture, conf = _classify_hand(hand_landmarks, idx)
        if conf >= CONFIDENCE_THRESHOLD and conf > best_conf:
            best_gesture, best_conf = gesture, conf

    return best_gesture, best_conf