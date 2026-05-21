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

# Replace with:
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision

_hand_landmarker = _mp_vision.HandLandmarker.create_from_options(
    _mp_vision.HandLandmarkerOptions(
        base_options=_mp_python.BaseOptions(
            model_asset_path="models/hand_landmarker.task"
        ),
        num_hands=4,
        min_hand_detection_confidence=0.75,
        min_hand_presence_confidence=0.75,
        min_tracking_confidence=0.6,
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
    )
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
PHONE_CONFIRM_FRAMES  = 4
PHONE_RELEASE_FRAMES  = 6
_phone_consecutive    = 0
_no_phone_consecutive = 0
_phone_active         = False

# phone detection

def _get_hand_crop(frame, hand_landmarks, padding: float = 0.30):
    """
    Return a cropped BGR region tightly around a hand landmark cluster,
    expanded by `padding` percent on each side.
    Returns None if the crop would be degenerate.
    """
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in hand_landmarks]
    ys = [lm.y * h for lm in hand_landmarks]

    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)

    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding

    x1 = max(0,   int(x1 - pad_x))
    y1 = max(0,   int(y1 - pad_y))
    x2 = min(w,   int(x2 + pad_x))
    y2 = min(h,   int(y2 + pad_y))

    if x2 - x1 < 20 or y2 - y1 < 20:
        return None

    return frame[y1:y2, x1:x2]

def _phone_aspect_ok(box) -> bool:
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    ratio = w / h
    return 0.35 < ratio < 2.9

def detectPhone(frame, hand_results=None) -> bool:
    """
    Only runs YOLO on the region immediately around each detected hand.
    No hands visible → no phone detection at all.
    This prevents on-screen mic icons, desk mics, and background objects
    from ever triggering a false positive.
    """
    global _phone_consecutive, _no_phone_consecutive, _phone_active

    if _phone_detector is None:
        return False

    # No hands in frame — can't be using a phone
    if hand_results is None or not hand_results.hand_landmarks:
        _phone_consecutive     = 0
        _no_phone_consecutive += 1
        if _no_phone_consecutive >= PHONE_RELEASE_FRAMES:
            _phone_active = False
        return _phone_active

    detected_this_frame = False

    for hand_lm in hand_results.hand_landmarks:
        crop = _get_hand_crop(frame, hand_lm, padding=0.30)
        if crop is None:
            continue

        try:
            results = _phone_detector(crop, verbose=False, conf=PHONE_CONFIDENCE_THRESHOLD)
            for result in results:
                for box in result.boxes:
                    if int(box.cls[0]) != 67:
                        continue
                    if float(box.conf[0]) < PHONE_CONFIDENCE_THRESHOLD:
                        continue
                    if not _phone_aspect_ok(box):
                        continue
                    detected_this_frame = True
                    break
                if detected_this_frame:
                    break
        except Exception:
            pass

        if detected_this_frame:
            break

    if detected_this_frame:
        _phone_consecutive   += 1
        _no_phone_consecutive = 0
        if _phone_consecutive >= PHONE_CONFIRM_FRAMES:
            _phone_active = True
    else:
        _no_phone_consecutive += 1
        _phone_consecutive     = 0
        if _no_phone_consecutive >= PHONE_RELEASE_FRAMES:
            _phone_active = False

    return _phone_active



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
    lm = hand_landmarks

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
        if spread > 0.08:
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
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = _hand_landmarker.detect(mp_image)

    if detectPhone(frame, results):
        return "Using Phone", 0.95

    if not results.hand_landmarks:
        return "No Gesture", 1.0

    best_gesture, best_conf = "No Gesture", 0.0

    for idx, hand_landmarks in enumerate(results.hand_landmarks):
        gesture, conf = _classify_hand(hand_landmarks, idx)
        if conf >= CONFIDENCE_THRESHOLD and conf > best_conf:
            best_gesture, best_conf = gesture, conf

    return best_gesture, best_conf