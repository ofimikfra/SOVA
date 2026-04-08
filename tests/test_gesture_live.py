"""
test_gesture_live.py — Live gesture test with landmark overlay
---------------------------------------------------------------
Run from the project root:
    python tests/test_gesture_live.py

Shows:
  • Full hand skeleton (landmarks + connections)
  • Per-finger extension indicators (coloured dots on fingertips)
  • Detected gesture label + confidence — updated EVERY frame
  • FPS counter
  • Wave history bar (fills as wrist oscillates)

Press  Q  to quit.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
import time

# ── Import the module under test ──────────────────────────────────────────────
from models.gesture_v4 import _hands, _mp_hands, _classify_hand, _hand_position_history
from models.gesture_v4 import detectGesture, _is_finger_extended, _is_thumb_extended

# ── Colour palette (BGR) ──────────────────────────────────────────────────────
C_BG_PANEL  = (18,  40,  45)
C_BONE      = (100, 200, 220)
C_JOINT     = (60,  150, 170)
C_TIP_ON    = (50,  230, 120)   # finger extended
C_TIP_OFF   = (60,   60, 100)   # finger curled
C_THUMB_ON  = (50,  200, 255)
C_THUMB_OFF = (60,   60, 100)
C_TEXT      = (253, 253, 253)
C_MUTED     = (140, 140, 140)
C_ACCENT    = (32,  150, 175)
C_WARN      = (87,  126, 255)
C_POS       = (83,  200,   0)

# Finger tip / pip / mcp landmark indices  [thumb, index, middle, ring, pinky]
TIPS  = [4,  8,  12, 16, 20]
PIPS  = [3,  6,  10, 14, 18]
MCPS  = [2,  5,   9, 13, 17]
NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

mp_draw   = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


def _draw_skeleton(frame, hand_landmarks, h, w):
    """Draw connections then joints on top."""
    mp_draw.draw_landmarks(
        frame,
        hand_landmarks,
        _mp_hands.HAND_CONNECTIONS,
        mp_draw.DrawingSpec(color=C_BONE, thickness=2, circle_radius=0),
        mp_draw.DrawingSpec(color=C_JOINT, thickness=2, circle_radius=3),
    )


def _draw_finger_dots(frame, lm, h, w):
    """Colour-coded dot on each fingertip: green = extended, dim = curled."""
    wrist = lm[0]

    for i, (tip_idx, pip_idx, mcp_idx) in enumerate(zip(TIPS, PIPS, MCPS)):
        tip = lm[tip_idx]
        pip = lm[pip_idx]
        mcp = lm[mcp_idx]

        if i == 0:  # thumb
            ext = _is_thumb_extended(tip, pip, lm[5], wrist)
            col = C_THUMB_ON if ext else C_THUMB_OFF
        else:
            ext = _is_finger_extended(tip, pip, mcp, wrist)
            col = C_TIP_ON if ext else C_TIP_OFF

        cx = int(tip.x * w)
        cy = int(tip.y * h)
        cv2.circle(frame, (cx, cy), 9, col, -1)
        cv2.circle(frame, (cx, cy), 9, C_TEXT, 1)


def _draw_hud(frame, gesture, conf, fps, h, w, num_hands):
    """Semi-transparent HUD panel in the top-left corner."""
    panel_h = 140
    panel_w = 340
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), C_BG_PANEL, -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Gesture label
    conf_pct = int(conf * 100)
    label    = gesture if gesture != "No Gesture" else "—"
    color    = C_POS if conf >= 0.88 else C_WARN if conf > 0 else C_MUTED
    cv2.putText(frame, label, (16, 44),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, color, 2, cv2.LINE_AA)

    # Confidence bar
    bar_x, bar_y, bar_w_max, bar_h2 = 16, 58, panel_w - 32, 6
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_max, bar_y + bar_h2),
                  (50, 50, 60), -1)
    filled = int(bar_w_max * conf)
    if filled > 0:
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h2),
                      color, -1)
    cv2.putText(frame, f"{conf_pct}%", (bar_x + bar_w_max + 6, bar_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_MUTED, 1, cv2.LINE_AA)

    # Finger labels
    finger_labels = ["T", "I", "M", "R", "P"]
    for fi, fl in enumerate(finger_labels):
        bx = 16 + fi * 42
        cv2.putText(frame, fl, (bx + 2, 94),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C_MUTED, 1, cv2.LINE_AA)

    # Stats row
    cv2.putText(frame,
                f"Hands: {num_hands}   FPS: {fps:.0f}",
                (16, 124),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_MUTED, 1, cv2.LINE_AA)

    # Quit hint bottom right
    cv2.putText(frame, "Q  quit", (w - 80, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C_MUTED, 1, cv2.LINE_AA)


def _draw_wave_bar(frame, hand_id, h, w, slot=0):
    """Mini waveform showing wrist oscillation — used for wave gesture detection."""
    history = _hand_position_history.get(hand_id)
    if not history or len(history) < 2:
        return

    xs      = [pos[0] for pos in list(history)]
    bar_len = 120
    bar_x   = w - bar_len - 16
    bar_y   = 16 + slot * 30

    # Background track
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_len, bar_y + 10),
                  (40, 40, 50), -1)

    # Plot normalised x positions as a mini waveform
    x_min, x_max = min(xs), max(xs)
    rng = x_max - x_min if x_max > x_min else 1e-5
    pts = []
    for i, xv in enumerate(xs[-bar_len:]):
        px = bar_x + int(i * bar_len / max(len(xs), 1))
        py = bar_y + 5 - int(((xv - x_min) / rng) * 4)
        pts.append((px, py))

    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], C_ACCENT, 1)

    # Direction-change count
    x_diff  = np.diff(xs[-10:]) if len(xs) >= 10 else np.array([])
    changes = sum(
        1 for i in range(len(x_diff) - 1) if x_diff[i] * x_diff[i + 1] < 0
    ) if len(x_diff) > 1 else 0
    wave_label = f"wave {changes}/3" if changes < 3 else "WAVE ✓"
    wc = C_POS if changes >= 3 else C_MUTED
    cv2.putText(frame, wave_label, (bar_x - 62, bar_y + 9),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, wc, 1, cv2.LINE_AA)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam (index 0).")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("[TEST] Gesture live test running — press Q to quit.")

    prev_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # mirror so it feels natural
        h, w  = frame.shape[:2]

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _hands.process(rgb)

        gesture, conf = "No Gesture", 0.0

        if results.multi_hand_landmarks:
            best_gesture, best_conf = "No Gesture", 0.0

            for idx, hand_lm in enumerate(results.multi_hand_landmarks):
                lm = hand_lm.landmark

                # Skeleton + fingertip dots
                _draw_skeleton(frame, hand_lm, h, w)
                _draw_finger_dots(frame, lm, h, w)

                # Wave bar (top-right, one row per hand)
                _draw_wave_bar(frame, idx, h, w, slot=idx)

                # Per-hand label above the wrist
                g, c = _classify_hand(hand_lm, idx)
                wrist_x = int(lm[0].x * w)
                wrist_y = int(lm[0].y * h) + 24
                hand_col = C_POS if c >= 0.88 else C_WARN if c > 0 else C_MUTED
                cv2.putText(frame, f"{g} ({int(c*100)}%)",
                            (wrist_x - 60, wrist_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, hand_col, 2, cv2.LINE_AA)

                if c >= 0.88 and c > best_conf:
                    best_gesture, best_conf = g, c

            gesture, conf = best_gesture, best_conf

        # FPS
        now    = time.time()
        fps    = 1.0 / max(now - prev_t, 1e-5)
        prev_t = now

        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
        _draw_hud(frame, gesture, conf, fps, h, w, num_hands)

        cv2.imshow("SOVA — Gesture Test (Q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[TEST] Done.")


if __name__ == "__main__":
    main()