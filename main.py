import cv2
import mediapipe as mp
import threading
import time
from src.screen_capture import getScreenFrame
from src.webcam_capture import getCameraFrame
from models.expression import detectExpression, face_mesh
from models.gesture_v4 import detectGesture
from models.body_action import detectBodyAction
from src.processor import processExpression, processGesture, processBodyAction, flushAll
from src.tts_engine import speak

# ── Global toggle — disable your own face detection during Meet ──
detect_self = True

def set_detect_self(value: bool):
    global detect_self
    detect_self = value
    print(f"[SOVA] Self-detection {'enabled' if value else 'disabled'}")

def speak_description(text):
    speak(text)


# ── Screen thread — Google Meet participants ──
def run_screen(stop_event, callback, headless):
    print("[SOVA] Screen thread started (Google Meet)")

    while not (stop_event and stop_event.is_set()):
        frame = getScreenFrame()
        if frame is None:
            continue

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Expressions
        results = face_mesh.detect(image)
        raw_expr, expr_conf = "Neutral", 1.0
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                raw_expr, expr_conf = detectExpression(face_landmarks, h, w)

        # Gestures & body
        raw_gest, gest_conf     = detectGesture(frame)
        raw_action, action_conf = detectBodyAction(frame)

        # Feed processor buffers
        processExpression(raw_expr, expr_conf)
        processGesture(raw_gest, gest_conf)
        processBodyAction(raw_action, action_conf)

        # Flush every N seconds (controlled by processor.INTERVAL)
        stable = flushAll()
        if stable:
            expr, gest, act, sentiment, sent_conf, description = stable

            # Speak the description via TTS
            threading.Thread(target=speak, args=(description,), daemon=True).start()

            # Send to dashboard
            if callback:
                callback(expr, gest, act, sentiment, sent_conf, description)

            print(f"📢 {description}")


# ── Webcam thread — YOUR own face (optional) ──
def run_webcam(stop_event, callback, headless):
    print("[SOVA] Webcam thread started (your face)")

    while not (stop_event and stop_event.is_set()):
        if not detect_self:
            time.sleep(0.5)
            continue

        frame = getCameraFrame()
        if frame is None:
            continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        results = face_mesh.detect(image)
        raw_expr, expr_conf = "Neutral", 1.0
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                raw_expr, expr_conf = detectExpression(face_landmarks, h, w)

        raw_gest, gest_conf     = detectGesture(frame)
        raw_action, action_conf = detectBodyAction(frame)

        processExpression(raw_expr, expr_conf)
        processGesture(raw_gest, gest_conf)
        processBodyAction(raw_action, action_conf)

        stable = flushAll()
        if stable:
            expr, gest, act, sentiment, sent_conf, description = stable
            threading.Thread(target=speak, args=(description,), daemon=True).start()
            if callback:
                callback(expr, gest, act, sentiment, sent_conf, description)
            print(f"📢 {description}")


# ── Main entry ────────────────────────────────
def run_system(callback=None, source="screen", headless=False, stop_event=None):
    print("[SOVA] Engine Active.")

    if source == "webcam":
        run_webcam(stop_event, callback, headless)

    elif source == "screen":
        run_screen(stop_event, callback, headless)

    elif source == "both":
        t1 = threading.Thread(target=run_webcam, args=(stop_event, callback, headless), daemon=True)
        t2 = threading.Thread(target=run_screen, args=(stop_event, callback, headless), daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_system(source="screen", headless=False)
