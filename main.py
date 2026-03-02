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


# ... [Keep your imports the same] ...

def run_system(callback=None, source="webcam"):
    # ... [Keep source selection logic the same] ...

    # State variables
    display_expr = "Neutral"
    display_gest = "No Gesture"
    display_act = "Person Center"

    # 1. ADD THIS: Track if dashboard should be shown
    dashboard_visible = True

    print("[SOVA] Engine Active. Press 'q' on the video window to stop.")

    while True:
        frame = get_frame()
        if frame is None: continue

        if mirror:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # 1. FACIAL EXPRESSIONS
        results = face_mesh.detect(image)
        raw_expr, expr_conf = "Neutral", 1.0
        if results.face_landmarks:
            for face_landmarks in results.face_landmarks:
                raw_expr, expr_conf = detectExpression(face_landmarks, h, w)
                # ... [Keep face drawing code] ...

        # 2. GESTURES & BODY ACTIONS
        raw_gest, gest_conf = detectGesture(frame)
        raw_action, action_conf = detectBodyAction(frame)

        # 3. FEED THE PROCESSOR
        processExpression(raw_expr, expr_conf)
        processGesture(raw_gest, gest_conf)
        processBodyAction(raw_action, action_conf)

        # 4. THE FLUSH (Triggered every 5 seconds)
        stable_results = flushAll()
        if stable_results:
            expr, gest, act = stable_results

            # 2. ADD THIS: Check for the Toggle Gesture
            # If the stable gesture is a Thumb Up, flip the visibility state
            if gest == "Thumb_Up":
                dashboard_visible = not dashboard_visible
                print(f"[SOVA] Gesture Toggle: Dashboard is now {'VISIBLE' if dashboard_visible else 'HIDDEN'}")

            # 3. UPDATE THE CALLBACK: Pass 'dashboard_visible' to app.py
            if callback:
                # We update the callback signature to include the visibility state
                callback(expr, gest, act, f"Detected {expr}", toggle_ui=dashboard_visible)

            # Run TTS
            threading.Thread(target=speak, args=(f"Detected {expr}",), daemon=True).start()

            # Update display strings
            display_expr, display_gest, display_act = expr, gest, act

        # 5. VISUAL OVERLAY (Keep this the same)
        overlay_lines = [
            f"Expression: {display_expr}",
            f"Gesture:    {display_gest}",
            f"Action:     {display_act}",
            f"Dashboard:  {'ON' if dashboard_visible else 'OFF'}"  # Added for feedback
        ]

        for i, line in enumerate(overlay_lines):
            cv2.putText(frame, line, (20, h - 30 - (i * 35)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show the processed feed
        cv2.imshow("SOVA - Assistance Engine", frame)

        # 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_system(source="webcam")