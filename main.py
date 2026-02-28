import cv2
import mediapipe as mp
import threading
import numpy as np

# Importing your custom modules
from src.screen_capture import getScreenFrame
from src.webcam_capture import getCameraFrame
from models.expression import detectExpression, face_mesh
from models.gesture_v4 import detectGesture
from models.body_action import detectBodyAction
from src.processor import processExpression, processGesture, processBodyAction, flushAll
from src.tts_engine import speak


def run_system(source="webcam"):
    """
    The core engine. 'source' can be 'webcam' or 'screen'.
    Triggered by your Flask server via the Chrome Extension.
    """

    # Selection logic based on the Extension's request
    if source == "screen":
        get_frame = getScreenFrame
        mirror = False
        print("[SOVA] Monitoring Screen...")
    else:
        get_frame = getCameraFrame
        mirror = True
        print("[SOVA] Monitoring Webcam...")

    # State variables for the on-screen display
    display_expr = "Neutral"
    display_gest = "No Gesture"
    display_act = "Person Center"

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

                # Draw Box around face
                xs = [lm.x * w for lm in face_landmarks]
                ys = [lm.y * h for lm in face_landmarks]
                cv2.rectangle(frame, (int(min(xs)), int(min(ys))), (int(max(xs)), int(max(ys))), (0, 255, 0), 2)

        # 2. GESTURES & BODY ACTIONS
        raw_gest, gest_conf = detectGesture(frame)
        raw_action, action_conf = detectBodyAction(frame)

        # 3. FEED THE PROCESSOR (Buffer)
        processExpression(raw_expr, expr_conf)
        processGesture(raw_gest, gest_conf)
        processBodyAction(raw_action, action_conf)

        # 4. THE FLUSH (Triggered every 5 seconds)
        stable_results = flushAll()
        if stable_results:
            display_expr, display_gest, display_act = stable_results

            # VOICE FEEDBACK: Run in a separate thread so the camera doesn't lag
            text_to_say = f"Status update: {display_expr}. {display_gest}."
            threading.Thread(target=speak, args=(text_to_say,), daemon=True).start()

        # 5. VISUAL OVERLAY
        overlay_lines = [
            f"Expression: {display_expr}",
            f"Gesture:    {display_gest}",
            f"Action:     {display_act}",
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
    # Default behavior if run manually
    run_system(source="webcam")