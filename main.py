import cv2
import mediapipe as mp

from src.screen_capture import getScreenFrame
from src.webcam_capture import getCameraFrame
from models.expression import detectExpression, face_mesh
from models.gesture_v4 import detectGesture
from models.body_action import detectBodyAction
from src.processor import processExpression, processGesture, processBodyAction


print("Select capture source:")
print("  [1] Screen capture")
print("  [2] Webcam")

while True:
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        get_frame = getScreenFrame
        mirror = False
        print("Using screen capture.")
        break
    elif choice == "2":
        get_frame = getCameraFrame
        mirror = True
        print("Using webcam.")
        break
    else:
        print("Invalid choice. Please enter 1 or 2.")


display_expression = "Neutral"
display_gesture    = "No Gesture"
display_action     = "Person Center"


while True:
    frame = get_frame()

    if mirror:
        frame = cv2.flip(frame, 1)

    # expression detection
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = face_mesh.detect(image)

    h, w, _ = frame.shape

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            raw_expr, expr_conf = detectExpression(face_landmarks, h, w)  # unpack tuple
            dominant_expr = processExpression(raw_expr, expr_conf)        # pass confidence
            if dominant_expr is not None:
                if dominant_expr != display_expression:
                    print(f"Expression: {dominant_expr}")
                display_expression = dominant_expr

            xs = [lm.x * w for lm in face_landmarks]
            ys = [lm.y * h for lm in face_landmarks]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # gesture detection
    raw_gest = detectGesture(frame)
    dominant_gest = processGesture(raw_gest)
    if dominant_gest is not None:
        if dominant_gest != display_gesture:
            print(f"Gesture:    {dominant_gest}")
        display_gesture = dominant_gest

    # Body action detection
    raw_action      = detectBodyAction(frame)
    dominant_action = processBodyAction(raw_action)
    if dominant_action is not None:
        if dominant_action != display_action:
            print(f"Action:     {dominant_action}")
        display_action = dominant_action

    # overlay text
    overlay_lines = [
        f"Expression: {display_expression}",
        f"Gesture:    {display_gesture}",
        f"Action:     {display_action}",
    ]
    for i, line in enumerate(overlay_lines):
        cv2.putText(
            frame, line,
            (15, h - 20 - i * 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9, (0, 255, 0), 2, cv2.LINE_AA,
        )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()