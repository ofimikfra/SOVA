import cv2
import mediapipe as mp

from src.screen_capture import getScreenFrame
from src.webcam_capture import getCameraFrame
from models.gesture_v4 import detectGesture
from models.body_action import detectBodyAction
from models.expression import detectExpression, face_mesh
from src.processor import processExpression, processGesture, processBodyAction, flushAll


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

    h, w, _ = frame.shape

    # Expression detection
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image   = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = face_mesh.detect(image)

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            raw_expr, expr_conf = detectExpression(face_landmarks, h, w)
            processExpression(raw_expr, expr_conf)

            # Draw face bounding box
            xs = [lm.x * w for lm in face_landmarks]
            ys = [lm.y * h for lm in face_landmarks]
            cv2.rectangle(frame,
                          (int(min(xs)), int(min(ys))),
                          (int(max(xs)), int(max(ys))),
                          (0, 255, 0), 2)
    else:
        processExpression("Neutral", 1.0)

    # Gesture detection
    raw_gest, gest_conf = detectGesture(frame)
    processGesture(raw_gest, gest_conf)

    # Body action detection
    raw_action, action_conf = detectBodyAction(frame)
    processBodyAction(raw_action, action_conf)

    # Flush all channels together every 30s
    flush_result = flushAll()
    if flush_result is not None:
        display_expression, display_gesture, display_action = flush_result

    # overlay
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

    cv2.imshow("SOVA", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()