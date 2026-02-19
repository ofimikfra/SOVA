import cv2
from src.screen_capture import getScreenFrame
from src.webcam_capture import getCameraFrame
from src.expressions import detectExpression, face_mesh
from src.processor import processExpression
import mediapipe as mp

previous_expression = None  # track previous dominant expression to detect changes

# prompt user to choose capture source
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

while True:

    frame = get_frame()  # capture frame from selected source

    if mirror:
        frame = cv2.flip(frame, 1)  # flip horizontally for natural mirror view

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = face_mesh.detect(image)

    if results.face_landmarks:

        h, w, _ = frame.shape

        for face_landmarks in results.face_landmarks:

            expr = detectExpression(face_landmarks, h, w)

            # pass raw expression into processor; only returns a result every BUFFER_SIZE frames
            dominant = processExpression(expr)

            if dominant is not None and dominant != previous_expression:
                print("Dominant Expression:", dominant)
                previous_expression = dominant

            # use the last known dominant expression for the overlay,
            # falling back to the raw expression until the first window completes
            display_expr = previous_expression if previous_expression is not None else expr

            x_coords = [lm.x * w for lm in face_landmarks]
            y_coords = [lm.y * h for lm in face_landmarks]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            cv2.putText(
                frame,
                display_expr,
                (x_min, y_max + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

    cv2.imshow("Expression Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()