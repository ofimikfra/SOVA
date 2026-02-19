import cv2
from src.screen_capture import getScreenFrame
from src.expressions import detectExpression, face_mesh
import mediapipe as mp

previous_expression = None  # track previous expression to detect changes

# ----------------- Webcam setup ----------------- #
cap = cv2.VideoCapture(0)  # default webcam
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Flip image horizontally for correct left/right
def flip_frame(frame):
    return cv2.flip(frame, 1)

# ----------------- Main loop ----------------- #
while True:

    frame = getScreenFrame() # capture frame from screen

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # bgr to rbg color conversion
    
    # convert frame to MediaImage format for new API
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = face_mesh.detect(image) # process frame to detect landmarks

    if results.face_landmarks:

        h, w, _ = frame.shape # get frame dimensions

        for face_landmarks in results.face_landmarks: # for each face detected

            expr = detectExpression(face_landmarks, h, w) # detect expression

            # only print when expression changes
            if expr != previous_expression:
                print("Detected Expression:", expr)
                previous_expression = expr
    
            # draw rectangle around face
            x_coords = [lm.x * w for lm in face_landmarks]
            y_coords = [lm.y * h for lm in face_landmarks]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
            # display expression on frame
            text_position = (x_min, y_max + 25)
            cv2.putText(
                frame,
                expr,
                (x_min, y_max + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

    # show frame with detected faces & expressions
    cv2.imshow("Expression Detection", frame)

    # q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ----------------- Cleanup ----------------- #
cap.release()
cv2.destroyAllWindows()
