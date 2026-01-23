import cv2
from expressionModel.expressions import detectExpression, face_mesh

#Webcam setup
cap = cv2.VideoCapture(0)  # default webcam
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# Flip image horizontally for correct left/right
def flip_frame(frame):
    return cv2.flip(frame, 1)

#Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = flip_frame(frame)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face in results.multi_face_landmarks:
            expr = detectExpression(face.landmark, h, w)

            # Get bounding box coordinates   x_coords = 
         [lm.x * w for lm in face.landmark]
            y_coords = [lm.y * h for lm in face.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            # Draw rectangle around face
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Put expression text below face box
            cv2.putText(
                frame,
                expr,
                (x_min, y_max + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

    cv2.imshow("Expression Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ----------------- Cleanup ----------------- #
cap.release()
cv2.destroyAllWindows()
