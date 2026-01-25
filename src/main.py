import cv2
from screenCapture.capture import getScreenFrame
from expressionModel.expressions import detectExpression, face_mesh

while True:
    frame = getScreenFrame()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        h, w, _ = frame.shape

        for face in results.multi_face_landmarks:
            expr = detectExpression(face.landmark, h, w)
    
            # Get bounding box coordinates from landmarks
            x_coords = [lm.x * w for lm in face.landmark]
            y_coords = [lm.y * h for lm in face.landmark]
    
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
    
            # Draw rectangle around face
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
            # Put text just below the face box
            text_position = (x_min, y_max + 25)  # 25 pixels below box
            cv2.putText(
                frame,
                expr,
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )

    cv2.imshow("Expression Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
