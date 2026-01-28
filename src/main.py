import cv2
from screenCapture.capture import getScreenFrame
from expressionModel.expressions import detectExpression, face_mesh

while True:

    frame = getScreenFrame() # capture frame from screen

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # bgr to rbg color conversion

    results = face_mesh.process(rgb) # process frame to detect landmarks

    if results.multi_face_landmarks:

        h, w, _ = frame.shape # get frame dimensions

        for face in results.multi_face_landmarks: # for each face detected

            expr = detectExpression(face.landmark, h, w) # detect expression
    
            # draw rectangle around face
            x_coords = [lm.x * w for lm in face.landmark]
            y_coords = [lm.y * h for lm in face.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
            # display expression on frame
            text_position = (x_min, y_max + 25)
            cv2.putText(
                frame,
                expr,
                text_position,
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

cv2.destroyAllWindows()
