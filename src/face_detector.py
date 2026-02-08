import cv2

# load haar cascade classifier to detect frontal faces
face_cascade = cv2.CascadeClassifier( 
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# detect faces from frame 
def detectFaces(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # bgr to grayscale color conversion
    
    # detect faces in frame using classifier
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,  
        minNeighbors=5 # prioritize quality of detection
    )
    
    return faces
