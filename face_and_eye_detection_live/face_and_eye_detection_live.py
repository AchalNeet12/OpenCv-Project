import cv2
import numpy as np

# Load the classifiers
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('haarcascade_eye.xml')

def face_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    
    # Check if any faces are detected
    if len(faces) == 0:
        return img
    
    for (x, y, w, h) in faces:
        x = x - 50
        w = w + 50
        y = y - 50
        h = h + 50
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        
        # Detect eyes within the face region
        eyes = eye_classifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2) 
    
    return img

# Start video capture
cap = cv2.VideoCapture(0)

# Set window to full screen
cv2.namedWindow('Our Face Extractor', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Our Face Extractor', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Apply the face detector and display the frame
    cv2.imshow('Our Face Extractor', face_detector(frame))
    
    # Exit the loop when Enter key is pressed
    if cv2.waitKey(1) == 13:  # 13 is the Enter Key
        break
        
# Release the capture and close the windows
cap.release()
cv2.destroyAllWindows()
