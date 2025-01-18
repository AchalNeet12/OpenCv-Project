import cv2
import numpy as np

# Create our body classifier
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Read the image
frame = cv2.imread('th.jpg')

if frame is None:
    print("Error: Image not found or unable to load.")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Display the output
    cv2.imshow('Pedestrians', frame)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed

cv2.destroyAllWindows()
