import cv2
import numpy as np

# Load the car classifier
car_classifier = cv2.CascadeClassifier('haarcascade_car.xml')

# Read the image
frame = cv2.imread('img2.jpg')

if frame is None:
    print("Error: Image not found or unable to load.")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)  # Improve contrast

    # Pass frame to our car classifier
    cars = car_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Extract bounding boxes for any cars identified
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Make the window resizable and display the image
    cv2.namedWindow('Cars', cv2.WINDOW_NORMAL)
    cv2.imshow('Cars', frame)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed

cv2.destroyAllWindows()
