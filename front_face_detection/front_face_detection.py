import numpy as np
import cv2

face_classifier = cv2.CascadeClassifier(r"haarcascade_frontalface_default.xml")

image = cv2.imread(r"img.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray,1.3,5)

if faces is ():
    print("NO FACES FOUND")

# We iterate through our faces array and drow a rectangle
# Over each face in faces
for (x,y,w,h) in faces: # X-coordinate,y-coordinate,w-width of the image,h-height of the image
    cv2.rectangle(image,(x,y),(x+w,y+h),(127,0,255),2)
    cv2.imshow('Face Detection',image)
    cv2.waitKey(0)

cv2.destroyAllWindows()