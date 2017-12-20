'''
Created on 24-Nov-2017

@author: aii32199
'''
#So We will load required libraries numpy for matrix operations
import numpy as np

#Import OpenCV library, in python we can call it cv2
import cv2

#OpenCV have module cascade classifier which is based on haar cascade and
#Adaboost algorithm, so we will call direct method.
#First we will load the pre trained classifiers for frontal face and eye
#detection, which are in the form of xml file.
face_cascade = cv2.CascadeClassifier('E:/OpenCV/opencv/sources/data/haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('E:/OpenCV/opencv/sources/data/haarcascades/haarcascade_eye.xml')

#Now let us load an image from the local directory
img = cv2.imread('download.jpg')

#Let's convert image into gray
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Here we will call the method which will find the faces in our input image
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#Lets run a loop to create sub images of faces from the input image using
#cv2.rectangle function
for (x,y,w,h) in faces:
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]

    #windows
    eyes = eye_cascade.detectMultiScale(roi_gray)
    #following function will create the rectangles around the eyes
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
#Following Lines will show the detected face images
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()