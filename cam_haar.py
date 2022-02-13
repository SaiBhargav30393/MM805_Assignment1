import numpy as np
import cv2, os

cascadeHeadFront = cv2.CascadeClassifier('cascades/cascade_front.xml')
cascadeHeadSide = cv2.CascadeClassifier('cascades/cascade_side.xml')
cascadeHeadBack = cv2.CascadeClassifier('cascades/cascade_back.xml')

cap = cv2.VideoCapture(0)
while True:
    ret, inputImage = cap.read()
    if ret == True:
        grayScaleImage = cv2.cvtColor(inputImage, cv2.COLOR_BGR2GRAY)
        front = cascadeHeadFront.detectMultiScale(grayScaleImage, 1.3, 5)
        side = cascadeHeadSide.detectMultiScale(grayScaleImage, 1.3, 5)
        back = cascadeHeadBack.detectMultiScale(grayScaleImage, 1.3, 5)
        removeBox = False
        if len(front) > 0:
            for (x,y,w,h) in front:
                cv2.rectangle(inputImage,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = grayScaleImage[y:y+h, x:x+w]
                roi_color = inputImage[y:y+h, x:x+w]
            removeBox = True

        if len(side) > 0 and removeBox == False:
            for (x,y,w,h) in side:
                cv2.rectangle(inputImage,(x,y),(x+w,y+h),(255,255,0),2)
                roi_gray = grayScaleImage[y:y+h, x:x+w]
                roi_color = inputImage[y:y+h, x:x+w]
            removeBox = True
                    
        if len(back) > 0 and removeBox == False:
            for (x,y,w,h) in back:
                cv2.rectangle(inputImage,(x,y),(x+w,y+h),(255,0,255),2)
                roi_gray = grayScaleImage[y:y+h, x:x+w]
                roi_color = inputImage[y:y+h, x:x+w]

        cv2.imshow('inputImage',inputImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
