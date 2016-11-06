#!/usr/bin/env python

import cv2
import numpy as np
import datetime
import time

SENSITIVITY_VALUE = 20
BLUR_SIZE = 10

trackingEnabled = False
debugMode = False
bExiting = False
debugWindowsVisible = False
pause = False

frame1 = None
frame2 = None
grayImage1 = None
grayImage2 = None
diffImage = None
threshImage = None

def searchForMovement(resThresh, cameraFeed):
    temp = None
    objectDetected = False
    temp = resThresh
    _, contours, hierarchy = cv2.findContours(temp,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if(contours.__len__()>0):
        objectDetected = True
    else:
        objectDetected = False

    if(objectDetected):
        largestContourVec = contours[contours.__len__() - 1] 
        x,y,w,h = cv2.boundingRect(largestContourVec)
        xpos = x + w/2
        ypos = y + h/2
        # cv2.rectangle(cameraFeed,(x,y),(x+w,y+h),(0,255,0),2)
        # cv2.ellipse(cameraFeed, (xpos,ypos)
        cv2.circle(cameraFeed,(xpos,ypos),10,(0,255,0),2)     

while(True):
    cap = cv2.VideoCapture("videos/bouncingBall.avi")
    while not cap.isOpened():
        cap = cv2.VideoCapture("videos/bouncingBall.avi")
        cv2.waitKey(1000)
        print "Wait for the header"
        

    while(cap.get(cv2.CAP_PROP_POS_FRAMES) < cap.get(cv2.CAP_PROP_FRAME_COUNT)-1):
        # read first frame and convert to grayscale
        flag, frame1 = cap.read()
        grayImage1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

        # read next frame and convert to grayscale
        flag, frame2 = cap.read()
        grayImage2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

        # get the difference
        diffImage = cv2.absdiff(grayImage1, grayImage2)

        # threshold intensity
        ret,threshImage = cv2.threshold(diffImage,SENSITIVITY_VALUE,255,cv2.THRESH_BINARY)
        
        if(debugMode):
            cv2.imshow("Difference Image", diffImage)
            cv2.imshow("Threshold Image", threshImage)
            debugWindowsVisible = True
        else:
            if(debugWindowsVisible):
                cv2.destroyWindow("Difference Image")
                cv2.destroyWindow("Threshold Image")

        # blur 
        threshImage = cv2.blur(threshImage, (BLUR_SIZE, BLUR_SIZE))

        ret,threshImage = cv2.threshold(threshImage,SENSITIVITY_VALUE,255,cv2.THRESH_BINARY)

        if(debugMode):
            cv2.imshow("Final Thresh", threshImage)
        else:
            if(debugWindowsVisible):
                cv2.destroyWindow("Final Thresh")
                debugWindowsVisible = False

        if (trackingEnabled):
            searchForMovement(threshImage, frame1)

        cv2.imshow("Video", frame1)

        ch = cv2.waitKey(1) & 0xFF
        if(ch == ord('q')):
            bExiting = True
            break
        elif (ch == ord('d')):
            debugMode = ~debugMode
        elif (ch == ord('t')):
            trackingEnabled = ~trackingEnabled
        elif (ch == ord('p')):
            pause = ~pause
            print "Code Paused"
            while (pause):
                if cv2.waitKey(25) & 0xFF == ord('p'):
                    pause = False
                    print "Code Resumed"
        

    cap.release()
    if(bExiting):
        break

# When everything done, release the capture
cv2.destroyAllWindows()