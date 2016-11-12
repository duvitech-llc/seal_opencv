#!/usr/bin/env python

import cv2
import numpy as np
import datetime
import time
from matplotlib import pyplot as plt

class Point:
    """ Point class represents and manipulates x,y coords. """

    def __init__(self, x=0, y=0):
        """ Create a new point at x, y """
        self.x = x
        self.y = y

    def print_point(pt):
        print("({0}, {1})".format(pt.x, pt.y))


class Rectangle:
    """ A class to manufacture rectangle objects """

    def __init__(self, posn, w, h):
        """ Initialize rectangle at posn, with width w, height h """
        self.corner = posn
        self.width = w
        self.height = h

    def __str__(self):
        return  "({0}, {1}, {2})".format(self.corner, self.width, self.height)  

    def grow(self, delta_width, delta_height):
        """ Grow (or shrink) this object by the deltas """
        self.width += delta_width
        self.height += delta_height

    def move(self, dx, dy):
        """ Move this object by the deltas """
        self.corner.x += dx
        self.corner.y += dy

    def draw(self, target):
        """ Draw rectangle on image """
        cv2.rectangle(target, (self.corner.x,self.corner.y), (self.corner.x + self.width, self.corner.y + self.height), (0,0,255),2)
    
    def contains(self, rect):
        """ Determing if rectangle resides inside this Rectangle """
        if((rect.corner.x >=self.corner.x and rect.corner.y >= self.corner.y) and \
            (((rect.corner.x + rect.width) <= (self.corner.x + self.width)) and ((rect.corner.y + rect.height) <= (self.corner.y + self.height)))):
            return True
        else:
            return False

# Set threshold and maxValue default 127
limit = 170
maxValue = 255
 
# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()
    
# Change thresholds
params.minThreshold = 180;
params.maxThreshold = 256;
    
# Filter by Area.
params.filterByArea = True
params.minArea = 50
    
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1
    
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.5
    
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.5
    
detector = cv2.SimpleBlobDetector_create(params)


SENSITIVITY_VALUE = 30
BLUR_SIZE = 30

trackingEnabled = True
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

poolRect = Rectangle(Point(55,35),210,210)
objtrack = []
currtrack = []
count = 0

def searchForMovement(resThresh, cameraFeed):    
    global count
    global currtrack

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
        br = Rectangle(Point(x,y), w, h)
        currtrack.append(br)

    return objectDetected 

while(True):
    cap = cv2.VideoCapture("videos/test.avi")
    while not cap.isOpened():
        cap = cv2.VideoCapture("videos/test.avi")
        cv2.waitKey(1000)
        print ("Wait for the header")
        

    while(cap.get(cv2.CAP_PROP_POS_FRAMES) < cap.get(cv2.CAP_PROP_FRAME_COUNT)-5):
        text = "Unoccupied"
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
        
        ret,thresh = cv2.threshold(grayImage1,limit,maxValue,cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(grayImage1, contours, -1, (0,255,0), 3)

        if(debugMode):
            cv2.imshow("prevFrame Image", grayImage1)
            cv2.imshow("Difference Image", diffImage)
            cv2.imshow("Threshold Image", threshImage)
            debugWindowsVisible = True
        else:
            if(debugWindowsVisible):
                cv2.destroyWindow("prevFrame Image")
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
            if(searchForMovement(threshImage, frame1)):                
                text = "Warning"
                #currtrack.clear();
                
        # draw Rectangles
        for box in currtrack:
            box.draw(frame1)
            if(poolRect.contains(box)):
                count = count + 1 

        currtrack.clear()
        
        # add security timestamp to display 
        cv2.putText(frame1, "                   Status: {}".format(text), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        cv2.putText(frame1, "                   Count: {}".format(count), (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        cv2.putText(frame1, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
            (10, frame1.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # show video stream
        # cv2.rectangle(frame1, (poolRect.corner.x, poolRect.corner.y), (poolRect.width, poolRect.height), (255,0,0), 2)
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
            print ("Code Paused")
            while (pause):
                if cv2.waitKey(25) & 0xFF == ord('p'):
                    pause = False
                    print ("Code Resumed")
        
        time.sleep(0.025)
        

    cap.release()    
    count = 0
    if(bExiting):
        break

# When everything done, release the capture
cv2.destroyAllWindows()