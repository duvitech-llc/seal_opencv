#!/usr/bin/env python

import cv2
import numpy as np
import time
import datetime

pool_zone = None
firstFrame = None
prevFrame = None


cap = cv2.VideoCapture("videos/Nov_19.mov")
while not cap.isOpened():
    cap = cv2.VideoCapture("videos/Nov_19.mov")
    cv2.waitKey(1000)
    print ("Wait for the header")

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 5

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

people_count = 0 


while(True):
    flag, frame = cap.read()

    text = "Unoccupied"
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    if flag:
        
        imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        imgray = cv2.GaussianBlur(imgray, (25, 25), 0)
        #keypoints = detector.detect(imgray)
        #im_with_keypoints = cv2.drawKeypoints(imgray, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        ret,thresh = cv2.threshold(imgray,limit,maxValue,cv2.THRESH_BINARY)
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # print "Contours: {}".format(contours.__len__())

        if(contours.__len__()>1):
            text = "Warning"

        # if the first frame is None, initialize it
        if firstFrame is None:
            firstFrame = frame
            pool_zone = contours[0]
            hull = cv2.convexHull(pool_zone)         
            cv2.fillConvexPoly(firstFrame,pool_zone,(255,255,255), lineType=8, shift=0)
            continue
        else:
            cv2.drawContours(imgray, contours, -1, (0,255,0), 3)


        # draw the text and timestamp on the frame
        cv2.putText(frame, "Status: {}".format(text), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            
        # The frame is ready and already captured
        cv2.imshow('blur', firstFrame)
        cv2.imshow('outline', imgray)
        cv2.imshow('original', frame)
            
        prevFrame = frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if pos_frame >= total_frames:
        print ("Reset position")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    time.sleep(0.015)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()