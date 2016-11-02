#!/usr/bin/env python

import cv2
import numpy as np
import time


cap = cv2.VideoCapture("videos/test.avi")
while not cap.isOpened():
    cap = cv2.VideoCapture("videos/test.avi")
    cv2.waitKey(1000)
    print "Wait for the header"

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 5

# Set threshold and maxValue default 127
limit = 150
maxValue = 255
 
# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()
    
# Change thresholds
params.minThreshold = 180;
params.maxThreshold = 256;
    
# Filter by Area.
params.filterByArea = True
params.minArea = 30
    
# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1
    
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.5
    
# Filter by Inertia
params.filterByInertia =False
params.minInertiaRatio = 0.5
    
detector = cv2.SimpleBlobDetector_create(params)


while(True):
    flag, frame = cap.read()

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    if flag:
        # keypoints = detector.detect(frame)
        # im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,limit,maxValue,cv2.THRESH_TOZERO)
        im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        
        # cv2.imshow('GrayScale', im2);

        # The frame is ready and already captured
        cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if pos_frame >= total_frames:
        print "Reset position"
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    time.sleep(0.015)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()