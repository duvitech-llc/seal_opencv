#!/usr/bin/env python

from imutils.object_detection import non_max_suppression
from imutils import paths
import cv2
import numpy as np
import time
import imutils
import datetime

blobsNotFound = []
pool_zone = None
firstFrame = None
prevFrame = None
# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()
    
# Change thresholds
params.minThreshold = 0;
params.maxThreshold = 256;
    
# Filter by Area.
params.filterByArea = True
params.minArea = 30
    
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1
    
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5
    
# Filter by Inertia
params.filterByInertia =True
params.minInertiaRatio = 0.5
    
detector = cv2.SimpleBlobDetector_create(params)

cap = cv2.VideoCapture("videos/Nov_19_2.mov")
while not cap.isOpened():
    cap = cv2.VideoCapture("videos/Nov_19.mov")
    cv2.waitKey(1000)
    print ("Wait for the header")

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 5

while(True):
    flag, frame = cap.read()

    text = "Unoccupied"
    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)


    if flag:
        
        myblur = cv2.GaussianBlur(frame, (25, 25), 0)
        hsv = cv2.cvtColor(myblur, cv2.COLOR_RGB2HSV_FULL)
    

        if firstFrame is None:
            firstFrame = frame
            refFrame = hsv

        frameDelta = cv2.absdiff(hsv, refFrame)
        
        # define range of purple color in HSV
        purpleMin = np.array([115,50,10])
        purpleMax = np.array([160,255,255])

        # Sets pixels to white if in purple range, else will be set to black
        mask = cv2.inRange(frameDelta, purpleMin, purpleMax)
        
        # Bitwise-AND of mask and purple only image - only used for display
        res = cv2.bitwise_and(frame, frame, mask= mask)

        # Detect blobs.
        reversemask=255-mask
        keypoints = detector.detect(reversemask)
        if keypoints:
            print ("found %d blobs" % len(keypoints))
            if len(keypoints) > 4:
                # if more than four blobs, keep the four largest
                keypoints.sort(key=(lambda s: s.size))
                keypoints=keypoints[0:3]
        else:
            print ("no blobs")
                
        # draw the text and timestamp on the frame
        cv2.putText(frame, "Status: {}".format(text), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            
        # The frame is ready and already captured
        cv2.imshow('original', frame)
        cv2.imshow('delta', frameDelta)
        cv2.imshow('hsv', hsv)
            
        prevFrame = frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if pos_frame >= total_frames:
        print ("Reset position")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    # time.sleep(0.015)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()