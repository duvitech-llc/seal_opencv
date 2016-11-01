#!/usr/bin/env python

import numpy as np
import cv2
import glob

# Load webcam calibration values for undistort()
# calibration values calculated using cv2.calibrateCamera() previously
# for our webcam
# calfile=np.load('webcam_calibration_data.npz')

# newcameramtx=calfile['newcameramtx']
# roi=calfile['roi']
# mtx=calfile['mtx']
# dist=calfile['dist']

blobsNotFound = []
cap = cv2.VideoCapture(0)

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

while(True):
    _, frame = cap.read()
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # define range of purple color in HSV
    purpleMin = np.array([115,50,10])
    purpleMax = np.array([160,255,255])

    # Sets pixels to white if in purple range, else will be set to black
    mask = cv2.inRange(hsv, purpleMin, purpleMax)
        
    # Bitwise-AND of mask and purple only image - only used for display
    res = cv2.bitwise_and(frame, frame, mask= mask)

    # mask = cv2.erode(mask, None, iterations=1)
    # commented out erode call, detection more accurate without it

    # dilate makes the in range areas larger
    mask = cv2.dilate(mask, None, iterations=1)    

    # Detect blobs.
    reversemask=255-mask
    keypoints = detector.detect(reversemask)
    if keypoints:
        print "found %d blobs" % len(keypoints)
        if len(keypoints) > 4:
            # if more than four blobs, keep the four largest
            keypoints.sort(key=(lambda s: s.size))
            keypoints=keypoints[0:3]
    else:
        print "no blobs"
 
    # Draw green circles around detected blobs
    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
    # open windows with original image, mask, res, and image with keypoints marked
    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)     
    cv2.imshow("Keypoints", im_with_keypoints)            
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

