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
        
        myblur = cv2.blur(frame, (8, 8))
        im_color = cv2.applyColorMap(myblur, cv2.COLORMAP_PINK)
        
        # Detect blobs.
        img_th = cv2.adaptiveThreshold(im_color,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        contours, hierarchy = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        
        # draw the text and timestamp on the frame
        cv2.putText(frame, "Status: {}".format(text), (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            
        prevFrame = frame

        cv2.imshow('original', frame)
        cv2.imshow("Frame", im_color)
        

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if pos_frame >= total_frames:
        print ("Reset position")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    time.sleep(0.015)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()