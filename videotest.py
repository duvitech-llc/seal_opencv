#!/usr/bin/env python

import cv2
import numpy as np
import time
import imutils
import datetime


scaling_factor = 1.0
    

def frame_diff(prev_frame, cur_frame, next_frame):
    diff_frames1 = cv2.absdiff(next_frame, cur_frame)
    diff_frames2 = cv2.absdiff(cur_frame, prev_frame)
    return cv2.bitwise_and(diff_frames1, diff_frames2)

def get_frame(cap):
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, 
            fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return cv2.applyColorMap(frame, cv2.COLORMAP_JET)


cap = cv2.VideoCapture("videos/Nov_19_2.mov")
while not cap.isOpened():
    cap = cv2.VideoCapture("videos/Nov_19.mov")
    cv2.waitKey(1000)
    print ("Wait for the header")

total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 5

prev_frame = get_frame(cap) 
cur_frame = get_frame(cap) 
next_frame = get_frame(cap) 

while(True):

    text = "Unoccupied"

    cv2.imshow("Object Movement", frame_diff(prev_frame, cur_frame, next_frame))

    # draw the text and timestamp on the frame
    cv2.putText(cur_frame, "Status: {}".format(text), (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(cur_frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
        (10, cur_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        

    cv2.imshow('original', cur_frame)
        
    prev_frame = cur_frame
    cur_frame = next_frame 
    next_frame = get_frame(cap)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.015)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()