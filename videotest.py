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

while(True):
    flag, frame = cap.read()

    pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

    if flag:
        # The frame is ready and already captured
        cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if pos_frame >= total_frames:
        print "Reset position"
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
    time.sleep(0.02)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()