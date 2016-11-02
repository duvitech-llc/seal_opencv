#!/usr/bin/env python

import cv2
import numpy as np
import time

fps = 0
count = 0
cap = cv2.VideoCapture(0)
start = time.time()

while(True):
    count = count + 1
    _, frame = cap.read()
    #fps = cap.get(cv2.CAP_PROP_FPS)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    overlay = frame.copy()
    output = frame.copy()
        
    cv2.putText(overlay, "FPS: {}".format(int(fps)),
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.addWeighted(overlay, 0.80, output, 1 - 0.80,
            0, output)


    # open windows with original image, mask, res, and image with keypoints marked
    cv2.imshow('frame',frame)
    cv2.imshow('grayscale',gray)
    cv2.imshow('overlay',overlay)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if time.time() - start > 1:
        seconds = time.time() - start;
        fps = count/seconds
        count = 0
        start = time.time()


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()