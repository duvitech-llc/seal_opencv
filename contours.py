#!/usr/bin/env python

import cv2
import numpy as np


cap = cv2.VideoCapture(0)


while(True):
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	overlay = frame.copy()
	output = frame.copy()
    	
    # draw a red rectangle surrounding Adrian in the image
	# along with the text "PyImageSearch" at the top-left
	# corner
	cv2.putText(overlay, "PyImageSearch: alpha={}".format(0.80),
		(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    cv2.addWeighted(overlay, 0.80, output, 1 - 0.80,
    		0, output)


    # open windows with original image, mask, res, and image with keypoints marked
    cv2.imshow('frame',frame)
    cv2.imshow('grayscale',gray)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()