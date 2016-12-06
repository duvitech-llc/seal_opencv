#!/usr/bin/env python

import cv2
import numpy as np
import time
import imutils
import datetime
from matplotlib import pyplot as plot



cap = cv2.VideoCapture("videos/Nov_19_2.mov")
while not cap.isOpened():
    cap = cv2.VideoCapture("videos/Nov_19.mov")
    cv2.waitKey(1000)
    print ("Wait for the header")

while(True):

    text = "Unoccupied"
    ret, img = cap.read()
    if ret is False:
        break
    
    plot.ion()

    img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    row, cols = img.shape
    crow, ccol = row / 2, cols / 2
    fshift[crow - 30: crow+30, ccol - 30: ccol + 30] = 0

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    plot.subplot(221), plot.imshow(img, cmap = "gray")
    plot.title("Input"), plot.xticks([]), plot.yticks([])

    plot.subplot(222), plot.imshow(magnitude_spectrum, cmap = "gray")
    plot.title('magnitude_spectrum'), plot.xticks([]), plot.yticks([])

    plot.subplot(223), plot.imshow(img_back, cmap = "gray")
    plot.title("Input in JET"), plot.xticks([]), plot.yticks([])
    plot.pause(0.01)
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()