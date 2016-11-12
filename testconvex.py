#!/usr/bin/env python

import cv2
import numpy as np
import time

def thresh_callback(thresh):
    edges = cv2.Canny(blur,thresh,thresh*2)
    drawing = np.zeros(img.shape,np.uint8)     # Image to draw the contours
    contours,hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        cv2.drawContours(drawing,[cnt],0,(0,255,0),2)   # draw contours in green color
        cv2.drawContours(drawing,[hull],0,(0,0,255),2)  # draw contours in red color
        cv2.imshow('output',drawing)
        cv2.imshow('input',img)

cap = cv2.VideoCapture("videos/test.avi")
while not cap.isOpened():
    cap = cv2.VideoCapture("videos/test.avi")
    cv2.waitKey(1000)
    print ("Wait for the header")


flag, img = cap.read()

pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)

cv2.namedWindow('input')

thresh = 100
max_thresh = 255

cv2.createTrackbar('canny thresh:','input',thresh,max_thresh,thresh_callback)

thresh_callback(thresh)

if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
