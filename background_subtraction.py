import cv2
import numpy as np
import time
import imutils
import datetime
from matplotlib import pyplot as plt


class Point:
    """ Point class represents and manipulates x,y coords. """

    def __init__(self, x=0, y=0):
        """ Create a new point at x, y """
        self.x = x
        self.y = y

    def print_point(pt):
        print("({0}, {1})".format(pt.x, pt.y))


class Rectangle:
    """ A class to manufacture rectangle objects """

    def __init__(self, posn, w, h):
        """ Initialize rectangle at posn, with width w, height h """
        self.corner = posn
        self.width = w
        self.height = h

    def __str__(self):
        return  "({0}, {1}, {2})".format(self.corner, self.width, self.height)  

    def grow(self, delta_width, delta_height):
        """ Grow (or shrink) this object by the deltas """
        self.width += delta_width
        self.height += delta_height

    def move(self, dx, dy):
        """ Move this object by the deltas """
        self.corner.x += dx
        self.corner.y += dy

    def draw(self, target):
        """ Draw rectangle on image """
        cv2.rectangle(target, (self.corner.x,self.corner.y), (self.corner.x + self.width, self.corner.y + self.height), (0,0,255),2)
    
    def contains(self, rect):
        """ Determing if rectangle resides inside this Rectangle """
        if((rect.corner.x >=self.corner.x and rect.corner.y >= self.corner.y) and \
            (((rect.corner.x + rect.width) <= (self.corner.x + self.width)) and ((rect.corner.y + rect.height) <= (self.corner.y + self.height)))):
            return True
        else:
            return False


currtrack = []

# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()
    
params.minDistBetweenBlobs = 50.0
params.filterByInertia = False
params.filterByConvexity = False
params.filterByColor = False
params.filterByCircularity = False
params.filterByArea = True
params.minArea = 20.0
params.maxArea = 500.0

def sort_contours(cnts, method="left-to-right"):
    	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def draw_contour(image, c, i):
	# compute the center of the contour area and draw a circle
	# representing the center
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
 
	# draw the countour number on the image
	cv2.putText(image, "#{}".format(i + 1), (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (255, 255, 255), 2)
 
	# return the image with the contour number drawn on it
	return image

def get_frame(cap, scaling_factor):
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=scaling_factor, 
            fy=scaling_factor, interpolation=cv2.INTER_AREA)
    return frame

def searchForMovement(resThresh, cameraFeed):    
    global count
    global currtrack
    minx = None
    miny = None
    maxw = None
    maxh = None
    
    temp = None
    objectDetected = False
    temp = resThresh
    _, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if(contours.__len__()>1):
        objectDetected = True
    else:
        objectDetected = False

    if(objectDetected):        
        # loop over the contours
        for c in contours:
            # if the contour is too small, ignore it
            if cv2.contourArea(c) < 10:
                continue
    
            # compute the bounding box for the contour, draw it on the frame,
            # and update the text
            (x, y, w, h) = cv2.boundingRect(c)
            if minx is None:
                minx = x 
                miny = y 
                maxw = w 
                maxh = h
            else:
                if x < minx:
                    minx = x 
                if y < miny:
                    miny = y 
                if w > maxw:
                    maxw = w 
                if h > maxh:
                    maxh = h 
        
        if minx is not None:
            # origin       
            xpos = minx + maxw/2
            ypos = miny + maxh/2
            br = Rectangle(Point(xpos,ypos), maxw, maxh)
            currtrack.append(br)
            cv2.rectangle(cameraFeed, (minx, miny), (minx + maxw, miny + maxh), (0, 255, 0), 2)

    return objectDetected 

if __name__=='__main__':
    cap = cv2.VideoCapture("videos/Nov_19.mov")
    while not cap.isOpened():
        cap = cv2.VideoCapture("videos/Nov_19.mov")
        cv2.waitKey(1000)
        print ("Wait for the header")
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    bgSubtractor = cv2.createBackgroundSubtractorMOG2()
        
    detector = cv2.SimpleBlobDetector_create(params)

    limit = 184
    maxValue = 255
    history = 100
    firstFrame = True
    ctp = None

    while True:
        frame = get_frame(cap, 1.0)
        if firstFrame:
            grayImage1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            ret,thp = cv2.threshold(grayImage1,limit,maxValue,cv2.THRESH_BINARY)
            imp, ctp, hierarchy = cv2.findContours(thp,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            firstFrame = False

        # cv2.drawContours(frame, ctp, -1, (180,0,0), 3)

        mask = bgSubtractor.apply(frame, learningRate=1.0/history)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        keypoints = detector.detect(mask & frame)
        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        imgray = cv2.cvtColor(mask & frame,cv2.COLOR_BGR2GRAY)
        ret,thresh = cv2.threshold(imgray,127,255,0)
        
        searchForMovement(thresh, frame)
        
        cv2.imshow('Original', frame)
        cv2.imshow('Moving Objects', mask & frame)
        c = cv2.waitKey(10)
        if c == 27:
            break
        
        time.sleep(0.020)

    cap.release()
    cv2.destroyAllWindows()
