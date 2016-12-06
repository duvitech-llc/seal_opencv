import cv2
import numpy as np
import time


cap = cv2.VideoCapture("videos/Nov_19_2.mov")


while(True):
    ret, img = cap.read()
    if ret is False:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 127, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img = cv2.drawContours(color, contours, -1, (0,255,0), 2)
    cv2.imshow("contours", color)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()