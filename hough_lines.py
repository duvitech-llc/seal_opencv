import cv2
import numpy as np
import time


cap = cv2.VideoCapture("videos/Nov_19_2.mov")


while(True):
    ret, img = cap.read()
    if ret is False:
        break

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,120)
    minLineLength = 20
    maxLineGap = 5
    lines = cv2.HoughLinesP(edges,1,np.pi/180,20,minLineLength,maxLineGap)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow("edges", edges)
    cv2.imshow("lines", img)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()