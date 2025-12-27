import cv2 as cv

import numpy as np
import time

cam = cv.VideoCapture(0) #Captures frames from  default webcam
while True:
    ret, frame = cam.read()
    if not ret:
        print('frame is not capturing..')
        break
    frame = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV) #Converts BGR frame to HSV color space for better color detection
    # color :
    # blue_lower = np.array([0, 0, 0])
    # blue_upper = np.array([180, 30, 46])
    blue_lower = np.array([100, 150, 70])
    blue_upper = np.array([130, 255, 255])
    mask = cv.inRange(hsv, blue_lower, blue_upper) 
    # Creates a mask using blue color range in HSV space
    mask = cv.erode(mask, None, iterations= 2) #
    mask = cv.dilate(mask, None, iterations= 2) # iteration: how many time we want to erode and dilate
    # Applies morphological operations (erode/dilate) to reduce noise in the mask
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    # Finds contours in the processed mask
    command = 'no object detected'
    o_detcet = False 
    if contours:
        largest = max(contours, key =cv.contourArea)
        area = cv.contourArea(largest)
        if area > 500: #Identifies the largest contour and draws bounding rectangle if area > 500 pixels
            x, y ,w , h = cv.boundingRect(largest)
            cx = x + w //2
            cy = y + h //2
            cv.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
            cv.circle(frame, (cx,cy), 5, (0,0,255), -2)
            o_detcet = False
            if o_detcet:
                print('blue object Detected')
                command = 'blue Object detected'
    cv.putText(frame, f'{command}', (10,40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2)
    cv.imshow('cam', frame)
    cv.imshow('mask', mask) # Displays results in two windows: original frame and mask
    if cv.waitKey(1) == ord('b') or command != 'no object detected':
        #     Exit conditions:
        # - Frame capture fails (ret == False)
        # - User presses 'b' key
        # - A blue object is detected (when o_detcet flag is True)
        time.sleep(1)
        break
cam.release()
cv.destroyAllWindows()