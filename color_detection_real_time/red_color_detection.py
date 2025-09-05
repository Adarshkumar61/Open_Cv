import cv2 as cv
import numpy as np
import time

cam = cv.VideoCapture(0)

while True:
    if not cam.isOpened():
        print('camera is not opening..')
        break

    ret, frame  = cam.read()
    if not ret:
        print('frame is not capturing..')
        break
    fram = cv.flip(frame, 1)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv.inRange(hsv, lower_red, upper_red)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2
    
    mask = cv.erode(mask, None, iterations= 2)
    mask = cv.dilate(mask, None, iterations= 2)
    
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    command = 'no red obj detected'
    o_detected = False
    if contours:
        largest = max(contours, key=  cv.contourArea)
        area = cv.contourArea(largest)

        if area> 500:
            x, y, w, h = cv.boundingRect(largest)
            cx = x+ w //2
            cy = y+h //2
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv.circle(frame, (cx, cy,), 5, (0, 0, 255), -2)

            o_detected = True
            if o_detected:
                print('red obj detected')
                command = 'red obj detected'

    cv.putText(frame, f'{command}', (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 250), 2)
    cv.imshow('cam', frame)
    cv.imshow('mask', mask)
    if cv.waitKey(1) == ord('b'):
        break