import cv2
import numpy as np
import datetime
import pyttsx3
cam = cv2.VideoCapture(0)

ret, frame1 = cam.read()
ret, frame2 = cam.read()

while True:
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilate = cv2.dilate(thresh, None, 3)
    contors, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    movement_detected = False
    for contor in contors:
        if cv2.contourArea(contor) < 700:
            continue 
        x, y, w, h = cv2.boundingRect(contor)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        movement_detected = True

    if movement_detected:
        print("Target detected! Shoot")
        cam.release()
        cv2.destroyAllWindows()
        exit()
    
    cv2.imshow('camera', frame1)
    frame1 = frame2
    ret, frame2 = cam.read()
    
    if cv2.waitKey(1) & 0xFF == ord('b'):
        break

cam.release()
cv2.destroyAllWindows()