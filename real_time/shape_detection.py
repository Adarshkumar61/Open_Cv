import cv2 as cv
import numpy as np
import math
cam = cv.VideoCapture(0)
while True:
    ret, frame = cam.read()
    if not ret:
        print('frame not capturing')
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gauss_blur = cv.GaussianBlur(gray, (5,5), 1)
    edge = cv.Canny(gauss_blur, 50, 150)
    
    contour, _ = cv.findContours(edge, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        area = cv.contourArea(cnt)
        if area > 400:
            perimeter = cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, 0.04 * perimeter, True)
            x, y, w, h = cv.boundingRect(approx)
            
            shape = 'unknown'
            sides = len(approx)
            if perimeter == 0:
                continue  # avoid division by zero
            circularity = 4 * math.pi * (area / (perimeter * perimeter))

            if sides == 3:
                shape = 'Triangle'
            elif sides == 4:
                
                as_ratio = float(w) / h
                shape = 'Square' if 0.95 < as_ratio < 1.05 else 'Rectangle'
            elif circularity > 0.75:
                shape = 'Circle'
            cv.drawContours(frame, [approx], 0, (0, 255, 0), 2)
            cv.putText(frame, shape, (x, y - 10), cv.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('b'):
        break
cam.release()
cv.destroyAllWindows()