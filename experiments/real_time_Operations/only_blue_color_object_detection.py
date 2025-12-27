import cv2
import numpy as np
import time

cam = cv2.VideoCapture(0)
while True:

    ret, frame = cam.read()
    if not ret:
        print('frame not capturing')
        break
    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 140, 70])
    upper_blue = np.array([130, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Noise removal:
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # finding the object shape:
    contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # it finds the edge of motion

    command = 'No Object Detected'
    detected = False

    if contour:
        # finding the largest blue object
        largest = max(contour, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        if area > 500:
            x, y, w, h = cv2.boundingRect(largest)
            cx = x + w // 2
            cy = y + h // 2
            detected = True
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -2)
            
           
            frame_width = frame.shape[1] // 2
            if cx < frame_width - 50:
                command = 'move left'
            elif cx > frame_width + 50:
                command = 'Move right'
            else:
                command = 'move forward'

    cv2.putText(frame, f'Command: {command}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    if cv2.waitKey(1) == ord('b'):
        break

cam.release()
cv2.destroyAllWindows()