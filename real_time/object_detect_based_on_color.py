import numpy as np
import cv2

# frame caputre:
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print('frame not capturing')
        
    frame = cv2.flip(frame, 1)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Applying masks:
    lower_blue = np.array([])
    upper_blue = np.array([])
    
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # mask = cv2.erode(frame, None, iterations= 2)
    # mask = cv2.dilate(frame, None, iterations= 2)
    
    if 'color' not in globals():
        color = 'blue'
    
    key = cv2.waitKey(1) &0xFF
    if key == ord('k'):
        color = 'black'
    elif key == ord('r'):
        color = 'red'
    elif key == ord('g'):
        color = 'green'
    elif key == ord('b'):
        color = 'blue'
        
        
    if color == 'black':
        lower = np.array([])
        upper = np.array([])
        mask = cv2.inRange(hsv, lower, upper)
    elif color == 'red':
        lower_red1 = np.array([])
        upper_red1 = np.array([])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        lower_red2 = np.array([])
        upper_red2 = np.array([])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 | mask2
        
    elif color == 'green':
        lower = np.array([])
        upper = np.array([])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Noise removal
        mask = cv2.erode(mask, None, iterations= 2)
        mask = cv2.dilate(mask, None, iterations= 2)
        
        #finding the object shape:
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        command = 'No Object Detected'
        detected =  False
        if contours:
            largest = max(contours, key = cv2.contourArea)
            area = cv2.contourArea(largest)
            
            if area > 500:
                x, y, w, h = cv2.boundingRect(largest)
                cx = x + w // 2
                cy = y + h // 2
                detected = True
                cv2.rectangle(frame, (x,y),(x+w, y+h), (0,255,0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -2)
                
                frame_center = frame.shape[1] //2
                if cx < frame_center -50:
                    command = 'move left'.title()
                elif cx > frame_center + 50:
                    command = 'move rigth'.title()
                else:
                    command = 'move forward'.title()
                    
        cv2.putText(frame, f'Command: {command}', (10,40), cv2.FONT_HERSHEY_COMPLEX, (0,0,255), 2)
        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        
        if key == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
