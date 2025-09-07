import numpy as np
import cv2 as cv

cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

cam = cv.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print('frame is not capturing')
        break
    frame = cv.flip(frame, 1)
    
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    face = cascade.detectMultiScale(gray, 1.3, 5) # scaleFactor=1.3, minNeighbors=5
    
    frame_center = frame.shape[1] // 2
    
    if len(face) > 0:
        for (x, y, w, h) in face:
            cv.putText(frame, 'Face', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            face_center = x + w // 2
            cv.line(frame, (frame_center, 0), (frame_center, frame.shape[0]), (255, 0, 0), 2)
            cv.circle(frame, (face_center, y+h // 2), 5, (0, 255, 0), -2)
            
            # DIRECTION LOGIC:
            if face_center < frame_center -30:
                direction = 'Move Left'
            elif face_center > frame_center +30:
                direction = 'Move Right'
            else:
                direction = 'Move Center'
            
            cv.putText(frame, f'direction: {direction}', (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            # print('direction: ', direction)
    else:
        cv.putText(frame, 'No face detected', (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.imshow('frame', frame)
        
    if cv.waitKey(1) == ord('b'):
        break
        
cam.release()
cv.destroyAllWindows()