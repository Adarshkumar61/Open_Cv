import numpy as np
import cv2 as cv
import torch as t

cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
model = t.hub.load('ultralytics/yolov5', 'yolov5s')

cam = cv.VideoCapture(0)
while True:
    ret, frame = cv.flip(frame, 1)
    if not ret:
        print('frame is not capturing')
        break
    result = model(frame)
    label = result.pandas().xyxy[0]['name'].tolist()
    final = np.squeeze(result.render())
    
    #code for face detection and direction logic implementation:
    gray = cv.cvtColor(final, cv.COLOR_BGR2GRAY)
    face = cascade.detectMultiScale(gray, 1.3, 5)
    
    frame_center = final.shape[1] // 2
    
    for (x, y, w, h) in face:
        cv.rectangle(final, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        face_center = x + w // 2
        cv.line(final, (frame_center, 0), (frame_center, final.shape[0]), (255, 0, 0), 2)
        cv.circle(final, (face_center, y+h // 2), 5, (0, 255, 0), -2)
        
        # DIRECTION LOGIC:
       