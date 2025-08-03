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
    