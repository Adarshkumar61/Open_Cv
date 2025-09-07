import numpy as np
import torch
import cv2 as cv

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cam = cv.VideoCapture(0)
while True:
    ret, frame = cam.read()
    if not ret:
        print('frame not capturing')
        break
    
    result = model(frame)
    label = result.pandas().xyxy[0]['name'].tolist()
    
    final = np.squeeze(result.render())
    cv.imshow('frame', final)
    
    if cv.waitKey(1) == ord('b'):
        break
cam.release()
cv.destroyAllWindows()