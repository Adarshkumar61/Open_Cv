import cv2
import numpy as np
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    if not ret:
        print('frame not capturing')
        break
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = model(frame)
    
    label = result.pandas().xyxy[0]['name'].tolist()
    
    final = np.squeeze(result.render())
    cv2.imshow('YOLO5', final)
    if cv2.waitKey(1) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()