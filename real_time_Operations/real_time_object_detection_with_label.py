# import numpy as np
# import torch
# import cv2 as cv

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# cam = cv.VideoCapture(0)
# while True:
#     ret, frame = cam.read()
#     if not ret:
#         print('frame not capturing')
#         break
    
#     result = model(frame)
#     label = result.pandas().xyxy[0]['name'].tolist()
    
#     final = np.squeeze(result.render())
#     cv.imshow('frame', final)
    
#     if cv.waitKey(1) == ord('b'):
#         break
# cam.release()
# cv.destroyAllWindows()


import torch
import cv2
import numpy as np

model = torch.hub.load('./yolov5', 'yolov5s', source='local')

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    results = model(frame)
    final = np.squeeze(results.render())

    cv2.imshow("YOLOv5", final)

    if cv2.waitKey(1) & 0xFF == ord('b'):
        break

cam.release()
cv2.destroyAllWindows()
