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


import cv2
import torch
import numpy as np

# Load YOLOv5 model
model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5s',
    pretrained=True
)

# Set confidence threshold (optional)
model.conf = 0.4  

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not opened")
    exit()

print("✅ YOLOv5 started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    # YOLO inference
    results = model(frame)

    # Render results on frame
    annotated_frame = np.squeeze(results.render())

    # Show output
    cv2.imshow("YOLOv5 Object Detection", annotated_frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
