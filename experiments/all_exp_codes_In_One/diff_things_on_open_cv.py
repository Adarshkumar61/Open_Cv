# import cv2
# import pyttsx3
# import numpy as np
# import time 
# import datetime
# img = cv2.imread("image/ada.png")
# cv2.putText(img, "this is my photo", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 25, 20), 2)
# # cv2.rectangle(img, (10, 70), (100, 250), (0,0, 255), 2)
# cv2.circle(img, (500, 400), 50, (0, 0, 255), -1)
# # cv2.line(img, (100, 200),(200, 300), (220, 222, 145), 30)
# # cv2.line(img, (100, 200), (200, 400), (255, 0, 0), 2)
# gray =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# while True:
#    cv2.imshow('adarsh', img)
#    if cv2.waitKey(1) == ord('b'):
#        break


# cam = cv2.VideoCapture(0)

# while True:
#     ret, frame = cam.read()
#     if not ret:
#         print('frame not capturing..')  
#         break 
#     cv2.imshow('webcam', frame) 
    
#     key = cv2.waitKey(1)
#     if key == ord('b'):
#         break
        
# cam.release()
# cv2.destroyAllWindows()

# import datetime
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# cam = cv2.VideoCapture(0)

# filter_mode = 'normal'

# while True:
#     ret, frame = cam.read()
#     if not ret:
#         print('frame not capturing')
#         break
#     output = frame.copy()
        
#         #applying filters:
#     if filter_mode == 'gray':
#             output = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
#     elif filter_mode == 'edge':
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             output = cv2.Canny(gray, 100, 200)
#     elif filter_mode == 'face':
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = face_cascade.detectMultiScale(gray, 1.1, 4)
#             for x, y, w, z in faces:
#                 cv2.rectangle(output, (x, y), (x+w, y+z), (120, 125, 127), 2)
#     elif filter_mode == 'countour':
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#         contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         cv2.drawContours(frame, contours, -1, (0,255,0), 3)
        

                
#     window = f'webcam Name : {filter_mode.upper()}'
#     cv2.imshow(window, output)
        
#     key = cv2.waitKey(1)
        
#     if key == ord('b'):
#             break
#     elif key == ord('f'):
#             filter_mode = 'face'
#     elif key == ord('n'):
#             filter_mode = 'normal'
#     elif key == ord('s'):
#             filename = f'capture_{datetime.datetime.now().strftime('%Y%m%d_%H%H%S')}.jpg'
#             cv2.imwrite(filename, frame)
#             print(f'saved{filename}')
#     elif key == ord('e'):
#             filter_mode = 'edge'
#     elif key == ord('g'):
#             filter_mode = 'gray'
# #     elif key == ord('c'):
# #             filter_mode = 'countour'
            
# cam.release()
# cv2.destroyAllWindows()



# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Red color mask 
#     lower_red = np.array([0, 120, 70])
#     upper_red = np.array([10, 255, 255])
#     mask = cv2.inRange(hsv, lower_red, upper_red)

#     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > 500:
#             x, y, w, h = cv2.boundingRect(cnt)
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#     cv2.imshow("Tracking", frame)
#     if cv2.waitKey(1) & 0xFF == ord('b'):
#         break

# cap.release()
# cv2.destroyAllWindows()


 # Motion Detection Project:

# cap = cv2.VideoCapture(0)
# ret, frame1 = cap.read()
# ret, frame2 = cap.read()

# while True: 
#     diff = cv2.absdiff(frame1, frame2)
#     gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5,5), 0)
#     _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY) 
#     dilated = cv2.dilate(thresh, None, iterations=3)
#     contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 

#     for contour in contours:
#         target_detect = False
#         if cv2.contourArea(contour) < 700:
#             continue
#         x, y, w, h = cv2.boundingRect(contour)
#         cv2.rectangle(frame1, (x,y), (x+w, y+h), (245,255,0), 2)
#         target_detect = True
    
#     if target_detect:
#         print('Target Detected')
#         cap.release()
#         cv2.destroyAllWindows()
#         exit()
#     cv2.imshow("Motion", frame1)
#     frame1 = frame2
#     ret, frame2 = cap.read()
    

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break


# def speak(text):
#     engine = pyttsx3.init()
#     engine.say(text)
#     engine.runAndWait()

# if target_detect:
#     speak("Target detected!") #not working right now coming soon.
#     print("Target detected!")
#     cap.release()
    
#     cv2.destroyAllWindows()
#     exit()
# cap.release()
# cv2.destroyAllWindows()


# Start the webcam
# cam = cv2.VideoCapture(0)

# while True:
#     ret, frame = cam.read()
#     if not ret:
#         break

#     # Flip the frame (optional)
#     frame = cv2.flip(frame, 1)

#     # Convert to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     lower_red = np.array([0, 120, 70])
#     upper_red = np.array([10, 255, 255])
#     mask1 = cv2.inRange(hsv, lower_red, upper_red)

#     lower_red2 = np.array([170, 120, 70])
#     upper_red2 = np.array([180, 255, 255])
#     mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

#     # Combine both masks
#     mask = mask1 + mask2
    
#     #another color :
    
#     # lower_blue = np.array([100, 150, 70])
#     # upper_blue = np.array([130, 255, 255])
    
    
    
#     # mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
#     # Noise removal:
#     mask = cv2.erode(mask, None, iterations=2)
#     mask = cv2.dilate(mask, None, iterations=2)

#     # Find contours
#     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     m_detect = False
#     command = "No Object Detected"

#     if contours:
#         largest = max(contours, key=cv2.contourArea)
#         area = cv2.contourArea(largest)

#         if area > 500:
#             # Get bounding box
#             x, y, w, h = cv2.boundingRect(largest)
#             cx = x + w // 2
#             cy = y + h // 2

#             # Draw rectangle and center
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.circle(frame, (cx, cy), 5, (255, 255, 255), -1)
#             m_detect = True
#             if m_detect:
#                 # print('red object detected')
#                 command = 'red color object detected'
#     cv2.putText(frame, f"Command: {command}", (10, 40),
#             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#     # time.sleep(1)

#     # Show both windows
#     cv2.imshow("Frame", frame)
#     cv2.imshow("Mask", mask)

#     # Exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cam.release()
# cv2.destroyAllWindows()



# import cv2
# import numpy as np

# def pick_color(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         pixel = hsv[y, x]
#         print(f'HSV: {pixel}')

# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     cv2.imshow("Frame", frame)
#     cv2.setMouseCallback("Frame", pick_color)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import torch
# import cv2
# import numpy as np


# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use small YOLOv5
# # this downloads the yolo5 model version from ultralystic from Github.
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = model(rgb)
#     # sends current frame to yolo5 model

#     # detect object like human cat dog, bottle, bikes etc..

#     labels = results.pandas().xyxy[0]['name'].tolist()
#     # results.pandas() converts all of that into a Pandas DataFrame 
#     # extract the detected class names from result
#     # xyxy[0]  Means bounding Boxes in (x1, y1, x2, y2) format
#     # 'name' gives the deteceted class name
#     # tolist() converts them into a regular python list
#     # Display results
#     annotated_frame = np.squeeze(results.render())
#     # np.squueze removes the unwanted dimensions 
#     #  results.render() draws the bounding boxes + labels on the frame
#     cv2.imshow('YOLOv5 Detection', annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



# cam= cv2.VideoCapture(0)
# while True:
    
#    ret, frame = cam.read()
#    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
 
#    gray = np.float32(gray)
#    dst = cv2.cornerHarris(gray,2,3,0.04)
 
# #result is dilated for marking the corners, not important
#    dst = cv2.dilate(dst,None)
 
# # Threshold for an optimal value, it may vary depending on the image.
#    frame[dst>0.01*dst.max()]=[0,0,255]
 
#    cv2.imshow('dst',frame)

#    if cv2.waitKey(1) & 0xff == ord('q'):
#        break
   
# cam.release()
# cv2.destroyAllWindows()


# import numpy as np
# import cv2

# filename = 'image/ada.png'
# img = cv2.imread(filename)
# gray = cv2.cv2tColor(img,cv2.COLOR_BGR2GRAY)

# # find Harris corners
# gray = np.float32(gray)
# dst = cv2.cornerHarris(gray,2,3,0.04)
# dst = cv2.dilate(dst,None)
# ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
# dst = np.uint8(dst)

# # find centroids
# ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

# # define the criteria to stop and refine the corners
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
# corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)

# # Now draw them

# res = np
# res = np.hstack((centroids, corners))
# res = np.round(res).astype(np.intp)
# for i in range(res.shape[0]):
#     cv2.circle(img, (res[i, 0], res[i, 1]), 5, (0, 0, 255), 1)
#     cv2.circle(img, (res[i, 2], res[i, 3]), 5, (0, 255, 0), 1)
# cv2.imwrite('subpixel5.png', img)

# # Display the result
# cv2.imshow('Subpixel Corners', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()



# import cv2
# import numpy as np
# import torch as t

# model = t.hub.load('ultralytics/yolov5', 'yolov5s')
# cam = cv2.VideoCapture(0)
# while True:
#     ret, frame = cam.read()
#     if not ret:
#         print('frame not caturing') 
#         break
#     result = model(frame)
    
#     detection = result.pandas().xyxy[0] 
#     labels = detection['name'].tolist()
    
#     # logic:
#     action = 'No Action' 
#     if 'person' in labels:
#         action = 'move forward'
#     elif 'dog' in labels:
#         action = 'move back'
#     elif 'bottle':
#         action = 'stop' 
    
#     result = np.squeeze(result.render())
#     cv2.putText(result, f'Action{action}', (10,40), cv2.FONT_HERSHEY_COMPLEX, 1,  (0, 255, 255), 2)
#     cv2.imshow('frame', result)
    
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('b'):
#         break
# cam.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import math
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert to grayscale and apply edge detection
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 1)
#     edges = cv2.Canny(blur, 50, 150)

#     # Find contours
#     contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     for cnt in contours:
#         area = cv2.contourArea(cnt)
#         if area > 400:
            
#             # Approximate shape
#             perimeter = cv2.arcLength(cnt, True)
#             approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
#             x, y, w, h = cv2.boundingRect(approx)

#             shape = "Unknown"
#             sides = len(approx)
#             if perimeter == 0:
#                 continue
#             circular = 4 * math.pi * (area/(perimeter * perimeter))

#             if sides == 3:
#                 shape = "Triangle"
#             elif sides == 4:
#                 aspectRatio = float(w) / h
#                 shape = "Square" if 0.95 < aspectRatio < 1.05 else "Rectangle"
#             elif circular > 0.75:
#                 shape = "Circle"

#             cv2.drawContours(frame, [approx], 0, (0, 255, 0), 2)
#             cv2.putText(frame, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     cv2.imshow("Shape Detection", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# will do next time 
# import torch
# import torchvision.transforms as T
# from torchvision.models.segmentation import deeplabv3_resnet101
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# # Load pretrained DeepLabV3 model
# model = deeplabv3_resnet101(pretrained=True).eval()

# # Define transforms
# transform = T.Compose([
#     T.ToPILImage(),
#     T.Resize(256),
#     T.ToTensor(),
#     T.Normalize(mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]),
# ])

# # Open camera
# cap = cv2.VideoCapture(0)

# # COCO labels
# LABELS = {15: "person", 2: "car", 0: "background", 21: "cow", 17: "cat", 3: "motorcycle"}

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     input_tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).unsqueeze(0)
#     with torch.no_grad():
#         output = model(input_tensor)['out'][0]
#     output_predictions = output.argmax(0).byte().cpu().numpy()

#     # Create color mask for "person"
#     mask = np.zeros_like(frame)
#     mask[output_predictions == 15] = [0, 255, 0]  # Green for person
#     mask[output_predictions == 2] = [0, 0, 255]   # Red for car

#     # Blend mask on frame
#     result = cv2.addWeighted(frame, 0.7, mask, 0.3, 0)

#     cv2.imshow("Semantic Segmentation", result)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# import cv2
# while True:
#     img = 'image/ada.png'
#     cv = cv2.imread(img)
#     resized = cv2.resize(cv, (1000, 1000))
#     cv2.imshow('img', resized)
#     cv2.imwrite('adarsh_in_prem_mandir.jpg', resized)
#     if cv2.waitKey(0) == ord('b'):
#         break
# cv2.destroyAllWindows()

# # save = 
# load image:
# import cv2

# img = cv2.imread('image/ball.jpg')
# cv2.imshow("Image", img)
# cv2.imwrite("saved_imageee.jpg", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# import cv2

# img = cv2.imread('image/ball.jpg')  # make sure this image is in the same folder or use full path
# if img is None:
#     print("Image not loaded. Check the path.")
# else:
#     cv2.imshow("Loaded Image", img)
#     cv2.imwrite("saved_car.jpg", img)
#     print("Image saved as saved_car.jpg")
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

# import cv2
# import mediapipe as mp

# # Initialize MediaPipe hand detector
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=1)
# mp_draw = mp.solutions.drawing_utils

# # Open webcam
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)  # Flip image for mirror view
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(rgb)

#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

#     cv2.imshow("Hand Tracking", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


# import cv2 as cv

# cam = cv.VideoCapture(0)

# while True:
#     ret, frame = cam.read()
#     if not ret:
#         print('frame not capturing')
#         break
#     cv.imshow('cam',frame)
#     if cv.waitKey(1) == ord('b'):
#         break
# cam.release()
# cv.destroyAllWindows()

# import cv2 as cv

# cam = cv.VideoCapture(0)

# ret, frame = cam.read()
# ret, frame1 = cam.read()

# while True:
#     diff = cv.absdiff(frame, frame1)
    
#     gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    
#     blur = cv.GaussianBlur(gray, (5,5), 0)
    
#     _, thresh = cv.threshold(blur, 20, 255, cv.THRESH_BINARY)
    
#     dilate = cv.dilate(thresh, None, 3)
    
#     contors, _ = cv.findContours(dilate, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
#     for contor in contors:
#         if cv.contourArea(contor) <700:
#             continue
#         x, y, w, h = cv.boundingRect(contor)
#         cv.rectangle(frame1, (x,y), (x+w, y+h), (0, 255, 0), 2)
#         cv.imshow('fr1', frame)
#         frame = frame1
#         ret, frame1 = cam.read()
        
#         if cv.waitKey(1) == ord('b'):
#             break
        
#     cam.release()
#     cv.destroyAllWindows()

import cv2
import torch
# Load the Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally (mirror effect)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    center_x_frame = frame.shape[1] // 2

    for (x, y, w, h) in faces:
        # Draw rectangle on face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Calculate center of the face
        face_center_x = x + w // 2

        # Draw center lines
        cv2.line(frame, (center_x_frame, 0), (center_x_frame, frame.shape[0]), (255, 0, 0), 2)
        cv2.circle(frame, (face_center_x, y + h // 2), 5, (0, 0, 255), -1)

        # Direction logic
        if face_center_x < center_x_frame - 30:
            direction = "LEFT"
        elif face_center_x > center_x_frame + 30:
            direction = "RIGHT"
        else:
            direction = "CENTER"

        cv2.putText(frame, f"Direction: {direction}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        print(f"Face detected. Move: {direction}")

    # Show the video feed
    cv2.imshow("Face Tracker", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('b'):
        break

# Cleanup 
cap.release()
cv2.destroyAllWindows()

