import cv2
import pyttsx3
import numpy as np
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
#     if cv2.waitKey(1) & 0xFF == ord('q'):
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
# cap = cv2.VideoCapture(0)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip the frame (optional)
#     frame = cv2.flip(frame, 1)

#     # Convert to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # lower_red = np.array([0, 120, 70])
#     # upper_red = np.array([10, 255, 255])
#     # mask1 = cv2.inRange(hsv, lower_red, upper_red)

#     # lower_red2 = np.array([170, 120, 70])
#     # upper_red2 = np.array([180, 255, 255])
#     # mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

#     # Combine both masks
#     # mask = mask1 + mask2
    
#     #another color :
    
#     lower_blue = np.array([100, 150, 70])
#     upper_blue = np.array([130, 255, 255])
    
    
    
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
#     # Noise removal:
#     mask = cv2.erode(mask, None, iterations=2)
#     mask = cv2.dilate(mask, None, iterations=2)

#     # Find contours
#     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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

#             # Frame width
#             frame_center = frame.shape[1] // 2

#             # Decision logic
#             if cx < frame_center - 50:
#                 command = "Move Left"
#             elif cx > frame_center + 50:
#                 command = "Move Right"
#             else:
#                 command = "Move Forward"

#     # Show command on screen
#     cv2.putText(frame, f"Command: {command}", (10, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

#     # Show both windows
#     cv2.imshow("Frame", frame)
#     cv2.imshow("Mask", mask)

#     # Exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
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

import torch
import cv2
import numpy as np


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Use small YOLOv5
# this downloads the yolo5 model version from ultralystic from Github.
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    # sends current frame to yolo5 model
    # detect object like human cat dog, bottle, bikes etc..
    labels = results.pandas().xyxy[0]['name'].tolist()
    # results.pandas() converts all of that into a Pandas DataFrame 
    # extract the detected class names from result
    # xyxy[0]  Means bounding Boxes in (x1, y1, x2, y2) format
    # 'name' gives the deteceted class name
    # tolist() converts them into a regular python list
    # Display results
    annotated_frame = np.squeeze(results.render())
    # np.squueze removes the unwanted dimensions 
    #  results.render() draws the bounding boxes + labels on the frame
    cv2.imshow('YOLOv5 Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
