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


import cv2
def pick_color(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = hsv[y , x]
        print(f'HSV:{pixel}')

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    if not ret:
        print('frame not capturing..')
        break
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cv2.imshow('cam', frame)
    cv2.setMouseCallback('cam', pick_color)
    
    if cv2.waitKey(1) == ord('b'):
        break

cam.release()
cv2.destroyAllWindows()
    
   
    