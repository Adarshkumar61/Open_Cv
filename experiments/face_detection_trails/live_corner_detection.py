import cv2
import numpy as np

cam = cv2.VideoCapture(0)
while True:
    ret, frame = cam.read()
    if not ret:
        print('frame is not capturing')
        break
    # now convert it to gray first:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # change into numpy float
    gray = np.float32(gray)
    
    #for corner detection:
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    #now dilation for more efficient result:
    dst = cv2.dilate(dst, None)
    
    #now threshold:
    frame[dst>0.01*dst.max()]=[0,0,255]
    cv2.imshow('thresh', frame)
    if cv2.waitKey(1) == ord('b'):
        break
cam.release()
cv2.destroyAllWindows()