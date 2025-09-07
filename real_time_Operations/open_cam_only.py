import cv2
camera = cv2.VideoCapture(0)
#start the camera

if not camera.isOpened():
    print('camera cant open')
    exit()
    #check whether the camera is open or not
    
while True:
    ret, frame = camera.read()
    
    # ret:    true if frame is captured
    # frame : actual image from webcam
    if not ret:
        print('Cant recieve Frame')
        break
    
    cv2.imshow('Cam', frame)  #shows frame
    
    if cv2.waitKey(1) == ord('b'):
        break
camera.release() #  Turn off camera
cv2.destroyAllWindows()  # close all cv2 windows