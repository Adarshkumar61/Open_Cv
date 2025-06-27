import cv2
import datetime
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Could not open webcam.")
    exit()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

filter_mode = 'normal'
while True:
    
    ret, frame = cam.read()
    if not ret:
        print('frame not captuiring')
        break
    output = frame.copy()
    
    if filter_mode == 'gray':
            output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    if filter_mode == 'edge':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            output = cv2.Canny(gray, 100, 200)
        
    if filter_mode == 'face':
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face = face_cascade.detectMultiScale(gray, 1.1, 4)
            for x, y, w, h in face:
                cv2.rectangle(output, (x, y), (x+w, y+h), (0,255,0), 4)
            
    window = f'webcam now: {filter_mode.upper()}'
    cv2.imshow(window, output)
    key = cv2.waitKey(1)
    
    if key == ord('b'):
        break
    elif key == ord('s'):
        filename = f'capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg'
        cv2.imwrite(filename, frame)
        print(f'saved{filename}')
    elif key == ord('f'):
        filter_mode = 'face'
    elif key == ord('g'):
        filter_mode = 'gray'
    elif key == ord('e'):
        filter_mode = 'edge'

cam.release()
cv2.destroyAllWindows()