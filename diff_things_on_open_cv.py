import cv2
 
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


cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        print('frame not capturing..')
        break
    cv2.imshow('webcam', frame)
    
    key = cv2.waitKey(1)
    if key == ord('b'):
        break
        
cam.release()
cv2.destroyAllWindows()