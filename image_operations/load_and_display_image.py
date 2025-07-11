import cv2 as cv
# img_path = 'c:\Users\adars\Pictures\robo.jpg'
# img = cv.imread(img_path)
# cv.imshow('img', img)
# cv.waitKey(0)
# cv.destroyAllWindows()
# OR
image = cv.imread('c:\Users\adars\Pictures\robo.jpg') 
# explanation: 
# there are many ways to display imagee

cv.imshow('img', image)
cv.waitKey(0)
cv.destroyAllWindows()