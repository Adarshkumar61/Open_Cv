import cv2 as cv

image =  cv.imread('image/ada.png')
resized = cv.resize(image, (980, 950))
# apllying gray color 
# gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

# applying HSV color:
# hsv = cv.cvtColor(resized, cv.COLOR_BGR2HSV)

#applying Lab :
# lab = cv.cvtColor(resized, cv.COLOR_BGR2LAB)

#applying blur :
# for better contour and edge detection : reduce noise from image:
# blur = cv.GaussianBlur(resized, (5,5), 0)

#threshholding:
# _, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)

#edge detection:
# edge = cv.Canny(blur, 50, 150)


# if image is None:
#     print('check image source')
# else:
#     # cv.imshow('image', image)
#     cv.imshow('resized', resized)
#     # cv.imshow('gray', gray)
#     # cv.imshow('hsv', hsv)
#     # cv.imshow('lab', lab)
#     # cv.imshow('blur', blur)
#     # cv.imshow('thresh', thresh)
#     # cv.imshow('edge', edge)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

import cv2 as cv

face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
img = cv.imread("adarsh_in_prem_mandir.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for i, (x, y, w, h) in enumerate(faces):
    cv.putText(img, f'Face {i+1}', (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv.imshow("Detected Face", img)
cv.waitKey(0)
cv.destroyAllWindows()
