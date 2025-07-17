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