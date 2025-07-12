import cv2 as cv

image =  cv.imread('image/ada.png')
resized = cv.resize(image, (980, 950))
# apllying gray color 
# gray = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)

# applying HSV color:
hsv = cv.cvtColor(resized, cv.COLOR_BGR2HSV)

#applying Lab :

if image is None:
    print('check image source')
else:
    # cv.imshow('gray', gray)
    # cv.imshow('resized', resized)
    cv.imshow('hsv', hsv)
    cv.waitKey(0)
    cv.destroyAllWindows()