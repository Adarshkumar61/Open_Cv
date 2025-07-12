import cv2 as cv

# there are many ways to read imagee
# 1. backslash: use forward slash
image = cv.imread('c:/Users/adars\Pictures/robo.jpg')

# 2. use double backslash:
# image = cv.imread('c:\\Users\\adars\Pictures\\robo.jpg')

# 3. use a raw string:
# image = cv.imread(r'c:/Users/adars\Pictures/robo.jpg')

if image is None:
    print('check path of the image'.title())
else:
    
    cv.imshow('img', image)
    cv.waitKey(0)
    cv.destroyAllWindows()