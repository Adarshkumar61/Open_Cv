import cv2

img = cv2.imread('image/ada.png', cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, )

cv2.imshow('org', img) 
cv2.imshow('gray', gray)
import numpy as np
 
# Create a black image
imgg = np.zeros((300, 300, 3), dtype=np.uint8)

# Draw a red dot at (50, 50)
cv2.circle(imgg, (50, 50), 3, (0, 0, 255), -1)

cv2.imshow("Dot at (50,50)", imgg)


cv2.waitKey(0)
cv2.destroyAllWindows()