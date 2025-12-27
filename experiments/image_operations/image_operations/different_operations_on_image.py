import cv2

img = cv2.imread('image/ada.png', cv2.IMREAD_UNCHANGED)
gray = cv2.cvtColor(img, )
cv2.putText(img, "OpenCV Rocks!", (0, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
cv2.line(img, (0, 0), (200, 200), (255, 0, 0), 2)
cv2.rectangle(img, (50, 50), (200, 150), (0, 0, 255), 3)
cv2.circle(img, (300, 100), 50, (0, 0, 255), -1)
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