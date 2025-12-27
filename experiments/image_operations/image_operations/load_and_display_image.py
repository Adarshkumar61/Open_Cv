import cv2 as cv
import sys

# there are many ways to read imagee
# 1. backslash: use forward slash
# image = cv.imread('WIN_20251015_22_00_01_Pro.mp4')

# 2. use double backslash:
# image = cv.imread('c:\\Users\\adars\\Pictures\\robo.jpg')

# 3. read video (file or webcam)
# Use cv.VideoCapture with a filename or device index (0 for webcam)

cap = cv.VideoCapture('WIN_20251015_22_00_01_Pro.mp4')  # or use 0 for webcam

if not cap.isOpened():
    print('Error: cannot open video')
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break  # end of video
    # optional: resize frame before showing
    frame_resized = cv.resize(frame, (640, 360))
    cv.imshow('video', frame_resized)
    if cv.waitKey(25) & 0xFF == ord('q'):  # press 'q' to quit early
        break

cap.release()
cv.destroyAllWindows()

# If you only want to play the video, exit so the image code below doesn't run:
sys.exit(0)
# image = cv.imread(r'c:\Users\adars\Pictures\robo.jpg')


# resizing image:
# if image is None:
#     print('Check path of the image')
# else:
#     resized_image = cv.resize(image, (300,300))
#     # cv.imshow('img', image)
#     cv.imshow('resized', resized_image)
#     cv.waitKey(0)
#     cv.destroyAllWindows()