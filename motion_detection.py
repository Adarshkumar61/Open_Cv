import cv2
import numpy as np
import datetime

cam = cv2.VideoCapture(0)

frame1 = cam.read()
frame2 = cam.read()

while True:
    