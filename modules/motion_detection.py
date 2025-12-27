import cv2
import time

cam = cv2.VideoCapture(0)

ret, frame1 = cam.read()
ret, frame2 = cam.read()

while True:
    diff = cv2.absdiff(frame1, frame2)
    #frame1 and frame2 are two pictures taken one after the other from the webcam.
    # cv2.absdiff() checks the difference between the two.
    # If something moved, the difference will be big.
    
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Converts the difference image to black and white (grayscale).
    # Easier for computer to work with one color channel than three (RGB).
    
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # Smooths the image to reduce small unwanted movements like tiny light flickers or camera shake.
    #(5,5) is the size of the smoothing filter. You can think of it as "softening the image" to remove noise.
   
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)   # make it black and white
    # Converts the blurred image into just black or white.
    # Anything brighter than 20 becomes white (255) — showing motion.
    # Everything else becomes black (0) — no motion.
    # So now the moving object looks white on a black background!
    
    dilate = cv2.dilate(thresh, None, 3)   # enlarge white spaces
    # Enlarges the white area using dilation.
    # This makes the motion areas more clear and connected — fills small gaps.
    
    contors, _ = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  #find edges of white spaces
    # Now, it tries to find the outlines of white shapes (motion areas).
    # These outlines help you draw rectangles later to show where motion is detected.
    
    movement_detected = False
    for contor in contors:
        if cv2.contourArea(contor) < 700:
            continue 
        x, y, w, h = cv2.boundingRect(contor)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        movement_detected = True

    # if movement_detected:
    #     print("Target detected! Shoot")
    #     time.sleep(1)
    #     cam.release()
    #     cv2.destroyAllWindows()
    #     exit()
    
    cv2.imshow('camera', frame1)
    frame1 = frame2
    ret, frame2 = cam.read()
    
    if cv2.waitKey(1) & 0xFF == ord('b'):
        break

cam.release()
cv2.destroyAllWindows()