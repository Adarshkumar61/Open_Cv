import cv2
 
camera = cv2.VideoCapture(0)
backsub = cv2.createBackgroundSubtractorMOG2()
if not camera.isOpened():
    print('camera not opening')
    exit()
while True:
    ret, frame = camera.read()
    fgmask = backsub.apply(frame)
    
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Initialize prev_gray and prev_pts for optical flow
    if 'prev_gray' not in locals():
        prev_gray = gray
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        motion = gray.copy()
        motion[:] = 0
        flow_vis = gray.copy()
        flow_vis[:] = 0
        diff = gray.copy()
        diff[:] = 0
    else:
        # Calculate optical flow
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None)
        flow_vis = frame.copy()
        if next_pts is not None and prev_pts is not None:
            for i, (new, old) in enumerate(zip(next_pts[status == 1], prev_pts[status == 1])):
                a, b = new.ravel()
                c, d = old.ravel()
                cv2.line(flow_vis, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                cv2.circle(flow_vis, (int(a), int(b)), 3, (0, 0, 255), -1)
        # Frame differencing for motion detection
        diff = cv2.absdiff(prev_gray, gray)
        prev_gray = gray
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

    # Threshold for motion mask
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    # Display output:
    cv2.imshow('frame', frame)
    cv2.imshow('foreground mask', fgmask)
    cv2.imshow('optical', flow_vis)
    cv2.imshow('motion', thresh)
    
    key = cv2.waitKey(1) 
    if key == ord('b'):
        break
camera.release()
cv2.destroyAllWindows()
