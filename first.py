import cv2
import datetime

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start camera
cap = cv2.VideoCapture(0)
filter_mode = 'normal'  # Modes: normal, gray, edge, face

print("Camera started. Press:")
print("  's' = save photo")
print("  'g' = grayscale mode")
print("  'e' = edge detection mode")
print("  'f' = face detection mode")
print("  'n' = normal mode")
print("  'q' = quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame.")
        break

    output = frame.copy()

    # Apply selected filter
    if filter_mode == 'gray':
        output = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif filter_mode == 'edge':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        output = cv2.Canny(gray, 100, 200)
    elif filter_mode == 'face':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show the output
    window_name = f"Webcam - Mode: {filter_mode.upper()}"
    cv2.imshow(window_name, output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = f"capture_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
    elif key == ord('g'):
        filter_mode = 'gray'
    elif key == ord('e'):
        filter_mode = 'edge'
    elif key == ord('f'):
        filter_mode = 'face'
    elif key == ord('n'):
        filter_mode = 'normal'

# Cleanup
cap.release()
cv2.destroyAllWindows()
