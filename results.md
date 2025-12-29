## 📊 Results & Observations

- Face Detection works reliably in normal and bright lighting
- Color Detection performs best with fixed color thresholds
- Edge Detection identifies object contours cleanly
- Realtime operations handle frame streams at ~20 FPS

### Notes
- Performance drops in low light
- Future improvement: integrate YOLO for deep learning-based detection


## ❌ What Didn’t Worked

- Haar cascade accuracy dropped significantly in low light
- Color detection was unstable without HSV tuning
- Multiple operations reduced FPS below usable range

## 🔮 Future Improvements

- Replace Haar Cascade with YOLOv8 for face detection
- Move OpenCV pipeline into a ROS2 vision node
- Deploy on ESP32-CAM for edge vision