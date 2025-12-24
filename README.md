# 🧠 Real-Time Computer Vision Projects using OpenCV
==================================================

_A curated collection of real-time computer vision projects built using OpenCV and Python._

This repository focuses on **practical implementation, modular design, and real-world performance considerations**.  
The goal is to build vision modules that can be **extended to robotics, automation, and AI systems**.

---

## 🎯 Project Objectives
--------------------------------------------------
- 📌 Understand and implement computer vision techniques
- 💡 Build real-time processing pipelines
- 🧩 Create reusable and modular vision components
- 🤖 Prepare code for future integration with robotics or AI systems

---

## 🚀 Projects Included
--------------------------------------------------
### 👁️ Face Detection
- Real-time human face detection using Haar Cascade classifiers  
- Detects and highlights faces in webcam video

### 🖼️ Edge Detection
- Edge extraction using the Canny algorithm  
- Useful for object boundary detection and feature extraction

### 🔧 Image Processing Basics
- Grayscale conversion  
- Image resizing  
- Blurring and noise reduction  
- Fundamental operations for computer vision preprocessing

Each project is implemented as an **independent module** for clarity and reuse.

---

## 🗂️ Repository Structure
--------------------------------------------------
Open_Cv/
│
├── demo/ # Screenshots and output images

├── face_detection.py # Face detection module

├── edge_detection.py # Edge detection module

├── image_processing.py # Image processing utilities

├── requirements.txt # Dependency file

└── README.md # Project documentation

---

## 🧰 Tech Stack
--------------------------------------------------
- 🐍 **Python**
- 👁️ **OpenCV**
- 🔢 **NumPy**
- 📷 **Webcam / Image Input**

---

## 🏗️ System Architecture
--------------------------------------------------

Camera / Image Input

↓

Preprocessing (Resize, Grayscale)

↓

Vision Algorithm (Face / Edge / Filters)

↓

Real-Time Output Visualization


This pipeline is modular and designed for easy extension toward more advanced algorithms (e.g., CNNs, object tracking).

---

## ⚙️ Installation & Setup
--------------------------------------------------

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Adarshkumar61/Open_Cv.git

2️⃣ Navigate to the Project Directory
cd Open_Cv

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Run a Project
python face_detection.py


(Replace the filename for other modules as needed)
```
👤 Face Detection Output
![Face Detection Output](demo/face_detection_output.jpg)

🖼️ Edge Detection Output
![Edge Detection Output](demo/edge_detection_output.jpg)

▶️ Demo video available in the `demo` folder

🧪 Performance & Observations

⚡ Real-time execution using webcam input

💡 Performs well in normal lighting conditions

⚠ Classical techniques are fast but less accurate than deep learning

📚 Learning Outcomes

Real-time image processing fundamentals

Camera feed handling and performance considerations

Modular vision pipeline design

Ability to extend to robotics or AI perception systems

🔮 Future Enhancements

🚀 Integration of YOLO / Deep Learning detectors

📡 ROS2 vision node implementation

🤖 Deployment on embedded vision hardware (e.g., Jetson Nano)

📲 ESP32-CAM integration for edge vision

👨‍💻 Author

Adarsh Kumar
🎓 BCA Student | 🤖 Robotics & AI Enthusiast

🔗 GitHub: https://github.com/Adarshkumar61
