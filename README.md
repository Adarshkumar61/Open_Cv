🧠 Real-Time Computer Vision Projects using OpenCV
📌 Overview

A curated collection of real-time computer vision projects built using OpenCV and Python.
This repository focuses on practical implementation, modular design, and future-ready integration with robotics and AI systems.

The goal is not just to implement algorithms, but to understand how computer vision behaves in real-world scenarios such as:

Live camera feeds

Lighting variations

Performance constraints

🎯 Project Objectives

✅ Implement real-time vision pipelines

✅ Understand classical computer vision algorithms

✅ Design reusable and modular code

✅ Prepare vision modules for robotics & AI systems

🚀 Projects Included

👤 Face Detection

Real-time human face detection using Haar Cascade classifiers

Optimized for speed and live camera input

🖼️ Edge Detection

Canny edge detection for object boundary extraction

Useful for object recognition and navigation tasks

🔧 Basic Image Processing

Grayscale conversion

Resizing and blurring

Preprocessing for advanced vision tasks

Each project is implemented as an independent module for clarity and reusability.

🧰 Tech Stack

🐍 Python

👁️ OpenCV

🔢 NumPy

📷 Webcam / Image Input

🏗️ System Architecture
Camera / Image Input
        ↓
Preprocessing (Resize, Grayscale)
        ↓
Vision Algorithm (Detection / Edge Processing)
        ↓
Output Visualization


This modular pipeline allows easy replacement of classical algorithms with deep learning models in the future.

⚙️ Installation & Usage
1️⃣ Clone the Repository
git clone https://github.com/Adarshkumar61/Open_Cv.git

2️⃣ Navigate to Project Directory
cd Open_Cv

3️⃣ Install Dependencies
pip install opencv-python numpy

4️⃣ Run a Project
python face_detection.py



📸 Results & Demo
👤 Face Detection Output



👉 ![Face Detection Output](demo/face_detection_output.jpg)


🖼️ Edge Detection Output



👉 ![Edge Detection Output](demo/edge_detection_output.jpg)


▶️ Demo Video
A short demo video showcasing real-time execution is available inside the demo/ folder.

🧪 Performance & Observations:

⚡ Works in real time with standard webcam

💡 Performs best under good lighting conditions

⚠️ Classical algorithms are fast but less accurate than deep learning models


📚 Learning Outcomes:

Real-time image processing

Camera pipeline handling

Performance vs accuracy trade-offs

Vision system design for robotics

🔮 Future Enhancements

🤖 YOLO / CNN-based object detection

📡 ESP32-CAM integration

🧠 ROS2 vision node implementation

🚗 Autonomous robot perception system

👨‍💻 Author

Adarsh Kumar
🎓 BCA Student | 🤖 Robotics & AI Enthusiast

🔗 GitHub: https://github.com/Adarshkumar61
