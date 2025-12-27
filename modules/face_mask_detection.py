import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

# dataset/
#    train/
#       with_mask/
#       without_mask/
#    test/
#       with_mask/
#       without_mask/

# Image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20,
                                   zoom_range=0.2, shear_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory("dataset/train",
                                              target_size=(128,128),
                                              batch_size=32,
                                              class_mode="binary")

test_set = test_datagen.flow_from_directory("dataset/test",
                                            target_size=(128,128),
                                            batch_size=32,
                                            class_mode="binary")

model = Sequential([
    Conv2D(32,(3,3),activation="relu",input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    
    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D(2,2),
    
    Flatten(),
    Dense(128,activation="relu"),
    Dropout(0.5),
    Dense(1,activation="sigmoid")
])

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.summary()


history = model.fit(train_set, epochs=10, validation_data=test_set)

model.save("face_mask_model.h5")

# Load trained model
from tensorflow.keras.models import load_model
model = load_model("face_mask_model.h5")

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (128,128))
        face = face/255.0
        face = np.expand_dims(face, axis=0)
        
        pred = model.predict(face)[0][0]
        
        if pred < 0.5:
            label = "Mask"
            color = (0,255,0)
        else:
            label = "No Mask"
            color = (0,0,255)
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,label,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
    
    cv2.imshow("Face Mask Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

