import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("mask_detector.keras")

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

st.title("😷 Face Mask Detection System")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while run:

    ret, frame = camera.read()

    if not ret:
        st.write("Failed to capture image")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:

        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face,(100,100))
        face = face/255.0
        face = np.reshape(face,(1,100,100,3))

        pred = model.predict(face)

        if pred < 0.5:
            label = "Mask"
            color = (0,255,0)
        else:
            label = "No Mask"
            color = (0,0,255)

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
        cv2.putText(frame,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    FRAME_WINDOW.image(frame)

camera.release()
