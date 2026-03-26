# Face_Mask_Detection
# 😷 Face Mask Detection with Live Alert System

## 📌 Overview

This project is a **real-time Face Mask Detection System** that uses **Deep Learning and Computer Vision** to identify whether a person is wearing a mask or not through a webcam.

The system detects faces using **Haar Cascade Classifier** and classifies them using a **CNN model trained on masked and unmasked face images**. It also triggers an alert when a mask is not detected.

---

## 🚀 Features

* Real-time face detection using webcam
* Mask / No Mask classification
* Alert system for no-mask detection
* Easy deployment with Streamlit
* Lightweight and beginner-friendly

---

## 🛠️ Tech Stack

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* Haar Cascade Classifier
* Streamlit (for web app)

---

## 📂 Project Structure

```
FaceMaskProject/
│
├── dataset/
│   ├── with_mask/
│   └── without_mask/
│
├── mask_detector.model
├── haarcascade_frontalface_default.xml
├── train_model.py
├── detect_mask.py
├── app.py
└── README.md
```

---

## 📊 Dataset

You can use the Kaggle dataset:

👉 https://www.kaggle.com/datasets/wobotintelligence/face-mask-detection-dataset

Dataset should be structured as:

```
dataset/
   with_mask/
   without_mask/
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/FaceMaskDetection.git
cd FaceMaskDetection
```

Install dependencies:

```bash
pip install tensorflow opencv-python numpy streamlit
```

---

## 🧠 Model Training

Run the training script:

```bash
python train_model.py
```

This will generate:

```
mask_detector.model
```

---

## 🎥 Run Real-Time Detection

```bash
python detect_mask.py
```

Press **ESC** to exit.

---

## 🌐 Run Streamlit App

```bash
streamlit run app.py
```

---

## ⚠️ Common Errors & Fixes

### Webcam not working

```python
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
```

### Empty frame error

Ensure:

```python
if not ret or frame is None:
    continue
```

---

## 📈 Future Improvements

* Use MobileNetV2 for higher accuracy
* Add sound/email alert system
* Deploy on cloud (AWS / Heroku)
* Multi-face tracking optimization

---

## 👨‍💻 Author

Simran Paul

---

## 📜 License

This project is open-source and free to use for educational purposes.

---
