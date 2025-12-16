# 🫁 Pneumonia Detection System (AI Powered)

A Deep Learning–based web application that detects **Pneumonia** from chest X-ray images using **Transfer Learning** and a **Streamlit** user interface.  
This project demonstrates the complete workflow from **model training** to **deployment**.

---

## 📌 Project Objective

To build an AI-powered Pneumonia Detection System that:
- Uses a **pre-trained CNN model**
- Classifies chest X-ray images as **Pneumonia** or **Normal**
- Provides prediction confidence
- Is deployed as a **Streamlit web application**

---

## 🧠 Technologies Used

- Python
- TensorFlow / Keras
- Transfer Learning (MobileNetV2)
- Convolutional Neural Networks (CNN)
- Streamlit
- NumPy
- Matplotlib
- Pillow (PIL)

---

## 📂 Dataset

**Dataset Used:**  
Chest X-Ray Images (Pneumonia) – Kaggle

**Classes:**
- Pneumonia
- Normal

**Dataset Structure:**
dataset/
├── train/
│ ├── Pneumonia/
│ └── Normal/
├── val/
│ ├── Pneumonia/
│ └── Normal/
└── test/
├── Pneumonia/
└── Normal/



---

## 🏗️ Model Architecture

- **Base Model:** MobileNetV2 (Pre-trained on ImageNet)
- **Transfer Learning:** Base layers frozen
- **Custom Layers Added:**
  - Global Average Pooling
  - Dense (128 neurons, ReLU)
  - Dropout (0.5)
  - Output Dense Layer (Sigmoid)

**Loss Function:** Binary Crossentropy  
**Optimizer:** Adam  
**Evaluation Metric:** Accuracy

---

## 📊 Model Performance

- Training Accuracy ✔️
- Validation Accuracy ✔️
- Loss Curve ✔️
- Final Test Accuracy ✔️

Training and validation graphs are saved as `training_plot.png`.

---

## 💾 Saved Model

- Model file: `pneumonia_model.h5`
- Format: Keras HDF5
- Used directly in the Streamlit application for inference

---

## 🌐 Streamlit Web Application

### Features:
- 📤 Upload chest X-ray image
- 🖼️ Image preview
- 🔍 Pneumonia detection button
- 📈 Prediction result with confidence percentage
- 📊 Probability bar chart

### App Title:
> **Pneumonia Detection System (AI Powered)**

---

## 🚀 Deployment

- **Platform:** Streamlit Cloud
- **Requirements:**
  - `app.py`
  - `requirements.txt`
  - Trained model file included in repository

---

## 📁 Project Structure
pneumonia_detection/
│
├── dataset/
│
├── train_model.py # Model training script
├── app.py # Streamlit application
├── pneumonia_model.h5 # Trained model
├── training_plot.png # Accuracy & loss curves
├── requirements.txt
└── README.md
