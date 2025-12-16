# app.py

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load trained model
model = load_model('pneumonia_model.h5')

st.title("Pneumonia Detection System (AI Powered)")
st.write("Upload a chest X-ray image to detect Pneumonia or Normal lungs.")

# Upload image
uploaded_file = st.file_uploader("Choose a Chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess image
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    
    if st.button("Detect Pneumonia"):
        pred = model.predict(x)[0][0]
        confidence = pred * 100 if pred > 0.5 else (1-pred)*100
        label = "Pneumonia" if pred > 0.5 else "Normal"
        st.write(f"Prediction: **{label}**")
        st.write(f"Confidence: **{confidence:.2f}%**")
        
        # Optional probability chart
        st.bar_chart([pred, 1-pred], use_container_width=True)
