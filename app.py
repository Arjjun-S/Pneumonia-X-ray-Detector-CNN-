import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Reduce TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load model once and cache
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pneumonia_model.h5")

model = load_model()

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Pneumonia Detector", page_icon="🩻", layout="centered")

# ===== CUSTOM STYLING =====
st.markdown("""
    <style>
    body {
        background-color: #f4fef4;
        font-family: Arial, sans-serif;
    }
    .main-header {
        background-color: #2e7d32;
        color: white;
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        font-size: 24px;
        margin-bottom: 20px;
    }
    footer {
        margin-top: 30px;
        font-size: 14px;
        color: #555;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown('<div class="main-header">Pneumonia Detector</div>', unsafe_allow_html=True)

st.write("Upload a chest X-ray image to check if it's **Normal** or **Pneumonia**.")

# ===== FILE UPLOADER =====
uploaded_file = st.file_uploader("Choose Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess the image
    img = image.convert("L").resize((150, 150))  # grayscale + resize
    img_array = np.array(img).reshape(1, 150, 150, 1) / 255.0

    # Predict button
    if st.button("Upload & Predict"):
        with st.spinner("Processing image..."):
            prediction = model.predict(img_array)
            result = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
            confidence = float(prediction[0][0])

        # Display result
        if result == "PNEUMONIA":
            st.error(f"**Prediction:** {result}\n\n**Confidence:** {confidence:.2f}")
        else:
            st.success(f"**Prediction:** {result}\n\n**Confidence:** {confidence:.2f}")

# ===== FOOTER =====
st.markdown("<footer>&copy; Arjjun S </footer>", unsafe_allow_html=True)
