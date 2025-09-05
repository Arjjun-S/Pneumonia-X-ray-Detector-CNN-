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

# ===== CUSTOM STYLING =====
st.set_page_config(page_title="Pneumonia Detector", page_icon="🩻", layout="centered")

# Add custom CSS to mimic your HTML theme
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
    }
    .container {
        background: white;
        padding: 20px;
        margin-top: 30px;
        border-radius: 15px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
        width: 380px;
        margin-left: auto;
        margin-right: auto;
    }
    footer {
        margin-top: 30px;
        font-size: 14px;
        color: #555;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# ===== APP HEADER =====
st.markdown('<div class="main-header">Pneumonia Detector</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ===== UPLOAD SECTION =====
st.markdown('<div class="container">', unsafe_allow_html=True)
st.write("Upload a chest X-ray image to check if it's **Normal** or **Pneumonia**.")

uploaded_file = st.file_uploader("Choose Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show preview
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess image
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

st.markdown('</div>', unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("<footer>&copy; Arjjun</footer>", unsafe_allow_html=True)
