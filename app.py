import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

# Reduce TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Load the trained CNN model
@st.cache_resource  # Cache model to avoid reloading on every run
def load_model():
    model = tf.keras.models.load_model("pneumonia_model.h5")
    return model

model = load_model()

# Streamlit app UI
st.set_page_config(page_title="Pneumonia X-ray Detector", page_icon="🩻", layout="centered")

st.title("🩻 Pneumonia X-ray Detector")
st.markdown("""
Upload a chest X-ray image, and the model will predict whether it shows **Pneumonia** or is **Normal**.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    # Preprocess the image
    img = image.convert("L").resize((150, 150))  # grayscale and resize
    img_array = np.array(img).reshape(1, 150, 150, 1) / 255.0

    # Predict button
    if st.button("Predict"):
        with st.spinner("Analyzing X-ray..."):
            prediction = model.predict(img_array)
            result = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
            confidence = float(prediction[0][0])

        # Show result
        if result == "PNEUMONIA":
            st.error(f"⚠️ Prediction: {result}\n\nConfidence: {confidence:.2f}")
        else:
            st.success(f"✅ Prediction: {result}\n\nConfidence: {confidence:.2f}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit and TensorFlow")
