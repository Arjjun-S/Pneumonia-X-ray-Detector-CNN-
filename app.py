import os
from flask import Flask, request, jsonify
from flask_cors import CORS  # <-- ADD THIS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Reduce TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = Flask(__name__)

# ✅ Allow requests ONLY from your Vercel frontend
CORS(app, resources={r"/*": {"origins": "https://pneumonia-x-ray-detector-cnn.vercel.app"}})

# Load your trained model
model = tf.keras.models.load_model("pneumonia_model.h5")

@app.route("/")
def home():
    return "Pneumonia Prediction API is running!"

@app.route("/healthz")
def health_check():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']

    # Preprocess image
    img = Image.open(io.BytesIO(file.read())).convert("L").resize((150, 150))
    img_array = np.array(img).reshape(1, 150, 150, 1) / 255.0

    # Make prediction
    prediction = model.predict(img_array)
    result = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"

    return jsonify({
        "prediction": result,
        "confidence": float(prediction[0][0])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
