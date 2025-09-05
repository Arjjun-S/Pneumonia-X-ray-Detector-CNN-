from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load the trained CNN model
MODEL_PATH = "pneumonia_model.h5"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")


def predict_pneumonia(img_path):
    """
    Takes an image path, preprocesses it, and returns prediction + confidence score
    """
    img = image.load_img(img_path, target_size=(150, 150), color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    # Prediction logic
    if prediction > 0.5:
        return "PNEUMONIA", float(prediction)
    else:
        return "NORMAL", float(prediction)

# Root route
@app.route('/')
def home():
    return "Pneumonia Prediction API is running!"

# Health check route for Render
@app.route('/healthz')
def health_check():
    return jsonify({"status": "ok"}), 200

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    try:
        label, confidence = predict_pneumonia(filepath)

        os.remove(filepath)

        return jsonify({
            'prediction': label,
            'confidence': round(confidence, 4)
        })

    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
