from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__, static_folder="frontend", template_folder="frontend")
CORS(app)

# Load the trained model
MODEL_PATH = "pneumonia_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def predict_pneumonia(img_path):
    img = image.load_img(img_path, target_size=(150, 150), color_mode="grayscale")
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]
    return ("PNEUMONIA", float(prediction)) if prediction > 0.5 else ("NORMAL", float(prediction))

@app.route('/')
def home():
    # Serve the index.html
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    label, confidence = predict_pneumonia(filepath)

    os.remove(filepath)  # Clean up
    return jsonify({'prediction': label, 'confidence': round(confidence, 4)})

if __name__ == "__main__":
    app.run(debug=True)
