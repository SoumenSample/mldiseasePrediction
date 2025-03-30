import os
import json
import numpy as np
import base64
import requests
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and class names
MODEL_PATH = "model/pest_disease_model.h5"
CLASS_NAMES_PATH = "model/class_names.json"

try:
    model = load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as file:
        class_names = json.load(file)
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model, class_names = None, None

# Function to download image from URL
def download_image(image_url):
    try:
        response = requests.get(image_url, timeout=5)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return img
        else:
            logger.error(f"❌ Failed to download image. Status Code: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"❌ Error downloading image: {e}")
        return None

# Function to process base64 image
def decode_base64_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(image_data))
        return img
    except Exception as e:
        logger.error(f"❌ Error decoding base64 image: {e}")
        return None

# Preprocess image for model
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or class_names is None:
        return jsonify({"error": "Model not loaded properly"}), 500
    
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    img = None

    # Process image from URL
    if "image_url" in data:
        img = download_image(data["image_url"])
    
    # Process image from base64
    elif "image" in data:
        img = decode_base64_image(data["image"])
    
    else:
        return jsonify({"error": "No valid image data provided"}), 400

    if img is None:
        return jsonify({"error": "Failed to process image"}), 400

    # Preprocess and make prediction
    img_array = preprocess_image(img)
    try:
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        probability = float(np.max(predictions))
        
        logger.info(f"✅ Prediction: {predicted_class}, Probability: {probability}")
        return jsonify({"class": predicted_class, "probability": probability})
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
