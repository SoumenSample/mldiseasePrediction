import os
import json
import numpy as np
import base64
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import logging
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.getcwd()
MODEL_PATH = os.path.join(BASE_DIR, "model", "pest_disease_model.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model", "class_names.json")

# Load model and class names
try:
    model = load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, "r") as file:
        class_names = json.load(file)
    logger.info("Model and class names loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model, class_names = None, None

# Image preprocessing
def preprocess_image(base64_string):
    try:
        image_data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(image_data))
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or class_names is None:
        return jsonify({"error": "Model not loaded properly"}), 500
    
    data = request.get_json()
    if "image" not in data:
        return jsonify({"error": "No image data provided"}), 400
    
    img_array = preprocess_image(data["image"])
    if img_array is None:
        return jsonify({"error": "Invalid image data"}), 400
    
    try:
        predictions = model.predict(img_array)
        predicted_class = class_names[str(np.argmax(predictions))]  # Fetch label from JSON using key
        probability = float(np.max(predictions))
        
        logger.info(f"Prediction: {predicted_class}, Probability: {probability}")
        return jsonify({"class": predicted_class, "probability": probability})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
