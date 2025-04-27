
import logging
from io import BytesIO

import torch
import requests

from fastapi import FastAPI
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification

# Set Logging Level
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# Initialize model and image processor
try:
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-18")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

app = FastAPI()

@app.get("/")
def read_root():
    return {"result": "Hello World!"}

@app.get("/classify_image")
def classify_image(url: str):
    try:
        response = requests.get(url, timeout=10)  # Added timeout to avoid hanging requests
        response.raise_for_status()  # Ensure request was successful
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching image: {e}")
        return {"error": "Failed to fetch image"}

    try:
        image = Image.open(BytesIO(response.content)).convert("RGB")  # Convert to RGB for consistency
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return {"error": "Invalid image format"}

    try:
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label = logits.argmax(-1).item()
        result = model.config.id2label[predicted_label].split(',')
        logging.info(result)
        return {"result": result}
    except Exception as e:
        logging.error(f"Error during classification: {e}")
        return {"error": "Classification failed"}
