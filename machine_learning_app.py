import joblib
import os
from io import BytesIO
import numpy as np
import requests
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

# Get the absolute path of the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "iris_model.joblib")

# Load the model
model = joblib.load("/Users/jatinjohry/Desktop/hackathon/models/iris_model.joblib")

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

app = FastAPI()

@app.get("/")
def read_root():
    return {"result": "Hello World!"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.post("/predict")
def predict_iris_species(features: IrisFeatures):
    # Extract features into a list in the correct order
    feature_array = [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]
    
    # Predict the class
    prediction = model.predict(feature_array)
    
    # Translate prediction into species name
    species = ["setosa", "versicolor", "virginica"]
    predicted_species = species[prediction[0]]
    
    return {"predicted_species": predicted_species}

@app.get("/classify")
def classify_image(url: str):
    try:
        # Download the image from the URL
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        img = Image.open(BytesIO(response.content))

        # Resize image to match model input shape
        img = img.resize((224, 224))  # Adjust size according to your model's input size

        # Convert image to numpy array and normalize pixel values
        img_array = np.array(img) / 255.0

        # Expand dimensions to match model input shape
        img_array = np.expand_dims(img_array, axis=0)

        # Simulate a model prediction (since your model isn't defined here)
        predicted_class = np.random.randint(0, 2)

        return {"prediction": "cat" if predicted_class == 0 else "dog"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to download image: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
