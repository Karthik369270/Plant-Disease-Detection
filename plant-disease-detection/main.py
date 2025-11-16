import uvicorn
import numpy as np
import tensorflow as tf
import json
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io

# Create the FastAPI app
app = FastAPI(title="Plant Disease Detection API")

# 1. Load the trained model and class names
model = tf.keras.models.load_model('plant_disease_model.h5')
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# 2. Define the image preprocessing function
def preprocess_image(image: Image.Image):
    """
    Preprocesses the image to the format the model expects.
    - Resizes to 224x224
    - Converts to a NumPy array
    - Rescales pixel values to 0-1
    - Adds a batch dimension
    """
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = image_array / 255.0  # Rescale
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# 3. Define the prediction endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Receives an image file, preprocesses it, makes a prediction,
    and returns the predicted class and confidence.
    """
    # Read the image file
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make a prediction
    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    # Get the class name using the index
    # Note: The JSON keys are strings, so we convert the index to a string
    predicted_class = class_names[str(predicted_index)]

    return {
        "predicted_class": predicted_class,
        "confidence": f"{confidence:.2%}"
    }

# This part is for running the app directly using `python main.py`
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)