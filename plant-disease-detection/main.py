import gradio as gr
import numpy as np
import tensorflow as tf
import json
from PIL import Image

# 1. Load the trained model and class names
# We put this outside the function so it only loads once
model = tf.keras.models.load_model('plant_disease_model.h5')
with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# 2. Define the image preprocessing function (same as yours)
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
    
    # Handle transparent (RGBA) images
    if image_array.shape[2] == 4:
        image_array = image_array[..., :3]
        
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# 3. Define the prediction function for Gradio
def predict(image: Image.Image):
    """
    Receives a PIL image from Gradio, preprocesses it, makes a
    prediction, and returns a dictionary for the Label output.
    """
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make a prediction
    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))

    # Get the class name
    predicted_class = class_names[str(predicted_index)]

    # Format the output for Gradio's Label component
    return {predicted_class: confidence}

# 4. Create and launch the Gradio Interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Plant Leaf Image"),
    outputs=gr.Label(num_top_classes=1, label="Prediction"),
    title="Plant Disease Detector",
    description="Upload an image of a plant leaf to detect its disease. The model will return the predicted class and confidence.",
    examples=[
        ["example_leaf_1.jpg"], # You can add example images to your repo
        ["example_leaf_2.jpg"]
    ]
)

# Launch the app (this is all you need for Hugging Face)
iface.launch()
