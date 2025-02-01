import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf

@st.cache_resource
def loadmodel():
    model = tf.keras.models.load_model("C:/Users/Wael/Downloads/brain_tumor_detector.keras")
    return model

model = loadmodel()

st.write("""
# Brain Tumor Detection
"""
)

file = st.file_uploader("Upload your brain axis scan image here", type=["jpg", "png", "jpeg"])

def import_predict(img, model):
    # Resize to 90x90
    img = img.resize((90, 90))
    img = np.array(img)  # Convert to NumPy array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (height, width, 1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, height, width, 1)
    img = img / 255.0  # Normalize the image (assuming model was trained on normalized images)
    
    # Predict the class (binary classification, so output is a probability)
    prediction = model.predict(img)  # Model predicts the output
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    # Open the uploaded image
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    # Make the prediction
    predictions = import_predict(image, model)
    
    # Handle the binary prediction (assuming output is a probability)
    if predictions[0] > 0.5:
        label = 'Tumor'  # Prediction is greater than 0.5, so it's a tumor
    else:
        label = 'No Tumor'  # Prediction is less than 0.5, so it's no tumor
    
    # Display the result
    string = f"This image is indicating: {label}"
    st.success(string)
