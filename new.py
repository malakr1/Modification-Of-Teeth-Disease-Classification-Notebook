import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load trained model
model = load_model(r'C:\Users\malak\Downloads\path_to_your_model.h5')

# Function to preprocess the input image
def preprocess_image(image, target_size):
   
    img = image.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to match model input shape
    img_array = img_array / 255.0  # Normalize the image (same as during training)
    return img_array

# Streamlit app
st.title("Teeth Disease Classification")
st.write("Upload an image to classify the type of teeth disease:")

uploaded_image = st.file_uploader("Choose an image...", type="jpg")
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image
    target_size = (256, 256)
    processed_image = preprocess_image(image, target_size)
    
    # Make predictions
    predictions = model.predict(processed_image)
    class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
    predicted_class = class_names[np.argmax(predictions)]
    
    st.write(f"Predicted Class: {predicted_class}")
