import streamlit as st
import tensorflow as tf
import os
import cv2
from PIL import Image, ImageOps
import numpy as np

# Define the path to the model
model_path = '/content/drive/My Drive/Final Exam - Emtech 2/cifar10_model.h5'

@st.cache_resource
def load_model(model_path):
    # Debugging statement to verify the path
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return None
    # Load the model
    model = tf.keras.models.load_model(model_path)
    return model

# Load the model
model = load_model(model_path)
if model is None:
    st.stop()  # Stop execution if the model couldn't be loaded

st.write("# CIFAR10 Detection System")

# File uploader
file = st.file_uploader("Insert Image", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (32, 32)
    image = ImageOps.fit(image_data, size)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...] / 255.0  # Normalize the image
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    string = 'OUTPUT : ' + class_names[np.argmax(prediction)]
    st.success(string)
