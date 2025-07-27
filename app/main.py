import streamlit as  st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

working_dir = os.path.dirname(os.path.realpath(__file__))
model_path = f"{working_dir}/trained_model/train_fashion_mnist_model.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# Define the class names
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Preprocess the image
def process_image(image):
    image = Image.open(image)
    image = image.resize((28, 28))  # Resize to 28x28 pixels
    image = image.convert("L")  # Convert to grayscale
    image_arr = np.array(image) / 255.0  # Normalize
    image_arr = image_arr.reshape((1,28,28,1))
    return image_arr

# Streamlit app
st.title("Fashion Mnist App")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        resized_image = image.resize((100, 100))
        st.image(resized_image)

    with col2:
        if st.button("Predict"):
            image_arr = process_image(uploaded_file)

            result = model.predict(image_arr)
            predicted_class = np.argmax(result)

            st.success(class_names[predicted_class])
