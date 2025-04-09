import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- Configuration ---
MODEL_PATH = './models/animal_model.h5'
IMG_WIDTH, IMG_HEIGHT = 224, 224
CLASS_NAMES = ["Class1", "Class2", "...", "Class80"] # Replace with your actual class names

# --- Load Model ---
@st.cache_resource # Cache the model loading
def load_keras_model(model_path):
    """Loads the Keras model from the specified path."""
    if not os.path.exists(model_path):
        st.error(f"Error: Model file not found at {model_path}")
        st.stop()
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_keras_model(MODEL_PATH)

# --- Image Preprocessing ---
def preprocess_image(image):
    """Preprocesses the uploaded image."""
    img = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Create a batch
    return img_array

# --- Streamlit App ---
st.set_page_config(page_title="Animal Detector", layout="centered")
st.title("🐾 Animal Detector 📸")
st.write("Upload an image, and the model will predict the animal!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess and Predict
    st.write("")
    st.write("Classifying...")

    try:
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction, axis=1)[0]

        if predicted_class_index < len(CLASS_NAMES):
            predicted_class_name = CLASS_NAMES[predicted_class_index]
            confidence = np.max(prediction) * 100
            st.success(f"Prediction: **{predicted_class_name}** (Confidence: {confidence:.2f}%)")
        else:
            st.error("Error: Predicted class index is out of bounds for the provided CLASS_NAMES list.")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

else:
    st.info("Please upload an image file.")

st.sidebar.header("About")
st.sidebar.info(
    "This app uses a trained deep learning model to predict the type of animal "
    "in an uploaded image. The model was trained on the Animals dataset."
    f"Model expected input size: ({IMG_WIDTH}x{IMG_HEIGHT})"
)

# Optional: Add instructions on how to find class names if needed
st.sidebar.markdown("---")
st.sidebar.warning("**Developer Note:** Ensure the `CLASS_NAMES` list in `app.py` matches the training data order.") 