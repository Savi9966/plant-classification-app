import streamlit as st
import tensorflow as tf
import numpy as np
import json
import pandas as pd
import gdown
from PIL import Image

# Download model from Google Drive
MODEL_URL = "https://drive.google.com/uc?id=1Ollcw9FIVoKABTraEhPEuxbyg2hCmtjS"
MODEL_PATH = "stacked_model.keras"
gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

LABELS_PATH = "class_labels.json"
CSV_PATH = "medicinal_uses(2).csv"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)

# Load class labels
with open(LABELS_PATH, "r") as f:
    class_labels = json.load(f)

# Load medicinal uses CSV
df = pd.read_csv(CSV_PATH, encoding="latin1")


def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to match model input size
    image = np.array(image) / 255.0   # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Streamlit UI
st.title("Plant Classification & Remedies")
st.write("Upload an image to classify and get medicinal remedies.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_index = np.argmax(prediction)
    class_name = class_labels.get(str(predicted_index), "Unknown")
    
    st.write(f"### Predicted Class: {class_name}")
    
    # Get utilities and remedies
    utilities = df.iloc[predicted_index]["utilities"]
    remedies = df.iloc[predicted_index]["Remedy"]
    st.write("### Medicinal Uses & Remedies:")
    st.write(utilities)
    st.write(remedies)
