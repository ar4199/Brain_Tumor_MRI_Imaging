
import streamlit as st
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array

st.title("MRI Image Classification")

def load_model_once():
    return load_model("best_custom_cnn_model.h5")

model = load_model_once()
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

st.markdown("Upload an MRI image below to detect the **type of brain tumor**.")

# File uploader
uploaded_file = st.file_uploader("🩺 Upload MRI Image", type=["jpg", "jpeg", "png"])

# Image display and prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        # Preprocess
        img = image.resize((224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_label = class_labels[predicted_index]
        confidence = predictions[predicted_index]

    # Display results
    st.markdown("---")
    st.subheader("Prediction Result")
    st.markdown(f"**Tumor Type:** `{predicted_label.upper()}`")
    st.progress(float(confidence))
    st.markdown(f"**Confidence Score:** `{confidence*100:.2f}%`")

    # Optional: Show all class probabilities
    st.markdown("#### Class Probabilities")
    prob_data = {label: f"{prob*100:.2f}%" for label, prob in zip(class_labels, predictions)}
    st.json(prob_data)
