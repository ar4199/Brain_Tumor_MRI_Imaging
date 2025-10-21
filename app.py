
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# App config
st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")

# Load model once (cached)
@st.cache_resource
def load_model_once():
    return load_model("best_custom_cnn_model.h5")  # Change to your model file if needed

model = load_model_once()
class_labels = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# Title
st.title("🧠 Brain Tumor MRI Classifier")
st.markdown("Upload an MRI image below to detect the **type of brain tumor**.")

# File uploader
uploaded_file = st.file_uploader("🩺 Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded MRI Image", use_container_width=True)

    # Preprocess and predict
    with st.spinner("Analyzing image..."):
        img = image.resize((224, 224))   # ResNet50 / CNN input size
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)[0]
        predicted_index = np.argmax(predictions)
        predicted_label = class_labels[predicted_index]
        confidence = predictions[predicted_index]

    # Show results
    st.markdown("---")
    st.subheader("🔍 Prediction Result")
    st.markdown(f"**Tumor Type:** `{predicted_label.upper()}`")
    st.progress(int(confidence * 100))
    st.markdown(f"**Confidence Score:** `{confidence*100:.2f}%`")

    # Show probabilities
    st.markdown("#### 📊 Class Probabilities")
    prob_data = {label: f"{prob*100:.2f}%" for label, prob in zip(class_labels, predictions)}
    st.json(prob_data)

