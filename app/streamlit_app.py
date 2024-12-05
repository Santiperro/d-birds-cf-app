import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


@st.cache_resource
def load_trained_model(model_path):
    return load_model(model_path)


def preprocess_image(image, target_size):
    image = image.resize(target_size)
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)


def predict_image(model, image, class_names):
    processed_image = preprocess_image(image, model.input_shape[1:3])
    prediction = model.predict(processed_image)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence


st.title("Сlassifier of domestic birds (baby chicken, cock, duck, goose, \
    hen, ostrich, turkey)")

MODEL_PATH = "models/final_model_99.h5"
CLASS_NAMES = ["baby chicken", "cock", "duck", "goose", 
               "hen", "ostrich", "turkey"]
model = load_trained_model(MODEL_PATH)  

uploaded_files = st.file_uploader("Upload Images", 
                                  type=["jpg", "jpeg", "png"], 
                                  accept_multiple_files=True)

if uploaded_files:
    st.write("### Results")
    for uploaded_file in uploaded_files:
        if uploaded_file.type not in ["image/jpeg", "image/png"]:
            st.error(f"{uploaded_file.name} has invalid file type. Please upload jpg or png files")
            continue
        
        image = Image.open(uploaded_file)
        
        predicted_class, confidence = predict_image(model, image, CLASS_NAMES)
        
        st.image(image, use_container_width=True)
        st.write(f"**Prediction:** {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2%}")
        st.write("---")