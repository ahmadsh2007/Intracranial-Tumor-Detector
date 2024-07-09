import streamlit as st
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model   # type: ignore
import tensorflow as tf

ModelPath = r'Models'
modelname_glioma = 'Glioma_16.Jun.2024.13.34.21.722415_15epochs.h5'
modelname_pituitary = 'Pituitary16.Jun.2024.14.08.07.528789_15epochs.h5'
modelname_meningioma = 'Meningioma_16.Jun.2024.13.55.00.328927_15epochs.h5'
modelname_tumor = 'Tumor16.Jun.2024.14.20.33.825138_15epochs.h5'

def preprocess_image(image):
    image = image.resize((256, 256))  # Resize image to match model's input shape (256x256)
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def load_trained_model_glioma():
    model_path = os.path.join(ModelPath, modelname_glioma)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def load_trained_model_pituitary():
    model_path = os.path.join(ModelPath, modelname_pituitary)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def load_trained_model_meningioma():
    model_path = os.path.join(ModelPath, modelname_meningioma)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def load_trained_model_tumor():
    model_path = os.path.join(ModelPath, modelname_tumor)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

model_glioma = load_trained_model_glioma()
model_pituitary = load_trained_model_pituitary()
model_meningioma = load_trained_model_meningioma()
model_tumor = load_trained_model_tumor()

# Streamlit app
st.title("Intracranial Tumor Detector")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    image = Image.open(uploaded_file)
    preprocessed_image = preprocess_image(image)
    
    # Make predictions
    if model_tumor is not None:
        prediction_tumor: float = model_tumor.predict(preprocessed_image)
        # Interpret results
        if prediction_tumor > 0.5 and prediction_tumor <= 1:
            st.error("Predicted class: Brain Tumor")
        elif prediction_tumor < 0.5 and prediction_tumor >= 0:
            st.success("Predicted class: No Tumor")
        elif prediction_tumor == 0.5:
            st.write("Neutral")
        else:
            st.write("Bugged Code?!")

    if modelname_meningioma is not None and modelname_pituitary is not None and modelname_pituitary is not None:
        # Predicting the probabilities for each type of tumor
        prediction_meningioma: float = model_meningioma.predict(preprocessed_image)
        prediction_pituitary: float = model_pituitary.predict(preprocessed_image)
        prediction_glioma: float = model_glioma.predict(preprocessed_image)

        # Writing the individual predictions
        st.write(f'{prediction_meningioma = }')
        st.write(f'{prediction_pituitary = }')
        st.write(f'{prediction_glioma = }')

        # Check if all prediction values are the same
        if prediction_meningioma < 0.5 and prediction_pituitary < 0.5 and prediction_glioma < 0.5:
            st.write(f'The highest prediction is for no Tumor with a value of less than 0.5 for all Tumor Predictions.')
        else:
            # Store predictions in a dictionary
            predictions = {
                "Meningioma": prediction_meningioma,
                "Pituitary": prediction_pituitary,
                "Glioma": prediction_glioma
            }

            # Find the maximum prediction value and its corresponding tumor type
            max_tumor = max(predictions, key=predictions.get)
            max_value = predictions[max_tumor]

            # Write the maximum prediction and its tumor type
            st.write(f'The highest prediction is for {max_tumor} with a value of {max_value}.')
        
    st.image(image, caption='Uploaded Image', use_column_width=True)