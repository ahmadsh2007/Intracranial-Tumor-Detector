#from operator import index
#import plotly.express as px
#from pycaret.regression import setup, compare_models, pull, save_model, load_model
#from pydantic.v1 import BaseSettings
#from pydantic_settings import BaseSettings
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
import tensorflow as tf
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import os


with st.sidebar: 
    st.image(r'AutoML Random Files/logo_without_background.png')
    st.title('Intracranial Tumor Detector')
    choice = st.radio('Navigation', ['Upload an Image', 'Result Analysis', 'Download our AI Trained Model', 'Feedback and Suggestions'])
    st.info('The ICTD project has been done by MSJS team.')


# Check if Dataset.csv Exists and Load it if it does
if os.path.exists(r'./AutoML Random Files/dataset.csv'): 
    df = pd.read_csv(r'AutoML Random Files/dataset.csv', index_col=None)
else:
    file = st.file_uploader('Upload Your Dataset')
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv(r'AutoML Random Files/dataset.csv', index=None)
        st.dataframe(df)
# Initialize a Session State Variable to Store the Profile Report
if 'profile_report' not in st.session_state:
    # Generate the Profile Report Once and Store it in the Session State
    profile_df = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    st.session_state.profile_report = profile_df


ModelPath = r'models'
modelname = 'ICTD_Trained_Model.h5'
FullModelPath = ModelPath + '/' + modelname
modelname_glioma = 'Glioma_17.Jun.2024.10.44.42.828487_512epochs.h5'
modelname_pituitary = 'Pituitary17.Jun.2024.18.32.54.485503_32epochs.h5'
modelname_meningioma = 'Meningioma_17.Jun.2024.12.11.21.67080_512epochs.h5'


# Load the pre-trained model
def load_trained_model():
    model_path = os.path.join(ModelPath, modelname)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

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

model = load_trained_model()
model_glioma = load_trained_model_glioma()
model_pituitary = load_trained_model_pituitary()
model_meningioma = load_trained_model_meningioma()

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize image to match model's input shape (256x256)
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def TumorType(preprocessed_image) -> None:
    if modelname_meningioma is not None and modelname_pituitary is not None and modelname_pituitary is not None:
        # Predicting the probabilities for each type of tumor
        prediction_meningioma: float = model_meningioma.predict(preprocessed_image)
        prediction_pituitary: float = model_pituitary.predict(preprocessed_image)
        prediction_glioma: float = model_glioma.predict(preprocessed_image)

        # # Writing the individual predictions
        # st.write(f'{prediction_meningioma = }')
        # st.write(f'{prediction_pituitary = }')
        # st.write(f'{prediction_glioma = }')

        st.markdown("<div style= 'margin-bottom: 1px;'><b>Note:</b> Determining the type of tumor is still in its early stages and has low accuracy (18-87%) accuracy.</div>", unsafe_allow_html=True)

        # Check if all prediction values are the same
        if prediction_meningioma < 0.5 and prediction_pituitary < 0.5 and prediction_glioma < 0.5:
            st.warning(f'Unknown Brain Tumor type. Check if the image of the brain is not distorted or corrupted')
        else:
            # Store predictions in a dictionary
            predictions = {
                "Meningioma": prediction_meningioma,
                "Pituitary": prediction_pituitary,
                "Glioma": prediction_glioma
            }

            # Find the maximum prediction value and its corresponding tumor type
            TumorType = max(predictions, key=predictions.get)

            # Write the maximum prediction and its tumor type
            st.error(f'Predicted class: Intracranial Tumor ({TumorType}).')


# def custom_info(text):  this is a comment: to justify text on command line 46
#    st.markdown(
#        f"""
#        <div class="custom-info">
#            {text}
#        </div>
#        """,
#        unsafe_allow_html=True
#    )

# st.markdown(  this is a comment: to justify text on command line 46
#    """
#    <style>
#   .custom-info {
#        background-color: d9edf7;
#        color: 31708f;
#        padding: 1rem;
#        border-radius: 0.5rem;
#        text-align: justify;
#        text-justify: inter-word;
#    }
#    </style>
#    """,
#    unsafe_allow_html=True
# ) this is a comment: very important if u want to justify line 46 u must change st.info --> custom_info


# if You wan to add "What is Intracranial Tumor Detector" Then Add it Here
# Dont Forget to Add "What is Intracranial Tumor Detector" in Navigation Bar at line 72


# if You wan to add "Model Result" Then Add it Here
# Dont Forget to Add "Model Result" in Navigation Bar at line 72


if choice == 'Upload an Image':
    # Streamlit app
    st.title("Intracranial Tumor Detector")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess the image
        image = Image.open(uploaded_file)
        preprocessed_image = preprocess_image(image)
        
        # Make predictions
        if model is not None:
            prediction: float = model.predict(preprocessed_image)
            # Interpret results
            if prediction > 0.5 and prediction <= 1:
                st.success("Predicted class: No Tumor")
            elif prediction < 0.5 and prediction >= 0:
                st.error("Predicted class: Intracranial Tumor")
            elif prediction == 0.5:
                st.write("Neutral")
            else:
                st.write("Bugged Code?!")

        # Make predictions
        if model is not None:
            prediction_tumor: float = model.predict(preprocessed_image)
            # Interpret results
            if prediction_tumor > 0.5 and prediction_tumor <= 1:
                st.success("Predicted class: No Tumor Detected to Determine its Type")
            elif prediction_tumor < 0.5 and prediction_tumor >= 0:
                TumorType(preprocessed_image)
            elif prediction_tumor == 0.5:
                st.write("Neutral")
            else:
                st.write("Bugged Code?!")
        
        st.image(image, caption='Uploaded Image', use_column_width=True)


if choice == 'Result Analysis':
    st.markdown("<h1 style='text-align: center;'>Analyzed Data from Model Results</h1>", unsafe_allow_html=True)
    st.markdown("<div style= 'margin-bottom: 1px;'><b>Note:</b> For better User Experience Change Appearance to Wide mode in Streamlit Settings</div>", unsafe_allow_html=True)
    st.markdown("<div style= 'margin-bottom: 1px;'><b>Note:</b> This may show results from models different than the final one</div>", unsafe_allow_html=True)

    # Display Profile Report
    st_profile_report(st.session_state.profile_report)


if choice == 'Download our AI Trained Model':
    st.markdown("<h2 style='text-align: center;'>Download ICTD by MSJS</h2>", unsafe_allow_html=True)

    # How to Center "the div" meme
    st.markdown(
        """
        <style>
        .stDownloadButton {
            display: flex;
            justify-content: center;
        }
        .stDownloadButton > button {
            font-size: 1.25rem;
            padding: 1rem 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Open the Trained Model File
    with open(FullModelPath, 'rb') as f:
        # Center the Lovely Button
        st.markdown('<div class="stDownloadButton">', unsafe_allow_html=True)
        st.download_button('Download the Model', f, 'ICTD_Trained_Model.h5')
        st.markdown('</div>', unsafe_allow_html=True)


# Define Feedback Tab Function
# To Run Feedback Tab Function, Use the Following Line: feedback_tab() 
def feedback_tab():
    st.title("Feedback and Suggestions")
    st.markdown("<h8>Please Contact us at msjs.yu@gmail.com</h8>", unsafe_allow_html=True)
    feedback_text = st.text_area("Please provide your feedback here:")
    submitted = st.button("Submit Feedback")

    if submitted:
        # Open the Feedback File and Count Existing Messages
        try:
            with open("Feedback_and_Suggestions.txt", "r", encoding="utf-8") as textfile:
                num_messages = sum(1 for line in textfile if line.strip() == "*" * 96) + 1
        except FileNotFoundError:
            num_messages = 1

        # Write the New Feedback with a Message Number
        with open("Feedback_and_Suggestions.txt", "a", encoding="utf-8") as textfile:
            textfile.write(f"Feedback {num_messages}:\n")
            textfile.write(f"\n{feedback_text}\n")
            textfile.write("\n" + "*" * 96 + "\n")  # split every feedback

        st.success("Feedback submitted successfully!")


if choice == 'Feedback and Suggestions':
    feedback_tab()