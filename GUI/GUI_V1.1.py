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
    st.image(r'C:\Users\shatn\OneDrive\Desktop\GitHubProjects\Intracranial-Tumor-Detector\GUI\AutoML Random Files\logo_without_background.png')
    st.title('Intracranial Tumor Detector')
    choice = st.radio('Navigation', ['Upload an Image', 'Result Analysis', 'Download our AI Trained Model', 'Feedback and Suggestions'])
    st.info('The ICTD project has been done by MSJS team.')


# Check if Dataset.csv Exists and Load it if it does
if os.path.exists(r'AutoML Random Files/4ClassesFirstModel.csv'): 
    df = pd.read_csv(r'AutoML Random Files/4ClassesFirstModel.csv', index_col=None)
else:
    file = st.file_uploader('Upload Your Dataset')
    if file:
        df = pd.read_csv(file, index_col=None)
        df.to_csv(r'AutoML Random Files/4ClassesFirstModel.csv', index=None)
        st.dataframe(df)
# Initialize a Session State Variable to Store the Profile Report
if 'profile_report' not in st.session_state:
    # Generate the Profile Report Once and Store it in the Session State
    profile_df = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    st.session_state.profile_report = profile_df


ModelPath = r'models'
modelname = r'C:\Users\shatn\OneDrive\Desktop\GitHubProjects\Intracranial-Tumor-Detector\GUI\models\ICTD_Trained_Model.h5'
DownloadPath = r'C:\Users\shatn\OneDrive\Desktop\GitHubProjects\Intracranial-Tumor-Detector\GUI\models\ICTD_Trained_Model.h5'
modelName3Classes = r'models/4Classes_31.Dec.2024.09.57.39.184201_32epochs.h5'


fourClassesodel = tf.keras.models.load_model(modelName3Classes, compile=False)


# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize image to match model's input shape (256x256)
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Function to classify the tumor type
def TumorType(preprocessed_image) -> None:
    # Get predictions
    yhat = fourClassesodel.predict(preprocessed_image)
    
    # Get the class with the highest prediction probability
    predicted_class = tf.argmax(yhat, axis=1).numpy()[0]
    st.write(predicted_class)
    st.write(fourClassesodel.summary())
    st.write(yhat)
    
    # Map the predicted class index to labels
    label_map = {
        0: "Glioma",
        1: "Meningioma",
        2: "No Tumor",
        3: "Pituitary"
    }
    
    # Determine the predicted label
    predicted_label = label_map.get(predicted_class, "Unknown")
        
    # Display prediction result
    st.error(f'Image has been Classified as Class {predicted_class} ({predicted_label} Detected)')


if choice == 'Upload an Image':
    st.title("Intracranial Tumor Detector")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Preprocess the image
        image = Image.open(uploaded_file)
        preprocessed_image = preprocess_image(image)


        # Make prediction with the second model (Tumor Type classification)
        if fourClassesodel is not None:
            prediction_tumor = fourClassesodel.predict(preprocessed_image)  # This is for tumor type classification
            # Get the class with the highest prediction probability
            predicted_class = np.argmax(prediction_tumor, axis=1)[0]  # Get the index of the predicted class

            label_map = {
                0: "Glioma",
                1: "Meningioma",
                2: "No Intercranial Tumor",
                3: "Pituitary"
            }

            # Display tumor type prediction
            predicted_label = label_map.get(predicted_class, "Unknown")
            if predicted_class == 2:
                st.success(f'Predicted class: {predicted_label} Detected')
            else:
                st.error(f'Predicted class: {predicted_label} Detected')

        st.image(image, caption='Uploaded Image', use_column_width=True)



if choice == 'Result Analysis':
    st.markdown("<h1 style='text-align: center;'>Analyzed Data from Model Results</h1>", unsafe_allow_html=True)
    st.markdown("<div style= 'margin-bottom: 1px;'><b>Note:</b> For better User Experience Change Appearance to Wide mode in Streamlit Settings</div>", unsafe_allow_html=True)
    st.markdown("<div style= 'margin-bottom: 1px;'><b>Note:</b> This may show results from models different than the final one</div>", unsafe_allow_html=True)

    # Display Profile Report
    st_profile_report(st.session_state.profile_report)


if choice == 'Download our AI Trained Model':
    st.markdown("<h2 style='text-align: center;'>Download ICTD (Intracranial Tumor Detector) for OLD Binary Classification Model by MSJS</h2>", unsafe_allow_html=True)

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
    with open(DownloadPath, 'rb') as f:
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