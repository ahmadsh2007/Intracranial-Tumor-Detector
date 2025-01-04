#from operator import index
#import plotly.express as px
#from pycaret.regression import setup, compare_models, pull, save_model, load_model
#from pydantic.v1 import BaseSettings
#from pydantic_settings import BaseSettings
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import tensorflow as tf
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os



with st.sidebar: 
    st.image(r'C:\Users\shatn\OneDrive\Desktop\GitHubProjects\Intracranial-Tumor-Detector\GUI\AutoML Random Files\logo_without_background.png')
    st.title('Intracranial Tumor Detector')
    choice = st.radio('Navigation', ['Upload an Image', 'Pre-Alpha Version','Result Analysis', 'Download our AI Trained Model', 'Feedback and Suggestions'])
    st.info('The ICTD project has been done by MSJS team.')


# Check if the two datasets exist and load them if they do
tumorCSV = r'C:\Users\shatn\OneDrive\Desktop\GitHubProjects\Intracranial-Tumor-Detector\V1.2\TumorClassesFirstModel.csv'
mainCSV = r'C:\Users\shatn\OneDrive\Desktop\GitHubProjects\Intracranial-Tumor-Detector\V1.2\MainClassesFirstModel.csv'

if os.path.exists(tumorCSV) and os.path.exists(mainCSV):
    dfT = pd.read_csv(tumorCSV, index_col=None)
    dfM = pd.read_csv(mainCSV, index_col=None)
else:
    file = st.file_uploader('Upload Your Dataset')
    if file:
        dfT = pd.read_csv(file, index_col=None)
        dfT.to_csv(tumorCSV, index=None)
        st.dataframe(dfT)

# Initialize a Session State Variable to Store the Profile Reports for both files
if 'profileReportFileTumor' not in st.session_state:
    # Generate the Profile Report for the first file and Store it in the Session State
    profile_dfT = ProfileReport(dfT, title="Pandas Profiling Report - Tumor Classes", explorative=True)
    st.session_state.profileReportFileTumor = profile_dfT

if 'profileReportFileMain' not in st.session_state:
    # Generate the Profile Report for the second file and Store it in the Session State
    profile_dfM = ProfileReport(dfM, title="Pandas Profiling Report - Main Classes", explorative=True)
    st.session_state.profileReportFileMain = profile_dfM

modelname = r'C:\Users\shatn\OneDrive\Desktop\GitHubProjects\Intracranial-Tumor-Detector\V1.2\models\MainClasses_01.Jan.2025.19.53.15.387435_32epochs.h5'
DownloadPath = r'C:\Users\shatn\OneDrive\Desktop\GitHubProjects\Intracranial-Tumor-Detector\V1.2\models\MainClasses_01.Jan.2025.19.53.15.387435_32epochs.h5'
modelName3Classes = r'C:\Users\shatn\OneDrive\Desktop\GitHubProjects\Intracranial-Tumor-Detector\V1.2\models\TumorClasses_01.Jan.2025.19.53.15.387435_32epochs.h5'


MainModel = tf.keras.models.load_model(modelname, compile=False)
fourClassesModel = tf.keras.models.load_model(modelName3Classes, compile=False)


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
    yhat = fourClassesModel.predict(preprocessed_image)
    
    # Get the class with the highest prediction probability
    predicted_class = tf.argmax(yhat, axis=1).numpy()[0]
    st.write(predicted_class)
    st.write(fourClassesModel.summary())
    st.write(yhat)
    
    # Map the predicted class index to labels
    label_map = {
        0: "Glioma",
        1: "Meningioma",
        2: "Pituitary"
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

        # Make prediction with the Main model (No Tumor/Tumor classification)
        if MainModel is not None:
            prediction: float = MainModel.predict(preprocessed_image)
            # Get the class with the highest prediction probability
            MainPredictedClass = np.argmax(prediction, axis=1)[0]  # Get the index of the predicted class

            MainLabelMap = {
                0: "No Intercranial Tumor",
                1: "Intercranial Tumor",
            }


            # Display tumor type prediction
            MainPredictedLabel = MainLabelMap.get(MainPredictedClass, "Unknown")
            if MainPredictedClass == 0:
                st.success(f'Predicted class: {MainPredictedLabel} Detected')
            elif MainPredictedClass == 1:
                # Make prediction with the second model (Tumor Type classification)
                if fourClassesModel is not None:
                    prediction_tumor = fourClassesModel.predict(preprocessed_image)  # This is for tumor type classification

                    # Get the class with the highest prediction probability
                    TumorPredictedClass = np.argmax(prediction_tumor, axis=1)[0]  # Get the index of the predicted class

                    TumorLapelMap = {
                        0: "Glioma",
                        1: "Meningioma",
                        2: "Pituitary"
                    }

                    # Display tumor type prediction
                    TumorPredictedLabel = TumorLapelMap.get(TumorPredictedClass, "Unknown")
                    if TumorPredictedClass == 0:
                        st.error(f'Predicted class: {TumorPredictedLabel} Detected')
                    elif TumorPredictedClass == 1:
                        st.error(f'Predicted class: {TumorPredictedLabel} Detected')
                    elif TumorPredictedClass == 2:
                        st.error(f'Predicted class: {TumorPredictedLabel} Detected')
                    else:
                        st.warning(f'Unknown Brain Tumor type. Check if the image of the brain is not distorted or corrupted')
            else:
                st.warning(f'Unknown Brain Tumor type. Check if the image of the brain is not distorted or corrupted')

        st.image(image, caption='Diagnostic Medical Image', use_container_width=True)


def HeatmapCaller(ImagePath, model):
    st.title("Grad-CAM Heatmap")
        
    # Load and preprocess the image
    img = Image.open(ImagePath)
    PI = preprocess_image(img)
    img_array = np.array(PI)

    # Add 'None' for the batch dimension (axis=0)
    img_array = np.expand_dims(img_array, axis=0)  # Shape will now be (1, 256, 256, 3)

    # Ensure the shape is correct (remove any extra dimensions, if present)
    img_array = np.squeeze(img_array)  # Removes any single-dimensional entries, which should leave (1, 256, 256, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize the image (assuming this is how your model was trained)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    
    def generate_gradcam_heatmap(model, img_array, target_layer_name):
        # Get the last convolutional layer
        last_conv_layer = model.get_layer(target_layer_name)
        
        # Create a model that outputs both the predictions and the last conv layer's output
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [last_conv_layer.output, model.output]
        )
        
        # Model prediction and class index
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            tape.watch(conv_outputs)
            class_idx = tf.argmax(predictions[0])
            class_channel = predictions[:, class_idx]
        
        # Gradient calculation
        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        # Compute weighted sum of feature maps
        conv_outputs = conv_outputs[0].numpy()
        pooled_grads = pooled_grads.numpy()
        for i in range(conv_outputs.shape[-1]):
            conv_outputs[:, :, i] *= pooled_grads[i]
        
        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)  # Remove negative values
        heatmap = heatmap / np.max(heatmap)  # Normalize to [0, 1]
        
        return heatmap

    # Generate the heatmap for the last convolutional layer (update layer name if needed)
    heatmap = generate_gradcam_heatmap(model, img_array, 'conv2d_2')  # Replace 'conv2d_2' with your layer name
    st.write("Heatmap generated successfully.")

    # Resize the heatmap to match the original image size
    img_width, img_height = img.size  # Now we get dimensions from the original PIL image (img)
    heatmap_resized = cv2.resize(heatmap, (img_width, img_height))

    # Convert heatmap to color map (jet)
    heatmap_colored = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)

    # Convert img_array (which is now (1, 256, 256, 3)) to (256, 256, 3)
    img_resized = np.squeeze(img_array, axis=0)  # Shape will now be (256, 256, 3)

    # Resize img_resized to match the size of heatmap_colored
    img_resized = cv2.resize(img_resized, (heatmap_colored.shape[1], heatmap_colored.shape[0]))

    # Now perform the element-wise addition for superimposing the heatmap
    superimposed_img = heatmap_colored * 0.4 + img_resized
    superimposed_img = np.uint8(superimposed_img)

    # Display the result in Streamlit
    st.image(superimposed_img, caption="Grad-CAM Heatmap Overlay", use_container_width =True)


if choice == 'Pre-Alpha Version':
    PreAlphaCol1, PreAlphaCol2 = st.columns(2)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    with PreAlphaCol1:
        st.title("Intracranial Tumor Detector")

        if uploaded_file is not None:
            # Preprocess the image
            image = Image.open(uploaded_file)
            preprocessed_image = preprocess_image(image)

            # Make prediction with the Main model (No Tumor/Tumor classification)
            if MainModel is not None:
                prediction: float = MainModel.predict(preprocessed_image)
                # Get the class with the highest prediction probability
                MainPredictedClass = np.argmax(prediction, axis=1)[0]  # Get the index of the predicted class

                MainLabelMap = {
                    0: "No Intercranial Tumor",
                    1: "Intercranial Tumor",
                }


                # Display tumor type prediction
                MainPredictedLabel = MainLabelMap.get(MainPredictedClass, "Unknown")
                if MainPredictedClass == 0:
                    st.success(f'Predicted class: {MainPredictedLabel} Detected')
                elif MainPredictedClass == 1:
                    # Make prediction with the second model (Tumor Type classification)
                    if fourClassesModel is not None:
                        prediction_tumor = fourClassesModel.predict(preprocessed_image)  # This is for tumor type classification

                        # Get the class with the highest prediction probability
                        TumorPredictedClass = np.argmax(prediction_tumor, axis=1)[0]  # Get the index of the predicted class

                        TumorLapelMap = {
                            0: "Glioma",
                            1: "Meningioma",
                            2: "Pituitary"
                        }

                        # Display tumor type prediction
                        TumorPredictedLabel = TumorLapelMap.get(TumorPredictedClass, "Unknown")
                        if TumorPredictedClass == 0:
                            st.error(f'Predicted class: {TumorPredictedLabel} Detected')
                        elif TumorPredictedClass == 1:
                            st.error(f'Predicted class: {TumorPredictedLabel} Detected')
                        elif TumorPredictedClass == 2:
                            st.error(f'Predicted class: {TumorPredictedLabel} Detected')
                        else:
                            st.warning(f'Unknown Brain Tumor type. Check if the image of the brain is not distorted or corrupted')
                else:
                    st.warning(f'Unknown Brain Tumor type. Check if the image of the brain is not distorted or corrupted')

            st.image(image, caption='Diagnostic Medical Image', use_container_width=True)


    with PreAlphaCol2:
        if uploaded_file is not None:
            HeatmapCaller(uploaded_file, fourClassesModel)


if choice == 'Result Analysis':
    col1, col2 = st.columns(2)

    # Content for the first column
    with col1:
        st.markdown("<h1 style='text-align: center;'>Main Classes Analysis</h1>", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 1px;'><b>Note:</b> This analysis pertains to the Main Classes dataset</div>", unsafe_allow_html=True)

        # Display Profile Report for the first file in the first column
        st_profile_report(st.session_state.profileReportFileMain)

    # Content for the second column
    with col2:
        st.markdown("<h1 style='text-align: center;'>Tumor Classes Analysis</h1>", unsafe_allow_html=True)
        st.markdown("<div style='margin-bottom: 1px;'><b>Note:</b> This analysis pertains to the Tumor Classes dataset</div>", unsafe_allow_html=True)

        # Display Profile Report for the second file in the second column
        st_profile_report(st.session_state.profileReportFileTumor)


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