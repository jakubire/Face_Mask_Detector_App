import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = load_model('face_mask_cnn_model1.keras')

def process_predict_image(img):
    # Normalize the image
    img = np.array(img) / 255.0
    
    # Reshape the image for the model
    img_reshape = np.reshape(img, [1, 128, 128, 3])

    # Make a prediction
    prediction = model.predict(img_reshape, verbose=0)
    return prediction

# Streamlit app
st.title('Face Mask Detector using CNN')

# User input
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpeg', 'jpg'])
if uploaded_file is not None:
    # Load and display the image
    image = load_img(uploaded_file, target_size=(128, 128))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image and get the prediction
    prediction = process_predict_image(image)
    prediction_prob = np.max(prediction)

    if np.argmax(prediction) == 1:
        st.write(f'*Prediction*: The uploaded image is predicted to be **wearing a face mask** with probability: {prediction_prob:.2f}')
    else:
        st.write(f'*Prediction*: The uploaded image is predicted to be **not wearing a face mask** with probability: {prediction_prob:.2f}')
        
