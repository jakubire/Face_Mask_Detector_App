import streamlit as st
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = load_model('face_mask_cnn_model.keras')

def process_predict_image(image):
    # Read the image
    img = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)
    img = cv2.resize(img, (150, 150))
    
    # Convert image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    # Normalize the image
    img = img / 255.0
    
    # Reshape the image for the model
    img_reshape = np.reshape(img, [1, 150, 150, 3])

    # Make a prediction
    prediction = model.predict(img_reshape)
    image_label = np.argmax(prediction)
    
    if image_label == 1:
        result = "The image is predicted to be wearing a face mask."
    else:
        result = "The image is predicted not to be wearing a face mask."
    
    return result

# Streamlit app
st.title('Face Mask Detector using CNN')

# User input
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpeg', 'jpg'])
if uploaded_file is not None:
    # Preprocess the image and get the prediction
    prediction = process_predict_image(uploaded_file)
    
    st.write(f'Prediction: {prediction}')

        
