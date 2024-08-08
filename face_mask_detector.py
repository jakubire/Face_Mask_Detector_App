import streamlit as st
import tensorflow 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
import numpy as np
import matplotlib.pyplot as plt

# Load the model
model = load_model('face_mask_cnn_model1.keras')

def process_predict_image(image_path):
    # Read the image
    img = load_img(image_path, target_size=(128, 128))
    # Display the image
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    
    # Normalize the image
    img = np.array(img) / 255.0
    
    # Reshape the image for the model
    img_reshape = np.reshape(img, [1, 128, 128, 3])

    # Make a prediction
    prediction = model.predict(img_reshape, verbose =0)
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

        
