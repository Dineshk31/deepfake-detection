import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import InceptionResNetV2

# Parameters
frame_height, frame_width = 160, 160  # Required input size for FaceNet
sequence_length = 10  # Number of frames per sequence
num_classes = 2  # Real or Fake

# Load the pre-trained FaceNet model (InceptionResNetV2)
@st.cache_resource
def load_facenet_model():
    base_model = InceptionResNetV2(include_top=False, input_shape=(frame_height, frame_width, 3), pooling='avg')
    return base_model

# Load a single video file and extract FaceNet embeddings
def load_single_video(video_file, facenet_model):
    cap = cv2.VideoCapture(video_file)
    embeddings = []
    while len(embeddings) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_height, frame_width))
        frame = frame / 255.0  # Normalize the frame
        embedding = facenet_model.predict(np.expand_dims(frame, axis=0))
        embeddings.append(embedding.flatten())
    cap.release()
    if len(embeddings) == sequence_length:
        data = np.array(embeddings)
        return data
    else:
        return None

# LSTM Model using FaceNet embeddings
def create_model():
    model = Sequential()
    
    # LSTM for temporal features
    model.add(LSTM(128, return_sequences=False, input_shape=(sequence_length, 1536)))  # Adjust input shape to match FaceNet embeddings
    model.add(Dropout(0.5))
    
    # Fully connected layers for classification
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to provide explanation based on the prediction
def generate_explanation(class_label, prediction_probabilities):
    explanations = []
    if class_label == 1:
        explanations.append("Inconsistencies detected in facial features, such as unnatural expressions or asymmetric movements.")
        explanations.append("Lighting and shading inconsistencies between the face and the rest of the scene.")
        explanations.append("Unnatural or jerky movements that do not align with expected human motion.")
        explanations.append("Blurriness or artifacts around the edges of the face, often indicating poor face-swapping.")
        explanations.append(f"Model confidence in this being fake: {prediction_probabilities[1]*100:.2f}%")
    else:
        explanations.append("No significant abnormalities detected in facial features or movements.")
        explanations.append("Consistent lighting and shading throughout the scene.")
        explanations.append("Smooth and natural movements that align with expected human motion.")
        explanations.append(f"Model confidence in this being real: {prediction_probabilities[0]*100:.2f}%")
    
    return explanations

# Streamlit UI
st.title("Deep Fake Video Detection")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.read())
    
    # Load FaceNet model
    facenet_model = load_facenet_model()
    
    # Extract embeddings
    frames = load_single_video("temp_video.mp4", facenet_model)

    if frames is not None:
        frames = np.expand_dims(frames, axis=0)  # Expand dimensions for batch processing

        # Create model
        model = create_model()
        
        # Load pre-trained weights if available
        # model.load_weights('path_to_trained_model_weights.h5')  # Uncomment and update if using saved weights

        # Predict
        prediction = model.predict(frames)
        class_label = np.argmax(prediction)
        prediction_probabilities = prediction[0]

        if class_label == 1:
            st.error("Confirmed Fake: Inconsistencies detected.")
        else:
            st.success("Confirmed Real: No abnormalities detected.")

        # Generate and display explanations
        explanations = generate_explanation(class_label, prediction_probabilities)
        st.subheader("Detailed Explanation:")
        for explanation in explanations:
            st.write("- " + explanation)
    else:
        st.warning("Insufficient frames for analysis.")