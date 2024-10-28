import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
import tempfile

# Parameters
frame_height, frame_width = 64, 64
sequence_length = 10
num_classes = 2

# Define a model that accepts video input
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv3D(32, kernel_size=(3, 3, 3), activation='relu', input_shape=(sequence_length, frame_height, frame_width, 3)),  # 10 frames of 64x64 RGB images
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
        tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation='relu'),
        tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Output layer for classification
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_single_video(video_file):
    cap = cv2.VideoCapture(video_file)
    frames = []
    while len(frames) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (frame_height, frame_width))
        frames.append(frame)
    cap.release()
    if len(frames) == sequence_length:
        data = np.array(frames)
        return data
    else:
        return None

def load_pretrained_model():
    model_path = r"C:\Users\hp\OneDrive\Desktop\New folder\Lair\StreamlitApp\my_model.h5"
    model = tf.keras.models.load_model(model_path)
    return model

def generate_explanation(class_label):
    explanations = {
        0: "The video appears to be real. The model did not detect significant inconsistencies in facial features or movements.",
        1: "The video is classified as fake. The model detected possible inconsistencies in facial features or movements, which are often indicative of deepfakes."
    }
    return explanations.get(class_label, "Unable to provide an explanation.")

# Streamlit Interface
st.title("Deepfake Detection")

# Upload the video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_video_file:
        tmp_video_file.write(uploaded_file.read())
        tmp_video_file_path = tmp_video_file.name
    
    st.video(tmp_video_file_path)

    # Load the video
    frames = load_single_video(tmp_video_file_path)

    if frames is not None:
        st.write(f"Frames shape: {frames.shape}")  # Debugging: Check frames shape

        # Process frames for prediction
        model = create_model()  # Create the model here
        prediction = model.predict(np.expand_dims(frames, axis=0))  # Add batch dimension
        class_label = np.argmax(prediction)  # Get the class with the highest probability

        # Generate an explanation based on the class label
        explanation = generate_explanation(class_label)

        # Display the prediction and explanation
        st.write(f"Prediction: {'Real' if class_label == 0 else 'Fake'}")
        st.write(f"Explanation: {explanation}")
    else:
        st.write("Not enough frames to make a prediction. Ensure the video has enough frames.")