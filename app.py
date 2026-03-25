import streamlit as st
import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from model import EmotionCNN
import os

st.set_page_config(page_title="Emotion Detector", layout="wide")

@st.cache_resource
def load_model():
    # Load class labels
    try:
        with open('classes.txt', 'r') as f:
            classes = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EmotionCNN(num_classes=len(classes))
    if os.path.exists('emotion_model.pth'):
        model.load_state_dict(torch.load('emotion_model.pth', map_location=device))
        model.to(device)
        model.eval()
    else:
        st.warning("Model file 'emotion_model.pth' not found. Please train the model first by running `python train.py`.")
    return model, classes, device

def predict_emotion(image, model, classes, device):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs.data, 1)

    return classes[predicted.item()]

st.title("Emotion Recognition from Facial Images")
model, classes, device = load_model()

# Create haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

tab1, tab2 = st.tabs(["Upload Photo", "Live Camera"])

with tab1:
    st.header("Upload a Photo")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Convert to cv2 format to detect faces
        img_array = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            st.warning("No face detected in the image.")
            st.image(image, caption="Uploaded Image", use_container_width=True)
        else:
            img_with_boxes = img_array.copy()
            for (x, y, w, h) in faces:
                cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)

                # Crop face
                face = gray[y:y+h, x:x+w]
                face_pil = Image.fromarray(face)

                # Predict
                emotion = predict_emotion(face_pil, model, classes, device)

                cv2.putText(img_with_boxes, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            st.image(img_with_boxes, caption="Detected Faces and Emotions", use_container_width=True)

with tab2:
    st.header("Live Camera Emotion Detector")
    st.write("Click 'Start' to begin the webcam stream and detect emotions in real-time.")

    # We use st.camera_input for capturing from webcam in Streamlit apps
    camera_image = st.camera_input("Take a picture to detect emotion")

    if camera_image is not None:
        image = Image.open(camera_image)
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        img_with_boxes = img_array.copy()

        if len(faces) == 0:
            st.warning("No face detected.")
        else:
            for (x, y, w, h) in faces:
                cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
                face = gray[y:y+h, x:x+w]
                face_pil = Image.fromarray(face)

                emotion = predict_emotion(face_pil, model, classes, device)
                cv2.putText(img_with_boxes, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        st.image(img_with_boxes, caption="Live Detection Results", use_container_width=True)
