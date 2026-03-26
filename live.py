import cv2
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from model import EmotionCNN

# ── Load classes ──────────────────────────────────────────────────────────────
try:
    with open('classes.txt', 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    classes = ['1','2','3','4','5','6','7','angry','disgust','fear','happy','neutral','sad','surprise']

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionCNN(num_classes=len(classes))
model.load_state_dict(torch.load('emotion_model.pth', map_location=device))
model.to(device)
model.eval()
print(f"Model loaded ({len(classes)} classes). Press 'q' to quit.\n")

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ── Face detector ─────────────────────────────────────────────────────────────
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ── Camera loop ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit(1)

last_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_crop = gray[y:y+h, x:x+w]
        face_pil  = Image.fromarray(face_crop)
        tensor    = transform(face_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(tensor)
            _, predicted = torch.max(outputs, 1)

        emotion = classes[predicted.item()]

        # Print to terminal only when emotion changes
        if emotion != last_emotion:
            print(f"Emotion detected: {emotion}")
            last_emotion = emotion

        # Draw on camera window
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if len(faces) == 0:
        last_emotion = None   # reset so next detection prints fresh

    cv2.imshow('Emotion Detection  (press q to quit)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
