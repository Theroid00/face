import cv2
import mediapipe as mp
import joblib
from utils import extract_features
model=joblib.load("../models/emotion_model.pkl")
mp_face_mesh=mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh()
cap=cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open Camera")
    exit()

while True:
    ret,frame=cap.read()
    if not ret:
        print("Camera not working")
        break
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    result=face_mesh.process(rgb)

    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            features=extract_features(face_landmarks)
            emotiom=model.predict([features])[0]

            cv2.putText(frame,emotiom,(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) , 2)
        cv2.imshow("Emotion Detector", frame)

        if cv2.waitKey(1) & 0xFF==27:
            break
cap.release()
cv2.destroyAllWindows()