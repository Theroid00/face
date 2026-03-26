import cv2
import mediapipe as mp
import csv 
from utils import extract_features

emotion=input("Enter emotion label(happy/sad/angry): ")

mp_face_mesh= mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh()

cap=cv2.VideoCapture(0)

with open("../data/dataset.csv","a", newline="") as f:
    writer=csv.writer(f)
    while True:
        ret, frame= cap.read()
        rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result=face_mesh.process(rgb)
        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                features=extract_features(face_landmarks)

                writer.writerow(list(features)+[emotion])
        cv2.imshow("COllecting Data", frame)
        if cv2.waitKey(1) & 0xFF==27:
            break
cap.release()
cv2.destroyAllWindows()

