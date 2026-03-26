import numpy as np
def extract_features(face_landmarks):
    features=[]
    #mouth corners
    left=face_landmarks.landmark[61]
    right=face_landmarks.landmark[291]
    #eye
    left_eye_top=face_landmarks.landmark[159]
    right_eye_top=face_landmarks.landmark[145]
    #eyebrow
    eyebrow=face_landmarks.landmark[70]
    eye=face_landmarks.landmark[33]

    #features
    mouth_width=abs(left.x-right.x)
    eye_open=abs(left_eye_top.y-right_eye_top.y)
    eyebrow_height=abs(eyebrow.y-eye.y)

    features.extend([mouth_width,eye_open,eyebrow_height])
    return np.array(features)