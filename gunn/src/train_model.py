import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

os.makedirs("../models", exist_ok=True)

data = pd.read_csv("../data/dataset.csv", header=None)

X = data.iloc[:, :-1] 
y = data.iloc[:, -1]  

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "../models/emotion_model.pkl")

print("Model trained and saved to models/emotion_model.pkl")