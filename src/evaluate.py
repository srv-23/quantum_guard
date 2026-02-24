import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score

df = pd.read_csv("../dataset/metadata_processed.csv")

X = df.drop("label",axis=1)
y = df["label"]

model = joblib.load("../models/random_forest.pkl")

pred = model.predict(X)

print("Accuracy:",accuracy_score(y,pred))
print("Precision:",precision_score(y,pred))
print("Recall:",recall_score(y,pred))