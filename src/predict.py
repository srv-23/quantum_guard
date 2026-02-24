import joblib
import pandas as pd

model = joblib.load("../models/random_forest.pkl")
scaler = joblib.load("../models/scaler.pkl")

sample = pd.read_csv("../dataset/metadata_processed.csv").drop("label",axis=1).head(1)

print(model.predict(sample))