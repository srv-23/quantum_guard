import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load raw metadata
df = pd.read_csv("../dataset/metadata_raw.csv")

# One-hot encode TLS record type
df = pd.get_dummies(df, columns=["record_type"])

# Scale features
scaler = StandardScaler()
scaled = scaler.fit_transform(df)

processed = pd.DataFrame(scaled, columns=df.columns)

# Inject synthetic anomalies (10%)
processed["label"] = 0
anomaly_indices = np.random.choice(len(processed), size=int(0.1 * len(processed)), replace=False)
processed.loc[anomaly_indices, "label"] = 1

# Save processed dataset
processed.to_csv("../dataset/metadata_processed.csv", index=False)

# Save scaler
joblib.dump(scaler, "../models/scaler.pkl")

print("Preprocessing complete with synthetic anomalies")