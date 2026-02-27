import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "final_train.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "if_scaler.pkl")

def load_benign_data(filepath):
    """
    Load data and filter for benign samples only.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training data not found at {filepath}")
        
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Filter Benign (label == 0)
    df_benign = df[df['label'] == 0].copy()
    
    if len(df_benign) == 0:
        raise ValueError("No benign samples found in training data.")
        
    return df_benign

def preprocess_data(df):
    """
    Prepare features and scale them.
    Isolation Forest is unsupervised, so we drop labels.
    """
    # Define features to use (exclude identifiers and label)
    drop_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'label']
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    print(f"Features selected for Isolation Forest: {list(X.columns)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

def train_isolation_forest(X):
    """
    Train Isolation Forest Model.
    """
    print("Training Isolation Forest...")
    # Requirements: n_estimators=200, contamination="auto", random_state=42
    clf = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X)
    return clf

def save_artifacts(model, scaler):
    """
    Save trained model and scaler to disk.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Scaler saved to: {SCALER_PATH}")

def main():
    print("=== isolation_forest_training.py Started ===")
    
    # 1. Load Data
    try:
        df_benign = load_benign_data(DATASET_PATH)
    except Exception as e:
        print(e)
        return

    # 2. Print Stats
    num_samples = len(df_benign)
    print(f"Total Benign Training Samples: {num_samples}")

    # 3. Preprocess
    X_scaled, scaler = preprocess_data(df_benign)

    # 4. Train
    model = train_isolation_forest(X_scaled)

    # 5. Save
    save_artifacts(model, scaler)
    
    print("=== Training Complete ===")

if __name__ == "__main__":
    main()
