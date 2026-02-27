import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "final_train.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

def load_data(filepath):
    """
    Load training data from CSV.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Training data not found at {filepath}")
        
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    """
    Separate features and target, and scale features.
    Returns X_scaled, y, and the fitted scaler.
    """
    # Define features to use (exclude IP/Port identifiers as they might lead to overfitting on specific hosts)
    # However, the prompt didn't explicitly ask to drop them, but usually for flow-based ML we drop 5-tuple identifiers.
    # Looking at previous context, we usually drop identifiers.
    # Let's drop IP/Port columns for the model features to ensure generalization.
    drop_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'label']
    
    # Check if these columns exist before dropping (in case they were not saved or named differently)
    # Based on metadata_pipeline.py, they are: 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'label'
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    y = df['label']
    
    print(f"Features selected: {list(X.columns)}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def train_model(X, y):
    """
    Train Random Forest Classifier.
    """
    print("Training Random Forest Classifier...")
    # Requirements: n_estimators=300, class_weight="balanced", random_state=42
    rf = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    return rf

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
    print("=== random_forest_training.py Started ===")
    
    # 1. Load Data
    try:
        df = load_data(DATASET_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    # 2. Print Stats
    num_samples = len(df)
    num_anomalies = df['label'].sum()
    print(f"Total Training Samples: {num_samples}")
    print(f"Anomaly Samples: {num_anomalies}")

    # 3. Preprocess
    X_scaled, y, scaler = preprocess_data(df)

    # 4. Train
    model = train_model(X_scaled, y)

    # 5. Save
    save_artifacts(model, scaler)
    
    print("=== Training Complete ===")

if __name__ == "__main__":
    main()
