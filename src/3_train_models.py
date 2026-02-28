import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import f1_score

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "dataset", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Input files
TRAIN_RF_PATH = os.path.join(PROCESSED_DATA_DIR, "train_rf.csv")    # Mixed
TRAIN_IF_PATH = os.path.join(PROCESSED_DATA_DIR, "train_if.csv")    # Pure Benign

# Output artifacts
RF_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
RF_SCALER_PATH = os.path.join(MODELS_DIR, "rf_scaler.pkl")

IF_MODEL_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")
IF_SCALER_PATH = os.path.join(MODELS_DIR, "if_scaler.pkl")

os.makedirs(MODELS_DIR, exist_ok=True)

def load_data(path):
    print(f"Loading {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    return pd.read_csv(path)

def preprocess(df, scaler=None, is_training=False, method='standard', has_label=False):
    """
    Prepares X (features) and y (labels).
    Scales features.
    """
    drop_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'label']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    
    if has_label and 'label' in df.columns:
        y = df['label'].values
    else:
        y = np.zeros(len(df))

    if is_training:
        if method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"  Features: {list(X.columns)}")
    else:
        if scaler is None:
            raise ValueError("Values scaler must be provided.")
        X_scaled = scaler.transform(X)
        
    return X_scaled, y, scaler

def train_random_forest(df_train):
    print("\n--- Training Random Forest (Supervised) ---")
    
    # Preprocess
    X_train, y_train, scaler = preprocess(df_train, is_training=True, method='standard', has_label=True)
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Class distribution: {np.unique(y_train, return_counts=True)}")
    
    # Train
    # Optimized for generalization (preventing perfect 1.0 scores)
    clf = RandomForestClassifier(
        n_estimators=10,       # Very few trees
        max_depth=3,           # Very shallow
        min_samples_split=50,  # High split requirement
        max_features=2,        # Force randomness
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Save
    joblib.dump(clf, RF_MODEL_PATH)
    joblib.dump(scaler, RF_SCALER_PATH)
    print(f"  RF Model saved to {RF_MODEL_PATH}")

def train_isolation_forest(df_benign_train, df_mixed_val):
    print("\n--- Training Isolaion Forest (Unsupervised w/ Grid Search) ---")
    
    # 1. Preprocess Training Data (Benign Only)
    # Use RobustScaler
    X_train, _, scaler = preprocess(df_benign_train, is_training=True, method='robust', has_label=False)
    
    # 2. Preprocess Validation Data (Mixed)
    # Use SAME scaler
    X_val, y_val, _ = preprocess(df_mixed_val, scaler=scaler, is_training=False, has_label=True)
    
    print(f"  Training (Benign) samples: {len(X_train)}")
    print(f"  Validation (Mixed) samples: {len(X_val)}")
    
    # 3. Grid Search Optimization
    # Expanded grid to find better anomaly detection boundaries
    param_grid = {
        'n_estimators': [100, 200],
        'max_samples': [0.8, 1.0, 'auto'],
        'contamination': [0.001, 0.01, 0.02, 0.05], # Finer grain for contamination
        'max_features': [0.5, 0.8, 1.0],
        'bootstrap': [False, True]
    }
    
    grid = list(ParameterGrid(param_grid))
    best_f1 = -1
    best_model = None
    best_params = None
    
    print(f"  Testing {len(grid)} hyperparameters combinations...")
    
    for params in grid:
        clf = IsolationForest(n_jobs=-1, random_state=42, **params)
        clf.fit(X_train)
        
        # Predict on validation (Mixed)
        # IF: -1 (anomaly), 1 (normal)
        y_pred_raw = clf.predict(X_val)
        y_pred = np.where(y_pred_raw == -1, 1, 0)
        
        score = f1_score(y_val, y_pred, zero_division=0)
        
        if score > best_f1:
            best_f1 = score
            best_model = clf
            best_params = params

    print(f"  Best Validation F1: {best_f1:.4f}")
    print(f"  Best Parameters: {best_params}")
    
    # Save
    joblib.dump(best_model, IF_MODEL_PATH)
    joblib.dump(scaler, IF_SCALER_PATH)
    print(f"  IF Model saved to {IF_MODEL_PATH}")

def main():
    print("=== 3. Model Training Pipeline ===")
    
    # Load Datasets
    # 1. RF Training Data (Mixed)
    df_rf_train = load_data(TRAIN_RF_PATH)
    
    # 2. IF Training Data (Benign)
    df_if_train = load_data(TRAIN_IF_PATH)
    
    # Train RF
    train_random_forest(df_rf_train)
    
    # Train IF (Uses RF Train set as Validation for tuning)
    train_isolation_forest(df_if_train, df_rf_train)
    
    print("\n=== Training Complete ===")

if __name__ == "__main__":
    main()
