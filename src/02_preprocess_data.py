import os
import joblib
from sklearn.preprocessing import StandardScaler
from utils import preprocess_dataset

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

def main():
    print("--- Step 2: Data Preprocessing ---")

    # Define paths
    train_raw_csv = os.path.join(DATASET_DIR, "train_raw.csv")
    test_raw_csv = os.path.join(DATASET_DIR, "test_raw.csv")
    train_processed_csv = os.path.join(DATASET_DIR, "train_processed.csv")
    test_processed_csv = os.path.join(DATASET_DIR, "test_processed.csv")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")

    # Ensure directories exist
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Preprocess Data
    print("Processing Training Data...")
    if os.path.exists(train_raw_csv):
        train_df = preprocess_dataset(
            train_raw_csv, 
            train_processed_csv, 
            scaler_path=scaler_path, 
            fit_scaler=True, 
            inject_anomalies=True
        )
    else:
        print(f"Warning: {train_raw_csv} not found. Skipping.")

    print("Processing Testing Data...")
    if os.path.exists(test_raw_csv):
        test_df = preprocess_dataset(
            test_raw_csv, 
            test_processed_csv, 
            scaler_path=scaler_path, 
            fit_scaler=False, 
            inject_anomalies=True
        )
    else:
        print(f"Warning: {test_raw_csv} not found. Skipping.")

    print("--- Preprocessing Complete ---")

if __name__ == "__main__":
    main()
