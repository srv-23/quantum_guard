import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")

# Input files
BENIGN_PATH = os.path.join(DATASET_DIR, "train_benign.csv")
MIXED_PATH = os.path.join(DATASET_DIR, "test_mixed.csv")

# Output files
TRAIN_IF_PATH = os.path.join(PROCESSED_DIR, "train_if.csv")      # Benign only for IF
TRAIN_RF_PATH = os.path.join(PROCESSED_DIR, "train_rf.csv")      # Mixed for RF
TEST_SET_PATH = os.path.join(PROCESSED_DIR, "test_set.csv")      # Unseen Test Set (Mixed)

os.makedirs(PROCESSED_DIR, exist_ok=True)

def main():
    print("=== 2. Create Split Datasets ===\n")

    # 1. Load Raw Data
    print(f"Loading {BENIGN_PATH}...")
    if not os.path.exists(BENIGN_PATH):
        print(f"[ERROR] {BENIGN_PATH} not found. Run extract_features.py first.")
        return

    df_benign_raw = pd.read_csv(BENIGN_PATH)
    # Ensure they are labeled 0
    df_benign_raw['label'] = 0
    
    print(f"Loading {MIXED_PATH}...")
    if not os.path.exists(MIXED_PATH):
        print(f"[ERROR] {MIXED_PATH} not found. Run extract_features.py first.")
        return

    df_mixed_raw = pd.read_csv(MIXED_PATH)
    
    # 2. Separate Mixed into Benign and Malicious
    df_mixed_benign = df_mixed_raw[df_mixed_raw['label'] == 0].copy()
    df_mixed_anomaly = df_mixed_raw[df_mixed_raw['label'] == 1].copy()
    
    print(f"\nRaw Counts:")
    print(f"  Benign Source: {len(df_benign_raw)}")
    print(f"  Mixed Source (Benign/Anomaly): {len(df_mixed_benign)} / {len(df_mixed_anomaly)}")

    # 3. Create TEST SET
    # We want the Test Set to be completely unseen.
    # Let's take 50% of the Anomalies for Test.
    # Let's take 20% of the Benign Source + 50% of Mixed Benign for Test (to get variety).
    
    # Split Anomalies
    if len(df_mixed_anomaly) > 0:
        train_anom, test_anom = train_test_split(df_mixed_anomaly, test_size=0.5, random_state=42)
    else:
        train_anom = pd.DataFrame()
        test_anom = pd.DataFrame()
    
    # Split Benign Source
    # We need a lot of benign for IF training.
    train_benign_main, test_benign_main = train_test_split(df_benign_raw, test_size=0.2, random_state=42)
    
    # Split Mixed Benign
    if len(df_mixed_benign) > 0:
        train_benign_mixed, test_benign_mixed = train_test_split(df_mixed_benign, test_size=0.5, random_state=42)
    else:
        train_benign_mixed = pd.DataFrame()
        test_benign_mixed = pd.DataFrame()
    
    # 4. Construct Datasets
    
    # A) Isolation Forest Training Set
    # strictly benign data.
    df_train_if = train_benign_main.copy()
    
    # B) Random Forest Training Set
    # Needs both Benign and Malicious.
    # Benign: 'train_benign_main' (reuse benign) + 'train_benign_mixed'
    # Malicious: 'train_anom'
    df_train_rf = pd.concat([train_benign_main, train_benign_mixed, train_anom], ignore_index=True)
    
    # C) Final Test Set
    # Contains remaining Benign and Malicious.
    df_test = pd.concat([test_benign_main, test_benign_mixed, test_anom], ignore_index=True)
    
    # 5. Save
    print("\nSAVING DATASETS:")
    
    print(f"  1. IF Training Set (Benign only): {len(df_train_if)} samples")
    df_train_if.to_csv(TRAIN_IF_PATH, index=False)
    
    print(f"  2. RF Training Set (Mixed): {len(df_train_rf)} samples")
    print(f"     Distribution: {df_train_rf['label'].value_counts().to_dict()}")
    df_train_rf.to_csv(TRAIN_RF_PATH, index=False)
    
    print(f"  3. Final Test Set (Unseen): {len(df_test)} samples")
    print(f"     Distribution: {df_test['label'].value_counts().to_dict()}")
    df_test.to_csv(TEST_SET_PATH, index=False)
    
    print(f"\nData split complete. Files saved to {PROCESSED_DIR}")

if __name__ == "__main__":
    main()
