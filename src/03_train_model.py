import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")

def main():
    print("--- Step 3: Model Training ---")

    # Define paths
    train_processed_csv = os.path.join(DATASET_DIR, "train_processed.csv")
    model_path = os.path.join(MODEL_DIR, "random_forest.pkl")

    if not os.path.exists(train_processed_csv):
        print(f"Error: Processed training data not found at {train_processed_csv}.")
        print("Please run 02_preprocess_data.py first.")
        return

    # Load Data
    train_df = pd.read_csv(train_processed_csv)
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]

    # Check class balance
    if len(y_train.unique()) < 2:
        print("Warning: Training set only has one class. Adding synthetic samples for demonstration.")
        other_class = 1 if y_train.iloc[0] == 0 else 0
        X_train.iloc[-1] = X_train.iloc[0] # Duplicate features
        y_train.iloc[-1] = other_class # Flip label

    # Train Model
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    
    # Save Model
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    print("--- Training Complete ---")

if __name__ == "__main__":
    main()
