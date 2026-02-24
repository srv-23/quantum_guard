import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from visualization_plots import (
    plot_confusion_matrix, plot_roc_curve, plot_feature_importance,
    plot_metrics_bar, plot_class_distribution, plot_precision_recall_curve,
    compare_train_test_accuracy
)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

def main():
    print("--- Step 4: Evaluation and Visualization ---")

    # Define paths
    train_processed_csv = os.path.join(DATASET_DIR, "train_processed.csv")
    test_processed_csv = os.path.join(DATASET_DIR, "test_processed.csv")
    model_path = os.path.join(MODEL_DIR, "random_forest.pkl")

    if not os.path.exists(test_processed_csv) or not os.path.exists(model_path):
        print(f"Error: Processed test data or trained model not found.")
        print("Please run steps 1-3 first.")
        return

    # Load Data
    test_df = pd.read_csv(test_processed_csv)
    train_df = pd.read_csv(train_processed_csv)
    
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]
    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    # Load Model
    model = joblib.load(model_path)
    print("Model loaded successfully.")

    # Predictions
    print("Running Predictions...")
    y_pred = model.predict(X_test)
    try:
        y_probs = model.predict_proba(X_test)[:, 1]
    except IndexError:
        y_probs = [0] * len(y_test)

    # Generate Visualizations
    print(f"Generating Plots in {PLOTS_DIR}...")
    
    # Check if we have both classes in test set for ROC
    unique_test_labels = y_test.unique()
    if len(unique_test_labels) < 2:
        print("Warning: Test set has only one class. Some plots (ROC, PR) might not be informative.")

    try:
        plot_confusion_matrix(y_test, y_pred)
        plot_roc_curve(y_test, y_probs)
        plot_feature_importance(model, X_test.columns)
        plot_metrics_bar(y_test, y_pred)
        plot_class_distribution(y_train, y_test)
        plot_precision_recall_curve(y_test, y_probs)
        compare_train_test_accuracy(model, X_train, y_train, X_test, y_test)
    except Exception as e:
        print(f"Error generating plots: {e}")

    print("--- Evaluation Complete ---")

if __name__ == "__main__":
    main()
