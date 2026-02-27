import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_PATH = os.path.join(BASE_DIR, "dataset", "test_mixed.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

def load_evaluation_components():
    """Load model, scaler, and test data."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or Scaler not found. Run training first.")
        
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found at {TEST_DATA_PATH}")

    print("Loading test data...")
    df_test = pd.read_csv(TEST_DATA_PATH)
    
    print("Loading model and scaler...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    return df_test, model, scaler

def prepare_test_data(df, scaler):
    """Separate features/labels and scale features."""
    # Define features same as training script (excluding identifiers)
    drop_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'label']
    
    # Check if these columns exist before dropping
    # Based on metadata_pipeline.py and 1_train_random_forest.py
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['label']
    
    # Check if scaler feature names match (if available in newer sklearn)
    # Just proceed with transform assuming columns match 1_train_random_forest.py
    X_scaled = scaler.transform(X)
    
    return X_scaled, y

def evaluate_predictions(X, y, model):
    """Predict and calculate metrics."""
    print("Predicting on test set...")
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1] # Probability for positive class (anomaly)
    
    # Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)
    
    # Classification Report
    report = classification_report(y, y_pred, zero_division=0)

    # ROC Curve Data
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'report': report,
        'fpr': fpr,
        'tpr': tpr,
        'auc': roc_auc,
        'y_pred': y_pred
    }
    return results

def plot_confusion_matrix(cm, output_path):
    """Plot confusion matrix using matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    fig.colorbar(cax)
    
    # Annotate cells
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f'{z}', ha='center', va='center', 
                color='white' if z > cm.max()/2 else 'black')
                
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0, 1], ['Benign', 'Anomaly'])
    plt.yticks([0, 1], ['Benign', 'Anomaly'])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion Matrix saved to {output_path}")

def plot_roc_curve(fpr, tpr, roc_auc, output_path):
    """Plot ROC curve using matplotlib."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.savefig(output_path)
    plt.close()
    print(f"ROC Curve saved to {output_path}")

def main():
    print("=== random_forest_evaluation.py Started ===")
    
    try:
        # Load
        df_test, model, scaler = load_evaluation_components()
        
        # Prepare
        X_test_scaled, y_test = prepare_test_data(df_test, scaler)
        
        # Evaluate
        results = evaluate_predictions(X_test_scaled, y_test, model)
        
        # Print Results
        print("\n" + "="*30)
        print(" Evaluation Metrics")
        print("="*30)
        print(f" Accuracy:  {results['accuracy']:.4f}")
        print(f" Precision: {results['precision']:.4f}")
        print(f" Recall:    {results['recall']:.4f}")
        print(f" F1 Score:  {results['f1']:.4f}")
        print(f" AUC Score: {results['auc']:.4f}")
        print("-" * 30)
        print(" Classification Report:")
        print(results['report'])
        print("-" * 30)
        print(" Confusion Matrix:")
        print(results['confusion_matrix'])
        print("="*30)
        
        # Plot
        cm_path = os.path.join(OUTPUTS_DIR, "rf_confusion.png")
        plot_confusion_matrix(results['confusion_matrix'], cm_path)
        
        roc_path = os.path.join(OUTPUTS_DIR, "rf_roc.png")
        plot_roc_curve(results['fpr'], results['tpr'], results['auc'], roc_path)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Evaluation Complete ===")

if __name__ == "__main__":
    main()
