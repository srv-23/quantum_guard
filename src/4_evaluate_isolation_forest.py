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
TEST_DATA_PATH = os.path.join(BASE_DIR, "dataset", "final_test.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "if_scaler.pkl")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

def load_evaluation_components():
    """Load IF model, scaler, and test data."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or Scaler not found. Run IF training first.")
        
    if not os.path.exists(TEST_DATA_PATH):
        raise FileNotFoundError(f"Test data not found at {TEST_DATA_PATH}")

    print("Loading test data...")
    df_test = pd.read_csv(TEST_DATA_PATH)
    
    print("Loading Isolation Forest model and scaler...")
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    return df_test, model, scaler

def prepare_test_data(df, scaler):
    """Separate features/labels and scale features."""
    drop_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'label']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df['label']
    
    X_scaled = scaler.transform(X)
    
    return X_scaled, y

def evaluate_predictions(X, y, model):
    """
    Predict using Isolation Forest and convert labels.
    IF Output:  1 = Normal, -1 = Anomaly
    Our Labels: 0 = Normal,  1 = Anomaly
    
    Conversion:
        IF(1)  -> 0
        IF(-1) -> 1
    """
    print("Predicting on test set...")
    
    # 1. Get raw predictions (1/-1)
    y_pred_raw = model.predict(X)
    
    # 2. Convert to 0/1 (0=Benign, 1=Anomaly)
    # y_pred_raw == 1  => 0 (Normal)
    # y_pred_raw == -1 => 1 (Anomaly)
    y_pred = np.where(y_pred_raw == 1, 0, 1)
    
    # 3. Get Anomaly Scores (for ROC/AUC)
    # decision_function: positive for inliers, negative for outliers
    # We want higher score = more anomalous for ROC
    # So we negate the decision_function
    y_scores = -model.decision_function(X) 
    
    # Metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred)
    
    # ROC Curve Data
    fpr, tpr, thresholds = roc_curve(y, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Classification Report
    report = classification_report(y, y_pred, zero_division=0)
    
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
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Oranges)
    plt.title('Isolation Forest Confusion Matrix')
    fig.colorbar(cax)
    
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f'{z}', ha='center', va='center')
                
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0, 1], ['Benign (0)', 'Anomaly (1)'])
    plt.yticks([0, 1], ['Benign (0)', 'Anomaly (1)'])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion Matrix saved to {output_path}")

def plot_roc_curve(fpr, tpr, roc_auc, output_path):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkred', lw=2, label=f'IF ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Isolation Forest ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    
    plt.savefig(output_path)
    plt.close()
    print(f"ROC Curve saved to {output_path}")

def main():
    print("=== isolation_forest_evaluation.py Started ===")
    
    try:
        # Load
        df_test, model, scaler = load_evaluation_components()
        
        # Prepare
        X_test_scaled, y_test = prepare_test_data(df_test, scaler)
        
        # Evaluate
        results = evaluate_predictions(X_test_scaled, y_test, model)
        
        # Print Results
        print("\n" + "="*30)
        print(" IF Evaluation Metrics")
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
        cm_path = os.path.join(OUTPUTS_DIR, "if_confusion.png")
        plot_confusion_matrix(results['confusion_matrix'], cm_path)
        
        roc_path = os.path.join(OUTPUTS_DIR, "if_roc.png")
        plot_roc_curve(results['fpr'], results['tpr'], results['auc'], roc_path)
        
    except Exception as e:
        print(f"Error during IF evaluation: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Evaluation Complete ===")

if __name__ == "__main__":
    main()
