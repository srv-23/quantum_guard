import pandas as pd
import joblib
import os
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_PATH = os.path.join(BASE_DIR, "dataset", "final_test.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Model Paths
RF_MODEL = os.path.join(MODELS_DIR, "random_forest.pkl")
RF_SCALER = os.path.join(MODELS_DIR, "scaler.pkl")
IF_MODEL = os.path.join(MODELS_DIR, "isolation_forest.pkl")
IF_SCALER = os.path.join(MODELS_DIR, "if_scaler.pkl")

def evaluate_rf(df):
    """Evaluate Random Forest Model."""
    if not os.path.exists(RF_MODEL):
        return None
        
    model = joblib.load(RF_MODEL)
    scaler = joblib.load(RF_SCALER)
    
    # Prepare
    drop_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'label']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = scaler.transform(df[feature_cols])
    y = df['label']
    
    # Predict
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    
    return {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1 Score': f1_score(y, y_pred, zero_division=0),
        'AUC': roc_auc_score(y, y_prob)
    }

def evaluate_if(df):
    """Evaluate Isolation Forest Model."""
    if not os.path.exists(IF_MODEL):
        return None
        
    model = joblib.load(IF_MODEL)
    scaler = joblib.load(IF_SCALER)
    
    # Prepare
    drop_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'label']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = scaler.transform(df[feature_cols])
    y = df['label']
    
    # Predict (Convert -1/1 to 1/0)
    y_pred_raw = model.predict(X) # 1=Normal, -1=Anomaly
    y_pred = np.where(y_pred_raw == 1, 0, 1)
    
    # Scores (invert for AUC because lower score = anomaly)
    y_scores = -model.decision_function(X)
    
    return {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1 Score': f1_score(y, y_pred, zero_division=0),
        'AUC': roc_auc_score(y, y_scores)
    }

def main():
    print("=== Model Comparison Started ===")
    
    if not os.path.exists(TEST_DATA_PATH):
        print("Error: Test data not found.")
        return

    print("Loading test data...")
    df = pd.read_csv(TEST_DATA_PATH)
    
    print("Evaluating Random Forest...")
    rf_metrics = evaluate_rf(df)
    
    print("Evaluating Isolation Forest...")
    if_metrics = evaluate_if(df)
    
    if not rf_metrics or not if_metrics:
        print("Error: One or both models missing.")
        return

    # Print Comparison Table
    print("\n" + "="*60)
    print(f"{'Metric':<15} | {'Random Forest':<15} | {'Isolation Forest':<15}")
    print("-" * 60)
    
    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    
    for metric in metrics_list:
        rf_val = rf_metrics.get(metric, 0)
        if_val = if_metrics.get(metric, 0)
        print(f"{metric:<15} | {rf_val:<15.4f} | {if_val:<15.4f}")
        
    print("=" * 60)

    # Interpretation
    print("\n[Interpretation]")
    if rf_metrics['F1 Score'] > if_metrics['F1 Score']:
        winner = "Random Forest"
        reason = "higher F1 score, indicating better balance between precision and recall."
    else:
        winner = "Isolation Forest"
        reason = "higher F1 score, suggesting typical anomaly detection capability."
        
    print(f"Based on the F1 Score, **{winner}** performs better.")
    print(f"Reason: It achieved a {reason}")
    print("\nNote for Research:")
    print("- Random Forest (Supervised) usually outperforms Isolation Forest (Unsupervised)")
    print("  if labeled training data is representative.")
    print("- Isolation Forest is better suited for zero-day attacks where anomalies are unknown.")
    print("="*60)

if __name__ == "__main__":
    main()
