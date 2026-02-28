import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve, precision_recall_curve, auc
)

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_DIR = os.path.join(BASE_DIR, "dataset", "processed")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Files
TEST_SET_PATH = os.path.join(DATASET_DIR, "test_set.csv")
CALIBRATION_SET_PATH = os.path.join(DATASET_DIR, "train_rf.csv") # Use mixed training set for IF calibration

RF_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
RF_SCALER_PATH = os.path.join(MODELS_DIR, "rf_scaler.pkl")

IF_MODEL_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")
IF_SCALER_PATH = os.path.join(MODELS_DIR, "if_scaler.pkl")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

def load_data(path):
    print(f"Loading {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run previous steps first.")
    return pd.read_csv(path)

def preprocess(df, scaler, drop_cols=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'label']):
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    if 'label' in df.columns:
        y = df['label'].values
    else:
        y = np.zeros(len(df))
    X_scaled = scaler.transform(X)
    return X_scaled, y

def calibrate_if_threshold(model, X_val, y_val):
    """
    Find best threshold for Isolation Forest using F1 Score on a labeled validation set.
    """
    print("Calibrating Isolation Forest Threshold...")
    if y_val.sum() == 0:
        print("[WARN] No anomalies in validation set. Skipping calibration, using default 0.")
        return 0.0

    scores = model.decision_function(X_val)
    min_score, max_score = scores.min(), scores.max()
    thresholds = np.linspace(min_score, max_score, 100)
    
    best_f1 = -1
    best_thresh = 0.0
    
    for thresh in thresholds:
        # Predict 1 (Anomaly) if score < thresh
        y_pred = (scores < thresh).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    print(f"  Best F1: {best_f1:.4f} at Threshold: {best_thresh:.4f}")
    return best_thresh

def evaluate_model(name, y_true, y_pred, y_scores, color):
    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    roc_auc = 0.5
    pr_auc = 0.0
    
    if y_scores is not None:
        try:
            roc_auc = roc_auc_score(y_true, y_scores)
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall, precision)
        except:
             pass

    print(f"\n--- {name} Results ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")
    print(f"PR AUC:    {pr_auc:.4f}")
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUTS_DIR, f"{name.replace(' ', '_')}_confusion.png"))
    plt.close()
    
    return {
        'name': name,
        'y_true': y_true,
        'y_scores': y_scores,
        'color': color,
        'metrics': [acc, prec, rec, f1, roc_auc, pr_auc]
    }

def plot_curves(results):
    # ROC Curve
    plt.figure(figsize=(8, 6))
    for res in results:
        if res['y_scores'] is not None:
            fpr, tpr, _ = roc_curve(res['y_true'], res['y_scores'])
            roc_auc = res['metrics'][4]
            plt.plot(fpr, tpr, color=res['color'], label=f"{res['name']} (AUC = {roc_auc:.2f})")
            
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve Comparison')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(OUTPUTS_DIR, "comparison_roc_curve.png"))
    plt.close()
    
    # PR Curve
    plt.figure(figsize=(8, 6))
    for res in results:
        if res['y_scores'] is not None:
            precision, recall, _ = precision_recall_curve(res['y_true'], res['y_scores'])
            pr_auc = res['metrics'][5]
            plt.plot(recall, precision, color=res['color'], label=f"{res['name']} (AUC = {pr_auc:.2f})")
            
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve Comparison')
    plt.legend(loc='lower left')
    plt.savefig(os.path.join(OUTPUTS_DIR, "comparison_pr_curve.png"))
    plt.close()

def main():
    print("=== 4. Evaluate Pipeline ===")
    
    # 1. Load Models
    print("Loading models...")
    rf_model = joblib.load(RF_MODEL_PATH)
    rf_scaler = joblib.load(RF_SCALER_PATH)
    
    if_model = joblib.load(IF_MODEL_PATH)
    if_scaler = joblib.load(IF_SCALER_PATH)
    
    # 2. Load Unseen Test Data
    print("Loading test dataset...")
    df_test = load_data(TEST_SET_PATH)
    print(f"Test samples: {len(df_test)}")
    print(f"Anomaly count: {df_test['label'].sum()}")
    
    # Preprocess
    X_test_rf, y_test = preprocess(df_test, rf_scaler) # Standard Scaler
    X_test_if, _ = preprocess(df_test, if_scaler)      # Robust Scaler (Same X, just different scale)
    
    # 3. RF Evaluation
    rf_pred = rf_model.predict(X_test_rf)
    rf_probs = rf_model.predict_proba(X_test_rf)[:, 1]
    
    rf_res = evaluate_model("Random Forest", y_test, rf_pred, rf_probs, 'green')
    
    # 4. IF Calibration & Evaluation
    print("\nLoading calibration set for IF...")
    df_cal = load_data(CALIBRATION_SET_PATH)
    X_cal, y_cal = preprocess(df_cal, if_scaler) # Apply IF Scaler
    
    # Calibrate Threshold
    best_thresh = calibrate_if_threshold(if_model, X_cal, y_cal)
    
    # Predict on Test
    if_scores_raw = if_model.decision_function(X_test_if)
    # Convert scores for plotting: Negate so Higher = Anomaly
    if_scores_anomaly = -if_scores_raw
    
    # Apply calibrated threshold
    if_pred = (if_scores_raw < best_thresh).astype(int)
    
    if_res = evaluate_model("Isolation Forest", y_test, if_pred, if_scores_anomaly, 'blue')
    
    # 5. Plot Comparison
    plot_curves([rf_res, if_res])
    
    # 6. Print Final Report
    print("\n" + "="*40)
    print("FINAL PERFORMANCE REPORT")
    print("="*40)
    print(f"{'Metric':<20} {'Random Forest':<18} {'Isolation Forest':<18}")
    print("-" * 56)

    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "PR AUC"]
    
    # Extract metrics from results dictionaries
    rf_m = rf_res['metrics']
    if_m = if_res['metrics']
    
    for i, metric in enumerate(metrics):
        print(f"{metric:<20} {rf_m[i]:<18.4f} {if_m[i]:<18.4f}")
    
    print("="*40)
    
    print("\n=== Evalution Complete ===")
    print(f"Results saved to {OUTPUTS_DIR}")

if __name__ == "__main__":
    main()
