import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "dataset", "final_train.csv")
TEST_DATA_PATH = os.path.join(BASE_DIR, "dataset", "final_test.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

os.makedirs(OUTPUTS_DIR, exist_ok=True)

def load_data_and_model():
    """Load model, scaler, and datasets."""
    print("Loading datasets and model...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Train RF first.")
        
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # We load train mainly to get feature names if not saved with model,
    # or just to ensure consistency. 
    df_train = pd.read_csv(TRAIN_DATA_PATH)
    df_test = pd.read_csv(TEST_DATA_PATH)
    
    return df_train, df_test, model, scaler

def prepare_features(df, scaler):
    """
    Separate features and scale.
    Consistent with training pipeline.
    """
    drop_cols = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'label']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = scaler.transform(df[feature_cols])
    y = df['label']
    
    return X, y, feature_cols

def plot_precision_recall(y_true, y_probs):
    """
    Generate and save Precision-Recall Curve.
    """
    print("Generating Precision-Recall Curve...")
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)
    
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(recall, precision, color='purple', lw=2, label=f'AP = {avg_precision:.2f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve (Random Forest)')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    
    output_path = os.path.join(OUTPUTS_DIR, "rf_pr_curve.png")
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")
    return avg_precision

def plot_feature_importance(model, feature_names):
    """
    Generate and save Feature Importance Plot.
    """
    print("Generating Feature Importance Plot...")
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot top 20 or all if fewer
    top_n = min(20, len(feature_names))
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(10, 8), dpi=300)
    plt.title("Feature Importance (Random Forest)")
    plt.barh(range(top_n), importances[top_indices], align="center", color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in top_indices])
    plt.xlabel("Importance Score")
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUTS_DIR, "rf_feature_importance.png")
    plt.savefig(output_path)
    plt.close()
    print(f"  Saved: {output_path}")
    
    return [feature_names[i] for i in top_indices[:5]]

def main():
    print("=== Advanced Visualizations Started ===")
    
    try:
        # 1. Load
        df_train, df_test, model, scaler = load_data_and_model()
        
        # 2. Prepare Test Data
        X_test, y_test, feature_names = prepare_features(df_test, scaler)
        
        # 3. Predict Probabilities (for PR Curve)
        y_probs = model.predict_proba(X_test)[:, 1]
        
        # 4. Generate Plots
        
        # A) Precision-Recall Curve
        ap_score = plot_precision_recall(y_test, y_probs)
        
        # B) Feature Importance
        # Note: Feature importance comes from the structure of the trees, 
        # independent of the test set, but we use feature names from data.
        top_5_features = plot_feature_importance(model, feature_names)
        
        # 5. Print Summary
        print("\n" + "="*40)
        print(" Visualization Summary")
        print("="*40)
        print(f" Average Precision Score: {ap_score:.4f}")
        print("-" * 40)
        print(" Top 5 Most Important Features:")
        for idx, feat in enumerate(top_5_features, 1):
            print(f"  {idx}. {feat}")
        print("="*40)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print("\n=== Visualization Complete ===")

if __name__ == "__main__":
    main()
